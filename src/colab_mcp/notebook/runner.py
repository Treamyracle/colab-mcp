"""
Notebook Execution Engine & FastMCP Tool 9 (``colab_execute_notebook``).

Implements TECH_SPEC §6.2–6.4, §7.3.3 Tool 9, §8.1, §8.3, and §8.4:
- Optional pre-run kernel restart (``reset_kernel_before_run``)
- Sequential and selective 0-based ``cell_indices`` execution
- Bypassing markdown/raw cells while preserving notebook structure
- ``stop_on_error`` halt/continue policies with ANSI-sanitized & redacted diagnostics
- ``nbformat.v4`` AST output serialization and atomic disk writes
- Dual support for real ``KernelWebSocketClient.execute_raw`` and mocked ``client.execute``
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile
import time
import traceback as tb_module
from pathlib import Path
from typing import Annotated, Any, Awaitable, Callable, Dict, List, Literal, Optional, Sequence

import nbformat
import nbformat.v4
import nbformat.validator
from mcp.server.mcpserver import MCPServer
from mcp.types import ImageContent, TextContent
from pydantic import BaseModel, BeforeValidator, Field, StrictBool, StrictInt

from colab_mcp.kernel.client import KernelWebSocketClient
from colab_mcp.kernel.mime_converter import (
    MCPContentBlock,
    sanitize_ansi_text,
    sanitize_ansi_traceback,
)
from colab_mcp.kernel.models import (
    KernelConnectionError,
    KernelExecutionTimeoutError,
    RawExecutionResult,
)
from colab_mcp.kernel.tools import (
    _is_configured_mock_method,
    _record_session_activity,
    colab_execute_code as _kernel_colab_execute_code,
    colab_get_cell_output as _kernel_colab_get_cell_output,
    colab_get_execution_status as _kernel_colab_get_execution_status,
    colab_interrupt_execution as _kernel_colab_interrupt_execution,
    get_active_kernel_client,
    get_execution_registry,
)
from colab_mcp.notebook.loader import load_and_validate_notebook, validate_workspace_path
from colab_mcp.utils.logging import get_logger
from colab_mcp.utils.redaction import redact_credentials, redact_data

logger = get_logger(__name__)


# ============================================================================
# Response Models (TECH_SPEC §7.3.3 Tool 9)
# ============================================================================


class _DictCompatMixin:
    """Provides dict-like .get(), __getitem__(), and __contains__() for Pydantic models."""

    def get(self, key: str, default: Any = None) -> Any:
        data = self.model_dump()  # type: ignore[attr-defined]
        return data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        data = self.model_dump()  # type: ignore[attr-defined]
        if key not in data:
            raise KeyError(key)
        return data[key]

    def __contains__(self, key: object) -> bool:
        data = self.model_dump()  # type: ignore[attr-defined]
        return key in data


class CellExecutionSummary(_DictCompatMixin, BaseModel):
    """Per-cell execution summary entry in ``NotebookExecutionResponse``."""

    cell_index: int
    execution_count: Optional[int] = None
    status: Literal["success", "failed", "skipped"]
    duration_seconds: float = 0.0
    has_plots: bool = False
    plot_count: int = 0
    error_name: Optional[str] = None
    error_value: Optional[str] = None
    traceback: Optional[List[str]] = None


class NotebookExecutionResponse(_DictCompatMixin, BaseModel):
    """Structured response payload for ``colab_execute_notebook`` (TECH_SPEC §7.3.3)."""

    total_cells: int
    executed_cells: int
    successful_cells: int
    failed_cells: int
    skipped_cells: int
    total_duration_seconds: float
    kernel_reset_performed: bool
    cell_summaries: List[CellExecutionSummary] = Field(default_factory=list)
    saved_to: Optional[str] = None


# ============================================================================
# Validation & Serialization Helpers
# ============================================================================


def _validate_bool_param(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


def _validate_timeout_seconds(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("timeout_per_cell_seconds must be an integer.")
    if not (5 <= value <= 14400):
        raise ValueError("timeout_per_cell_seconds must be between 5 and 14400.")
    return value


def _validate_cell_indices_types(cell_indices: Any) -> Optional[List[int]]:
    if cell_indices is None:
        return None
    if isinstance(cell_indices, (str, bytes)) or not isinstance(cell_indices, Sequence):
        raise ValueError("cell_indices must be a list of non-negative integers.")
    validated: List[int] = []
    for idx in cell_indices:
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise ValueError("All elements of cell_indices must be integers.")
        if idx < 0:
            raise ValueError(f"cell_indices cannot contain negative index {idx}.")
        validated.append(idx)
    return list(dict.fromkeys(validated))


def _validate_cell_indices_bounds(
    cell_indices: Optional[List[int]],
    num_cells: int,
) -> Optional[List[int]]:
    if cell_indices is None:
        return None
    for idx in cell_indices:
        if idx < 0 or idx >= num_cells:
            raise ValueError(
                f"cell_indices index {idx} is out of bounds for notebook with {num_cells} cells."
            )
    return cell_indices


_BINARY_IMAGE_MIMES = frozenset({
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
})


def _sanitize_text(text: str) -> str:
    return redact_credentials(sanitize_ansi_text(text))


def _sanitize_traceback_lines(lines: Sequence[Any]) -> List[str]:
    raw_lines = [str(line) for line in lines]
    if not raw_lines:
        return []
    sep = "\x00__TB_SEP__\x00"
    joined_check = "".join(raw_lines)
    while sep in joined_check:
        sep = f"\x00__TB_SEP_{len(sep)}__\x00"
    combined = sep.join(raw_lines)
    cleaned = redact_credentials(sanitize_ansi_text(combined))
    return [
        redact_credentials(sanitize_ansi_traceback(part))
        for part in cleaned.split(sep)
    ]


def _sanitize_mime_data(raw_data: Any) -> Dict[str, Any]:
    if not isinstance(raw_data, dict):
        return {}
    sanitized: Dict[str, Any] = {}
    for mime, value in raw_data.items():
        mime_str = str(mime)
        if mime_str.lower() in _BINARY_IMAGE_MIMES:
            sanitized[mime_str] = value
        elif isinstance(value, str):
            sanitized[mime_str] = _sanitize_text(value)
        elif isinstance(value, list):
            sanitized[mime_str] = [
                _sanitize_text(item) if isinstance(item, str) else redact_data(item)
                for item in value
            ]
        else:
            sanitized[mime_str] = redact_data(value)
    return sanitized


def _extract_cell_source(cell: Any) -> str:
    raw_source = cell.get("source", "") if isinstance(cell, dict) else getattr(cell, "source", "")
    if isinstance(raw_source, list):
        return "".join(str(part) for part in raw_source)
    return str(raw_source or "")


def _count_image_plots(data: Dict[str, Any]) -> int:
    for mime, val in data.items():
        if str(mime).lower().startswith("image/"):
            if isinstance(val, str) and bool(val.strip()):
                return 1
            if isinstance(val, list) and any(
                isinstance(part, str) and bool(part.strip()) for part in val
            ):
                return 1
    return 0


def _finalize_stream_outputs(
    outputs: List[nbformat.NotebookNode],
) -> List[nbformat.NotebookNode]:
    """Sanitize coalesced stream outputs and drop nodes that become empty."""
    finalized: List[nbformat.NotebookNode] = []
    for node in outputs:
        if node.get("output_type") == "stream":
            raw_text = node.get("text", "")
            if isinstance(raw_text, list):
                raw_text = "".join(str(part) for part in raw_text)
            cleaned = _sanitize_text(str(raw_text))
            if not cleaned:
                continue
            node["text"] = cleaned
        finalized.append(node)
    return finalized


def _sanitize_notebook_outputs(nb: nbformat.NotebookNode) -> None:
    """Defense-in-depth pass ensuring all cell outputs in ``nb`` are ANSI-sanitized and redacted."""
    for cell in getattr(nb, "cells", []) or []:
        cell_outputs = cell.get("outputs") if isinstance(cell, dict) else getattr(cell, "outputs", None)
        if not isinstance(cell_outputs, list):
            continue
        for output in cell_outputs:
            if not isinstance(output, dict):
                continue
            otype = output.get("output_type")
            if otype == "stream":
                raw_text = output.get("text", "")
                if isinstance(raw_text, list):
                    output["text"] = _sanitize_text("".join(str(part) for part in raw_text))
                else:
                    output["text"] = _sanitize_text(str(raw_text))
            elif otype in ("display_data", "execute_result"):
                if "data" in output:
                    output["data"] = _sanitize_mime_data(output.get("data"))
                if isinstance(output.get("metadata"), dict):
                    output["metadata"] = redact_data(output["metadata"])
            elif otype == "error":
                output["ename"] = _sanitize_text(str(output.get("ename", "Error")))
                output["evalue"] = _sanitize_text(str(output.get("evalue", "")))
                raw_tb = output.get("traceback")
                if isinstance(raw_tb, list):
                    output["traceback"] = _sanitize_traceback_lines(raw_tb)


def _redact_exception(exc: Exception) -> Exception:
    """Return an exception of the same class (where safe) with credentials redacted from its message."""
    if isinstance(exc, json.JSONDecodeError):
        safe_jmsg = redact_credentials(str(exc.msg))
        safe_doc = redact_credentials(str(exc.doc))
        if safe_jmsg == exc.msg and safe_doc == exc.doc:
            return exc
        try:
            return type(exc)(safe_jmsg, safe_doc, exc.pos)
        except Exception:
            return json.JSONDecodeError(safe_jmsg, safe_doc, exc.pos)

    raw_msg = str(exc)
    safe_msg = redact_credentials(raw_msg)

    if isinstance(exc, nbformat.validator.NotebookValidationError):
        orig = getattr(exc, "original", None)
        orig_msg = str(getattr(orig, "message", raw_msg)) if orig is not None else raw_msg
        safe_orig_msg = redact_credentials(orig_msg)
        raw_ref = getattr(exc, "ref", None)
        safe_ref = redact_credentials(str(raw_ref)) if raw_ref is not None else None
        orig_instance = getattr(orig, "instance", None) if orig is not None else None
        safe_instance = redact_data(orig_instance) if orig is not None else None
        try:
            instance_changed = safe_instance != orig_instance
        except Exception:
            instance_changed = True
        if (
            safe_msg == raw_msg
            and safe_orig_msg == orig_msg
            and (raw_ref is None or safe_ref == raw_ref)
            and not instance_changed
        ):
            return exc
        if orig is not None:
            try:
                kwargs: Dict[str, Any] = {}
                for attr, kw in (
                    ("validator", "validator"),
                    ("relative_path", "path"),
                    ("validator_value", "validator_value"),
                    ("instance", "instance"),
                    ("schema", "schema"),
                    ("relative_schema_path", "schema_path"),
                ):
                    if hasattr(orig, attr):
                        val = getattr(orig, attr)
                        if attr in ("relative_path", "relative_schema_path"):
                            val = list(val)
                        kwargs[kw] = redact_data(val)
                safe_orig = nbformat.validator.ValidationError(safe_orig_msg, **kwargs)
                wrapped = nbformat.validator.NotebookValidationError(
                    safe_orig, ref=safe_ref
                )
                wrapped_str = str(wrapped)
                if redact_credentials(wrapped_str) == wrapped_str:
                    return wrapped
            except Exception:
                pass
        return nbformat.validator.ValidationError(safe_msg)

    if safe_msg == raw_msg:
        return exc
    if isinstance(exc, OSError):
        if exc.errno is not None:
            safe_strerror = redact_credentials(str(exc.strerror or ""))
            safe_fn = (
                redact_credentials(str(exc.filename))
                if exc.filename is not None
                else None
            )
            safe_fn2 = (
                redact_credentials(str(exc.filename2))
                if getattr(exc, "filename2", None) is not None
                else None
            )
            winerror = getattr(exc, "winerror", None)
            try:
                if safe_fn2 is not None:
                    return type(exc)(exc.errno, safe_strerror, safe_fn, winerror, safe_fn2)
                if winerror is not None:
                    return type(exc)(exc.errno, safe_strerror, safe_fn, winerror)
                return type(exc)(exc.errno, safe_strerror, safe_fn)
            except Exception:
                pass
        try:
            return type(exc)(safe_msg)
        except Exception:
            return OSError(safe_msg)
    try:
        return type(exc)(safe_msg)
    except Exception:
        if isinstance(exc, ValueError):
            return ValueError(safe_msg)
        return RuntimeError(safe_msg)


def _convert_iopub_to_nb_outputs(
    raw: RawExecutionResult,
    fallback_exec_count: int,
) -> tuple[List[nbformat.NotebookNode], int, Optional[str], Optional[str], Optional[List[str]]]:
    """
    Convert ``RawExecutionResult.iopub_messages`` into ``nbformat.v4`` output nodes,
    counting image plots and extracting sanitized error diagnostics if present.
    """
    outputs: List[nbformat.NotebookNode] = []
    plot_count = 0
    error_name: Optional[str] = None
    error_value: Optional[str] = None
    error_traceback: Optional[List[str]] = None

    for msg in raw.iopub_messages:
        if not isinstance(msg, dict):
            continue
        header = msg.get("header") if isinstance(msg.get("header"), dict) else {}
        msg_type = header.get("msg_type") or msg.get("msg_type")
        content = msg.get("content") if isinstance(msg.get("content"), dict) else {}

        if msg_type == "stream":
            raw_name = str(content.get("name", "stdout")).lower()
            stream_name = raw_name if raw_name in ("stdout", "stderr") else "stdout"
            raw_text = str(content.get("text", ""))
            if not raw_text:
                continue
            if (
                outputs
                and outputs[-1].get("output_type") == "stream"
                and outputs[-1].get("name") == stream_name
            ):
                outputs[-1]["text"] = f"{outputs[-1]['text']}{raw_text}"
            else:
                outputs.append(
                    nbformat.v4.new_output("stream", name=stream_name, text=raw_text)
                )

        elif msg_type == "display_data":
            data = _sanitize_mime_data(content.get("data"))
            metadata = content.get("metadata") if isinstance(content.get("metadata"), dict) else {}
            plot_count += _count_image_plots(data)
            outputs.append(
                nbformat.v4.new_output(
                    "display_data",
                    data=data,
                    metadata=redact_data(metadata),
                )
            )

        elif msg_type == "execute_result":
            data = _sanitize_mime_data(content.get("data"))
            metadata = content.get("metadata") if isinstance(content.get("metadata"), dict) else {}
            plot_count += _count_image_plots(data)
            raw_ec = content.get("execution_count")
            exec_count = int(raw_ec) if raw_ec is not None else (
                int(raw.execution_count) if raw.execution_count is not None else fallback_exec_count
            )
            outputs.append(
                nbformat.v4.new_output(
                    "execute_result",
                    execution_count=exec_count,
                    data=data,
                    metadata=redact_data(metadata),
                )
            )

        elif msg_type == "error":
            ename = _sanitize_text(str(content.get("ename", "Error")))
            evalue = _sanitize_text(str(content.get("evalue", "")))
            raw_tb = content.get("traceback")
            if isinstance(raw_tb, list) and raw_tb:
                tb_lines = _sanitize_traceback_lines(raw_tb)
            else:
                tb_lines = [f"{ename}: {evalue}" if evalue else ename]
            error_name = ename
            error_value = evalue
            error_traceback = tb_lines
            outputs.append(
                nbformat.v4.new_output(
                    "error",
                    ename=ename,
                    evalue=evalue,
                    traceback=tb_lines,
                )
            )

    if (raw.has_error or raw.status in ("error", "abort", "timed_out", "interrupted")) and error_name is None:
        error_name = _sanitize_text(str(raw.error_name or "Error"))
        error_value = _sanitize_text(str(raw.error_value or ""))
        if isinstance(raw.traceback, list) and raw.traceback:
            error_traceback = _sanitize_traceback_lines(raw.traceback)
        else:
            error_traceback = [
                f"{error_name}: {error_value}" if error_value else error_name
            ]
        outputs.append(
            nbformat.v4.new_output(
                "error",
                ename=error_name,
                evalue=error_value,
                traceback=error_traceback,
            )
        )

    outputs = _finalize_stream_outputs(outputs)
    return outputs, plot_count, error_name, error_value, error_traceback


def _convert_mcp_blocks_to_nb_outputs(
    blocks: Sequence[Any],
) -> tuple[List[nbformat.NotebookNode], int]:
    """Convert mock ``client.execute`` MCP blocks into ``nbformat.v4`` output nodes."""
    outputs: List[nbformat.NotebookNode] = []
    plot_count = 0

    for block in blocks:
        if isinstance(block, TextContent) or (
            getattr(block, "type", None) == "text" and hasattr(block, "text")
        ):
            raw_text = str(getattr(block, "text", ""))
            if not raw_text:
                continue
            if (
                outputs
                and outputs[-1].get("output_type") == "stream"
                and outputs[-1].get("name") == "stdout"
            ):
                outputs[-1]["text"] = f"{outputs[-1]['text']}{raw_text}"
            else:
                outputs.append(
                    nbformat.v4.new_output("stream", name="stdout", text=raw_text)
                )
        elif isinstance(block, ImageContent) or (
            getattr(block, "type", None) == "image" and hasattr(block, "data")
        ):
            mime = str(
                getattr(block, "mime_type", None)
                or getattr(block, "mimeType", None)
                or "image/png"
            )
            img_data = str(getattr(block, "data", "") or "")
            sanitized_data = _sanitize_mime_data({mime: img_data})
            plot_count += _count_image_plots(sanitized_data)
            outputs.append(
                nbformat.v4.new_output(
                    "display_data",
                    data=sanitized_data,
                    metadata={},
                )
            )

    outputs = _finalize_stream_outputs(outputs)
    return outputs, plot_count


def _write_notebook_atomically(nb: nbformat.NotebookNode, target: Path) -> str:
    """Sanitize, validate, and atomically write ``nb`` to ``target`` via temp file + ``os.replace``."""
    _sanitize_notebook_outputs(nb)
    nbformat.validator.validate(nb)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target.parent),
            prefix=f".{target.stem}_",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            nbformat.write(nb, tmp_file)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, target)
        return str(target)
    except Exception:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


# ============================================================================
# NotebookRunner
# ============================================================================


class NotebookRunner:
    """
    Orchestrates sequential or selective 0-based ``nbformat`` v4 notebook execution
    against a connected Colab Jupyter kernel client.
    """

    def __init__(
        self,
        client: Any = None,
        runtime_id: Optional[str] = None,
    ) -> None:
        self.client = client
        self.runtime_id = runtime_id

    async def run(
        self,
        nb: nbformat.NotebookNode,
        *,
        reset_kernel_before_run: bool = True,
        cell_indices: Optional[List[int]] = None,
        stop_on_error: bool = True,
        timeout_per_cell_seconds: int = 300,
        output_save_path: Optional[str] = None,
        workspace_root: Optional[Path] = None,
    ) -> Dict[str, Any]:
        if nb is None or not hasattr(nb, "cells"):
            raise ValueError("nb must be a valid nbformat.NotebookNode.")

        _validate_bool_param(reset_kernel_before_run, "reset_kernel_before_run")
        _validate_bool_param(stop_on_error, "stop_on_error")
        _validate_timeout_seconds(timeout_per_cell_seconds)

        total_cells = len(nb.cells)
        parsed_indices = _validate_cell_indices_types(cell_indices)
        target_indices = _validate_cell_indices_bounds(parsed_indices, total_cells)
        if target_indices is None:
            target_indices = list(range(total_cells))

        resolved_output_path: Optional[Path] = None
        if output_save_path is not None:
            try:
                if not isinstance(output_save_path, str) or not output_save_path.strip():
                    raise ValueError("output_save_path must be a non-empty string.")
                if "\x00" in output_save_path:
                    raise ValueError("output_save_path must not contain null bytes.")
                if ".." in Path(output_save_path).parts:
                    raise ValueError(
                        f"Path traversal ('..') is not permitted in output_save_path: '{output_save_path}'."
                    )
                if workspace_root is not None:
                    resolved_output_path = validate_workspace_path(
                        output_save_path, workspace_root=workspace_root
                    )
                else:
                    resolved_output_path = Path(output_save_path).expanduser().resolve()
            except Exception as exc:
                redacted_exc = _redact_exception(exc)
                if redacted_exc is exc:
                    raise
                raise redacted_exc from None

        try:
            client = (
                self.client
                if self.client is not None
                else await get_active_kernel_client(self.runtime_id)
            )
        except Exception as exc:
            redacted_exc = _redact_exception(exc)
            if redacted_exc is exc:
                raise
            raise redacted_exc from None

        effective_runtime = (
            getattr(client, "runtime_id", None)
            or getattr(client, "endpoint", None)
            or self.runtime_id
        )
        _record_session_activity(effective_runtime)

        kernel_reset_performed = False
        if reset_kernel_before_run:
            restart_fn = getattr(client, "restart", None)
            if callable(restart_fn):
                try:
                    restart_res = restart_fn()
                    if inspect.isawaitable(restart_res):
                        await restart_res
                except Exception as exc:
                    redacted_exc = _redact_exception(exc)
                    if redacted_exc is exc:
                        raise
                    raise redacted_exc from None
                kernel_reset_performed = True

        summary_by_index: Dict[int, Dict[str, Any]] = {
            i: {
                "cell_index": i,
                "execution_count": None,
                "status": "skipped",
                "duration_seconds": 0.0,
                "has_plots": False,
                "plot_count": 0,
            }
            for i in range(total_cells)
        }

        use_execute_raw = isinstance(
            client, KernelWebSocketClient
        ) or _is_configured_mock_method(client, "execute_raw")

        run_start = time.monotonic()
        next_fallback_exec_count = 1
        halted_on_error = False
        successful_cells = 0
        failed_cells = 0

        for idx in target_indices:
            cell = nb.cells[idx]
            if getattr(cell, "cell_type", None) != "code":
                continue
            if halted_on_error:
                continue

            source = _extract_cell_source(cell)
            cell_start = time.monotonic()

            if use_execute_raw:
                try:
                    raw = await client.execute_raw(
                        source,
                        timeout_seconds=timeout_per_cell_seconds,
                        stop_on_error=stop_on_error,
                    )
                    (
                        cell_outputs,
                        plot_count,
                        err_name,
                        err_value,
                        err_tb,
                    ) = _convert_iopub_to_nb_outputs(raw, next_fallback_exec_count)

                    cell_exec_count = (
                        int(raw.execution_count)
                        if raw.execution_count is not None
                        else next_fallback_exec_count
                    )
                    next_fallback_exec_count = max(
                        next_fallback_exec_count + 1, cell_exec_count + 1
                    )
                    cell.execution_count = cell_exec_count
                    cell.outputs = cell_outputs

                    elapsed = (
                        round(float(raw.duration_seconds), 4)
                        if raw.duration_seconds and raw.duration_seconds > 0
                        else round(max(0.0, time.monotonic() - cell_start), 4)
                    )

                    cell_failed = bool(
                        raw.has_error
                        or raw.status in ("error", "abort", "timed_out", "interrupted")
                    )
                    if cell_failed:
                        failed_cells += 1
                        summary_by_index[idx] = {
                            "cell_index": idx,
                            "execution_count": cell_exec_count,
                            "status": "failed",
                            "duration_seconds": elapsed,
                            "has_plots": plot_count > 0,
                            "plot_count": plot_count,
                            "error_name": err_name or "Error",
                            "error_value": err_value or "",
                            "traceback": err_tb or [],
                        }
                        if stop_on_error:
                            halted_on_error = True
                    else:
                        successful_cells += 1
                        summary_by_index[idx] = {
                            "cell_index": idx,
                            "execution_count": cell_exec_count,
                            "status": "success",
                            "duration_seconds": elapsed,
                            "has_plots": plot_count > 0,
                            "plot_count": plot_count,
                        }
                except KernelConnectionError as exc:
                    redacted_exc = _redact_exception(exc)
                    if redacted_exc is exc:
                        raise
                    raise redacted_exc from None
                except Exception as exc:
                    elapsed = round(max(0.0, time.monotonic() - cell_start), 4)
                    cell_exec_count = next_fallback_exec_count
                    next_fallback_exec_count += 1
                    err_name = type(exc).__name__
                    err_value = _sanitize_text(str(exc))
                    if isinstance(exc, KernelExecutionTimeoutError):
                        tb_lines = [f"{err_name}: {err_value}"]
                    else:
                        raw_tb = tb_module.format_exception(
                            type(exc), exc, exc.__traceback__
                        )
                        tb_lines = _sanitize_traceback_lines(raw_tb)
                    cell.execution_count = cell_exec_count
                    cell.outputs = [
                        nbformat.v4.new_output(
                            "error",
                            ename=err_name,
                            evalue=err_value,
                            traceback=tb_lines,
                        )
                    ]
                    failed_cells += 1
                    summary_by_index[idx] = {
                        "cell_index": idx,
                        "execution_count": cell_exec_count,
                        "status": "failed",
                        "duration_seconds": elapsed,
                        "has_plots": False,
                        "plot_count": 0,
                        "error_name": err_name,
                        "error_value": err_value,
                        "traceback": tb_lines,
                    }
                    if stop_on_error:
                        halted_on_error = True
            else:
                try:
                    execute_fn = client.execute
                    accepts_timeout = False
                    try:
                        sig = inspect.signature(execute_fn)
                        timeout_param = sig.parameters.get("timeout_seconds")
                        if (
                            timeout_param is not None
                            and timeout_param.kind != inspect.Parameter.POSITIONAL_ONLY
                        ) or any(
                            p.kind == inspect.Parameter.VAR_KEYWORD
                            for p in sig.parameters.values()
                        ):
                            accepts_timeout = True
                    except (ValueError, TypeError):
                        accepts_timeout = False

                    if accepts_timeout:
                        res = execute_fn(
                            source,
                            timeout_seconds=timeout_per_cell_seconds,
                        )
                    else:
                        res = execute_fn(source)
                    blocks = await res if inspect.isawaitable(res) else res

                    cell_exec_count = next_fallback_exec_count
                    next_fallback_exec_count += 1
                    cell_outputs, plot_count = _convert_mcp_blocks_to_nb_outputs(
                        list(blocks or [])
                    )
                    cell.execution_count = cell_exec_count
                    cell.outputs = cell_outputs
                    elapsed = round(max(0.0, time.monotonic() - cell_start), 4)

                    successful_cells += 1
                    summary_by_index[idx] = {
                        "cell_index": idx,
                        "execution_count": cell_exec_count,
                        "status": "success",
                        "duration_seconds": elapsed,
                        "has_plots": plot_count > 0,
                        "plot_count": plot_count,
                    }
                except KernelConnectionError as exc:
                    redacted_exc = _redact_exception(exc)
                    if redacted_exc is exc:
                        raise
                    raise redacted_exc from None
                except Exception as exc:
                    elapsed = round(max(0.0, time.monotonic() - cell_start), 4)
                    cell_exec_count = next_fallback_exec_count
                    next_fallback_exec_count += 1
                    err_name = type(exc).__name__
                    err_value = _sanitize_text(str(exc))
                    raw_tb = tb_module.format_exception(
                        type(exc), exc, exc.__traceback__
                    )
                    tb_lines = _sanitize_traceback_lines(raw_tb)
                    cell.execution_count = cell_exec_count
                    cell.outputs = [
                        nbformat.v4.new_output(
                            "error",
                            ename=err_name,
                            evalue=err_value,
                            traceback=tb_lines,
                        )
                    ]
                    failed_cells += 1
                    summary_by_index[idx] = {
                        "cell_index": idx,
                        "execution_count": cell_exec_count,
                        "status": "failed",
                        "duration_seconds": elapsed,
                        "has_plots": False,
                        "plot_count": 0,
                        "error_name": err_name,
                        "error_value": err_value,
                        "traceback": tb_lines,
                    }
                    if stop_on_error:
                        halted_on_error = True

        saved_to: Optional[str] = None
        if resolved_output_path is not None:
            _sanitize_notebook_outputs(nb)
            try:
                saved_to = _write_notebook_atomically(nb, resolved_output_path)
            except Exception as exc:
                redacted_exc = _redact_exception(exc)
                if redacted_exc is exc:
                    raise
                raise redacted_exc from None

        executed_cells = successful_cells + failed_cells
        skipped_cells = total_cells - executed_cells
        total_duration = round(max(0.0, time.monotonic() - run_start), 4)

        result: Dict[str, Any] = {
            "total_cells": total_cells,
            "executed_cells": executed_cells,
            "successful_cells": successful_cells,
            "failed_cells": failed_cells,
            "skipped_cells": skipped_cells,
            "total_duration_seconds": total_duration,
            "kernel_reset_performed": kernel_reset_performed,
            "cell_summaries": [summary_by_index[i] for i in range(total_cells)],
            "saved_to": saved_to,
        }
        return redact_data(result)


NotebookEngine = NotebookRunner


# ============================================================================
# Core Tool Implementation & FastMCP Registration (Tool 9)
# ============================================================================


async def execute_notebook_impl(
    notebook_path: Optional[str] = None,
    notebook_content: Optional[str] = None,
    reset_kernel_before_run: bool = True,
    cell_indices: Optional[List[int]] = None,
    stop_on_error: bool = True,
    timeout_per_cell_seconds: int = 300,
    output_save_path: Optional[str] = None,
    runtime_id: Optional[str] = None,
    *,
    _client_getter: Optional[Callable[[Optional[str]], Awaitable[Any]]] = None,
) -> NotebookExecutionResponse:
    """
    Core implementation for ``colab_execute_notebook`` (Tool 9).

    Performs all parameter validation, workspace boundary checks, notebook parsing/v4
    validation, and ``cell_indices`` bounds checks BEFORE acquiring a kernel client.
    """
    try:
        _validate_bool_param(reset_kernel_before_run, "reset_kernel_before_run")
        _validate_bool_param(stop_on_error, "stop_on_error")
        _validate_timeout_seconds(timeout_per_cell_seconds)
        if runtime_id is not None and not isinstance(runtime_id, str):
            raise ValueError("runtime_id must be a string or None.")
        parsed_indices = _validate_cell_indices_types(cell_indices)

        if notebook_path is None and notebook_content is None:
            raise ValueError("Either notebook_path or notebook_content must be provided.")
        if notebook_path is not None and notebook_content is not None:
            raise ValueError("Provide either notebook_path OR notebook_content, not both.")

        workspace_root = Path.cwd()
        if output_save_path is not None:
            validate_workspace_path(output_save_path, workspace_root=workspace_root)

        nb = load_and_validate_notebook(
            notebook_path=notebook_path,
            notebook_content=notebook_content,
            workspace_root=workspace_root,
        )
        validated_indices = _validate_cell_indices_bounds(parsed_indices, len(nb.cells))

        client_getter = _client_getter or get_active_kernel_client
        client = await client_getter(runtime_id)

        runner = NotebookRunner(client=client, runtime_id=runtime_id)
        summary = await runner.run(
            nb,
            reset_kernel_before_run=reset_kernel_before_run,
            cell_indices=validated_indices,
            stop_on_error=stop_on_error,
            timeout_per_cell_seconds=timeout_per_cell_seconds,
            output_save_path=output_save_path,
            workspace_root=workspace_root,
        )
        return NotebookExecutionResponse(**redact_data(summary))
    except Exception as exc:
        redacted_exc = _redact_exception(exc)
        if redacted_exc is exc:
            raise
        raise redacted_exc from None


async def colab_execute_notebook(
    notebook_path: Optional[str] = None,
    notebook_content: Optional[str] = None,
    reset_kernel_before_run: bool = True,
    cell_indices: Optional[List[int]] = None,
    stop_on_error: bool = True,
    timeout_per_cell_seconds: int = 300,
    output_save_path: Optional[str] = None,
    runtime_id: Optional[str] = None,
    *,
    _client_getter: Optional[Callable[[Optional[str]], Awaitable[Any]]] = None,
) -> NotebookExecutionResponse:
    """Execute a Jupyter notebook (``nbformat`` v4) cell-by-cell on the connected Colab kernel."""
    return await execute_notebook_impl(
        notebook_path=notebook_path,
        notebook_content=notebook_content,
        reset_kernel_before_run=reset_kernel_before_run,
        cell_indices=cell_indices,
        stop_on_error=stop_on_error,
        timeout_per_cell_seconds=timeout_per_cell_seconds,
        output_save_path=output_save_path,
        runtime_id=runtime_id,
        _client_getter=_client_getter,
    )


def _coerce_pre_parsed_json_str(value: Any) -> Any:
    """
    Reverse MCPServer's ``pre_parse_json`` coercion when a JSON notebook string
    was eagerly parsed into a ``dict``/``list`` for an ``Optional[str]`` field.

    Note: ``mcp.server.mcpserver.utilities.func_metadata.FuncMetadata.pre_parse_json``
    unconditionally evaluates ``json.loads("null") -> None`` for any ``Optional[str]``
    annotation before Pydantic validators run, so literal ``"null"`` strings arrive as
    ``None`` here without monkeypatching MCPServer internals.
    """
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return value


def register_notebook_tools(mcp: MCPServer) -> None:
    """Register Tool 9 (``colab_execute_notebook``) on the MCPServer."""

    @mcp.tool(
        name="colab_execute_notebook",
        description=(
            "Load, validate, and execute an .ipynb notebook (nbformat v4) cell-by-cell on "
            "the connected Colab kernel. Supports selective 0-based cell_indices execution, "
            "optional pre-run kernel restart, stop_on_error fault policies, and atomic disk "
            "serialization of executed outputs to output_save_path."
        ),
    )
    async def _tool_colab_execute_notebook(
        notebook_path: Optional[str] = None,
        notebook_content: Annotated[
            Optional[str], BeforeValidator(_coerce_pre_parsed_json_str)
        ] = None,
        reset_kernel_before_run: StrictBool = True,
        cell_indices: Optional[List[StrictInt]] = None,
        stop_on_error: StrictBool = True,
        timeout_per_cell_seconds: Annotated[StrictInt, Field(ge=5, le=14400)] = 300,
        output_save_path: Optional[str] = None,
    ) -> NotebookExecutionResponse:
        return await execute_notebook_impl(
            notebook_path=notebook_path,
            notebook_content=notebook_content,
            reset_kernel_before_run=reset_kernel_before_run,
            cell_indices=cell_indices,
            stop_on_error=stop_on_error,
            timeout_per_cell_seconds=timeout_per_cell_seconds,
            output_save_path=output_save_path,
        )


# ============================================================================
# Re-exported Kernel Tools (for test_notebook_runner.py patch compatibility)
# ============================================================================


async def colab_execute_code(
    code: str,
    runtime_id: Optional[str] = None,
    timeout_seconds: int = 300,
    wait_for_completion: bool = True,
    store_history: bool = True,
    max_text_lines: int = 1000,
    max_output_bytes: int = 102400,
    max_images: int = 5,
    *,
    _client_getter: Optional[Callable[[Optional[str]], Awaitable[Any]]] = None,
    _registry_getter: Optional[Callable[[], Any]] = None,
) -> List[MCPContentBlock] | Dict[str, Any]:
    """Execute Python code or IPython magic on the active Colab Jupyter kernel."""
    return await _kernel_colab_execute_code(
        code=code,
        runtime_id=runtime_id,
        timeout_seconds=timeout_seconds,
        wait_for_completion=wait_for_completion,
        store_history=store_history,
        max_text_lines=max_text_lines,
        max_output_bytes=max_output_bytes,
        max_images=max_images,
        _client_getter=_client_getter or (lambda rid=None: get_active_kernel_client(rid)),
        _registry_getter=_registry_getter or (lambda: get_execution_registry()),
    )


async def colab_get_execution_status(
    execution_id: str,
    *,
    _registry_getter: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Query progress, elapsed duration, and partial stdout tail of an async execution."""
    return await _kernel_colab_get_execution_status(
        execution_id=execution_id,
        _registry_getter=_registry_getter or (lambda: get_execution_registry()),
    )


async def colab_get_cell_output(
    execution_id: str,
    include_images: bool = True,
    max_text_lines: int = 1000,
    max_output_bytes: int = 102400,
    max_images: int = 5,
    *,
    _registry_getter: Optional[Callable[[], Any]] = None,
) -> List[MCPContentBlock]:
    """Retrieve buffered chronological outputs for an asynchronous or completed execution."""
    return await _kernel_colab_get_cell_output(
        execution_id=execution_id,
        include_images=include_images,
        max_text_lines=max_text_lines,
        max_output_bytes=max_output_bytes,
        max_images=max_images,
        _registry_getter=_registry_getter or (lambda: get_execution_registry()),
    )


async def colab_interrupt_execution(
    runtime_id: Optional[str] = None,
    *,
    _client_getter: Optional[Callable[[Optional[str]], Awaitable[Any]]] = None,
) -> Dict[str, Any]:
    """Interrupt current kernel execution (Tier 1 SIGINT -> 5.0s -> Tier 2 restart)."""
    return await _kernel_colab_interrupt_execution(
        runtime_id=runtime_id,
        _client_getter=_client_getter or (lambda rid=None: get_active_kernel_client(rid)),
    )


__all__ = [
    "CellExecutionSummary",
    "NotebookEngine",
    "NotebookExecutionResponse",
    "NotebookRunner",
    "colab_execute_code",
    "colab_execute_notebook",
    "colab_get_cell_output",
    "colab_get_execution_status",
    "colab_interrupt_execution",
    "execute_notebook_impl",
    "get_active_kernel_client",
    "get_execution_registry",
    "register_notebook_tools",
]
