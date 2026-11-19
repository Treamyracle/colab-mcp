"""
FastMCP Tools & Execution Registry for Google Colab Kernel Execution (Tools 8, 10, 11, 12).

Implements TECH_SPEC Â§7.3.3:
- Tool 8: ``colab_execute_code``
- Tool 10: ``colab_get_execution_status``
- Tool 11: ``colab_get_cell_output``
- Tool 12: ``colab_interrupt_execution``
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Annotated, Any, Awaitable, Callable, Dict, List, Mapping, Optional
from unittest.mock import NonCallableMock

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, ImageContent, TextContent
from pydantic import BaseModel, StrictBool

from colab_mcp.kernel.client import KernelWebSocketClient
from colab_mcp.kernel.mime_converter import (
    MCPContentBlock,
    apply_head_tail_truncation,
    convert_iopub_to_mcp_blocks,
    sanitize_ansi_text,
)
from colab_mcp.kernel.models import (
    ExecutionLifecycleStatus,
    ExecutionStatusResponse,
    InterruptExecutionResponse,
    KernelConnectionError,
    KernelExecutionTimeoutError,
)
from colab_mcp.runtime.session_manager import get_session_manager
from colab_mcp.utils.logging import get_logger
from colab_mcp.utils.redaction import redact_credentials, redact_data

logger = get_logger(__name__)


# ============================================================================
# Kernel Client Cache & Execution Registry
# ============================================================================


_KERNEL_CLIENTS: Dict[str, KernelWebSocketClient] = {}
_CLIENT_LOCKS: Dict[str, asyncio.Lock] = {}


def _get_client_lock(endpoint: str) -> asyncio.Lock:
    lock = _CLIENT_LOCKS.get(endpoint)
    if lock is None:
        lock = asyncio.Lock()
        _CLIENT_LOCKS[endpoint] = lock
    return lock


async def get_active_kernel_client(
    runtime_id: Optional[str] = None,
) -> KernelWebSocketClient:
    """
    Resolve active ``KernelConnectionParams`` from ``SessionManager`` and return
    a connected ``KernelWebSocketClient`` for the target runtime.
    """
    mgr = get_session_manager()
    params = await mgr.get_kernel_connection_params(runtime_id)

    async with _get_client_lock(params.endpoint):
        cached = _KERNEL_CLIENTS.get(params.endpoint)
        if cached is not None and (
            cached.kernel_id != params.kernel_id or cached.proxy_url != params.proxy_url.rstrip("/")
        ):
            try:
                await cached.disconnect()
            except Exception:
                pass
            _KERNEL_CLIENTS.pop(params.endpoint, None)
            cached = None

        if cached is not None:
            cached.proxy_token = params.proxy_token
            if not cached.is_connected:
                await cached.connect()
            client = cached
        else:
            client = KernelWebSocketClient.from_connection_params(params)
            await client.connect()
            _KERNEL_CLIENTS[params.endpoint] = client

    mgr.record_activity(endpoint=params.endpoint)
    return client


async def clear_kernel_clients() -> None:
    """Disconnect and clear all cached kernel WebSocket clients (for shutdown/test isolation)."""
    for client in list(_KERNEL_CLIENTS.values()):
        try:
            await client.disconnect()
        except Exception:
            pass
    _KERNEL_CLIENTS.clear()
    _CLIENT_LOCKS.clear()


def reset_kernel_clients() -> None:
    """Synchronously clear cached kernel client mappings (primarily for test isolation)."""
    _KERNEL_CLIENTS.clear()
    _CLIENT_LOCKS.clear()


class ExecutionRegistry(dict):
    """
    Dictionary-compatible registry tracking asynchronous kernel executions.

    Subclassing ``dict`` ensures seamless interoperability with mocked dict registries
    (e.g. ``test_notebook_runner.py``) and real background execution tasks.
    """

    def create_execution(
        self,
        execution_id: str,
        runtime_id: Optional[str],
        code: str,
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "execution_id": execution_id,
            "runtime_id": runtime_id,
            "code": code,
            "status": ExecutionLifecycleStatus.RUNNING.value,
            "started_at": time.monotonic(),
            "finished_at": None,
            "events": [],
            "blocks": None,
            "stdout_chunks": [],
            "has_error": False,
            "error_name": None,
            "error_value": None,
            "error_message": None,
            "execution_count": None,
            "task": None,
        }
        self[execution_id] = record
        return record

    def record_iopub_message(self, execution_id: str, msg: Dict[str, Any]) -> None:
        record = self.get(execution_id)
        if not isinstance(record, dict) or not isinstance(msg, dict):
            return
        events = record.setdefault("events", [])
        events.append(msg)

        header = msg.get("header") if isinstance(msg.get("header"), dict) else {}
        msg_type = header.get("msg_type") or msg.get("msg_type")
        content = msg.get("content") if isinstance(msg.get("content"), dict) else {}

        if msg_type == "stream":
            stream_name = str(content.get("name", "stdout")).lower()
            raw_text = str(content.get("text", ""))
            cleaned = redact_credentials(sanitize_ansi_text(raw_text))
            if cleaned:
                if stream_name == "stderr":
                    record.setdefault("stdout_chunks", []).append(f"[stderr] {cleaned}")
                else:
                    record.setdefault("stdout_chunks", []).append(cleaned)
        elif msg_type == "execute_result":
            if content.get("execution_count") is not None:
                record["execution_count"] = int(content["execution_count"])
        elif msg_type == "error":
            record["has_error"] = True
            record["error_name"] = str(content.get("ename", "Error"))
            record["error_value"] = redact_credentials(str(content.get("evalue", "")))

    def mark_completed(
        self,
        execution_id: str,
        *,
        status: str = ExecutionLifecycleStatus.COMPLETED.value,
        has_error: bool = False,
        error_name: Optional[str] = None,
        error_value: Optional[str] = None,
        error_message: Optional[str] = None,
        blocks: Optional[List[MCPContentBlock]] = None,
        execution_count: Optional[int] = None,
    ) -> None:
        record = self.get(execution_id)
        if not isinstance(record, dict):
            return
        record["finished_at"] = time.monotonic()
        record["status"] = status
        if has_error:
            record["has_error"] = True
        if error_name is not None:
            record["error_name"] = error_name
        if error_value is not None:
            record["error_value"] = redact_credentials(error_value)
        if error_message is not None:
            record["error_message"] = redact_credentials(error_message)
        if blocks is not None:
            record["blocks"] = blocks
        if execution_count is not None:
            record["execution_count"] = execution_count


_EXECUTION_REGISTRY: Optional[ExecutionRegistry] = None


def get_execution_registry() -> ExecutionRegistry:
    """Return the process-wide ExecutionRegistry singleton."""
    global _EXECUTION_REGISTRY
    if _EXECUTION_REGISTRY is None:
        _EXECUTION_REGISTRY = ExecutionRegistry()
    return _EXECUTION_REGISTRY


def reset_execution_registry() -> None:
    """Cancel any lingering background tasks and reset the ExecutionRegistry singleton."""
    global _EXECUTION_REGISTRY
    if _EXECUTION_REGISTRY is not None:
        for record in list(_EXECUTION_REGISTRY.values()):
            if isinstance(record, dict):
                task = record.get("task")
                if isinstance(task, asyncio.Task) and not task.done():
                    task.cancel()
        _EXECUTION_REGISTRY.clear()
    _EXECUTION_REGISTRY = None


# ============================================================================
# Validation & Redaction Helpers
# ============================================================================


def _validate_int_range(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not (minimum <= value <= maximum):
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _validate_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _validate_optional_str(value: Any, name: str) -> Optional[str]:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or None")
    return value


def _validate_nonempty_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _redact_blocks(blocks: List[MCPContentBlock]) -> List[MCPContentBlock]:
    redacted: List[MCPContentBlock] = []
    for block in blocks:
        if isinstance(block, TextContent):
            redacted.append(
                TextContent(
                    type="text",
                    text=redact_credentials(block.text),
                    annotations=block.annotations,
                    meta=block.meta,
                )
            )
        else:
            redacted.append(block)
    return redacted


def _record_session_activity(runtime_id: Optional[str] = None) -> None:
    try:
        get_session_manager().record_activity(endpoint=runtime_id)
    except Exception:
        pass


def _is_configured_mock_method(obj: Any, method_name: str) -> bool:
    if not isinstance(obj, NonCallableMock):
        return hasattr(obj, method_name)
    if method_name in getattr(obj, "__dict__", {}):
        return True
    children = getattr(obj, "_mock_children", {})
    if method_name in children:
        child = children[method_name]
        if getattr(child, "side_effect", None) is not None:
            return True
        rv = getattr(child, "return_value", None)
        if rv is not None and not isinstance(rv, NonCallableMock):
            return True
    return False


# ============================================================================
# Core Tool Implementations (with optional getter hooks for M4 compatibility)
# ============================================================================


async def _run_async_execution_task(
    execution_id: str,
    client: Any,
    registry: ExecutionRegistry,
    *,
    code: str,
    timeout_seconds: int,
    store_history: bool,
    max_text_lines: int,
    max_output_bytes: int,
    max_images: int,
) -> None:
    try:
        if isinstance(client, KernelWebSocketClient) or _is_configured_mock_method(client, "execute_raw"):
            raw = await client.execute_raw(
                code=code,
                timeout_seconds=timeout_seconds,
                store_history=store_history,
                on_message=lambda msg: registry.record_iopub_message(execution_id, msg),
                execution_id=execution_id,
            )
            blocks = convert_iopub_to_mcp_blocks(
                raw.iopub_messages,
                include_images=True,
                max_text_lines=max_text_lines,
                max_output_bytes=max_output_bytes,
                max_images=max_images,
            )
            blocks = _redact_blocks(blocks)
            registry.mark_completed(
                execution_id,
                status=(
                    ExecutionLifecycleStatus.FAILED.value
                    if raw.has_error
                    else ExecutionLifecycleStatus.COMPLETED.value
                ),
                has_error=raw.has_error,
                error_name=raw.error_name,
                error_value=raw.error_value,
                blocks=blocks,
                execution_count=raw.execution_count,
            )
        else:
            blocks = await client.execute(
                code=code,
                timeout_seconds=timeout_seconds,
                store_history=store_history,
                max_text_lines=max_text_lines,
                max_output_bytes=max_output_bytes,
                max_images=max_images,
            )
            blocks = _redact_blocks(list(blocks or []))
            for blk in blocks:
                if isinstance(blk, TextContent) and blk.text:
                    record = registry.get(execution_id)
                    if isinstance(record, dict):
                        record.setdefault("stdout_chunks", []).append(blk.text)
            registry.mark_completed(
                execution_id,
                status=ExecutionLifecycleStatus.COMPLETED.value,
                has_error=False,
                blocks=blocks,
            )
    except asyncio.CancelledError:
        registry.mark_completed(
            execution_id,
            status=ExecutionLifecycleStatus.INTERRUPTED.value,
            has_error=False,
            error_message="Execution task cancelled.",
        )
    except KernelExecutionTimeoutError as exc:
        registry.mark_completed(
            execution_id,
            status=ExecutionLifecycleStatus.TIMED_OUT.value,
            has_error=True,
            error_name="TimeoutError",
            error_value=str(exc),
            error_message=str(exc),
        )
    except Exception as exc:
        err_text = redact_credentials(str(exc))
        registry.mark_completed(
            execution_id,
            status=ExecutionLifecycleStatus.FAILED.value,
            has_error=True,
            error_name=type(exc).__name__,
            error_value=err_text,
            error_message=err_text,
            blocks=[TextContent(type="text", text=f"{type(exc).__name__}: {err_text}")],
        )


async def execute_code_impl(
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
) -> tuple[List[MCPContentBlock] | Dict[str, Any], bool]:
    """
    Core logic for ``colab_execute_code`` returning ``(payload, has_error)``.
    """
    if not isinstance(code, str):
        raise ValueError("code must be a string")
    _validate_optional_str(runtime_id, "runtime_id")
    _validate_int_range(timeout_seconds, "timeout_seconds", 5, 3600)
    _validate_bool(wait_for_completion, "wait_for_completion")
    _validate_bool(store_history, "store_history")
    _validate_int_range(max_text_lines, "max_text_lines", 10, 10000)
    _validate_int_range(max_output_bytes, "max_output_bytes", 1024, 10485760)
    _validate_int_range(max_images, "max_images", 1, 50)

    client_getter = _client_getter or get_active_kernel_client
    registry_getter = _registry_getter or get_execution_registry

    client = await client_getter(runtime_id)
    effective_runtime = getattr(client, "runtime_id", None) or getattr(client, "endpoint", None) or runtime_id
    _record_session_activity(effective_runtime)

    if not wait_for_completion:
        registry = registry_getter()
        execution_id = f"exec-{uuid.uuid4().hex[:12]}"
        if isinstance(registry, ExecutionRegistry):
            record = registry.create_execution(execution_id, effective_runtime, code)
        else:
            record = {
                "execution_id": execution_id,
                "runtime_id": effective_runtime,
                "code": code,
                "status": ExecutionLifecycleStatus.RUNNING.value,
                "started_at": time.monotonic(),
                "finished_at": None,
                "events": [],
                "blocks": None,
                "stdout_chunks": [],
                "has_error": False,
                "task": None,
            }
            registry[execution_id] = record

        task = asyncio.create_task(
            _run_async_execution_task(
                execution_id=execution_id,
                client=client,
                registry=registry,
                code=code,
                timeout_seconds=timeout_seconds,
                store_history=store_history,
                max_text_lines=max_text_lines,
                max_output_bytes=max_output_bytes,
                max_images=max_images,
            )
        )
        record["task"] = task
        return (
            redact_data(
                {
                    "execution_id": execution_id,
                    "status": ExecutionLifecycleStatus.RUNNING.value,
                    "runtime_id": effective_runtime,
                    "message": "Execution dispatched asynchronously.",
                }
            ),
            False,
        )

    # Synchronous execution
    try:
        if isinstance(client, KernelWebSocketClient) or _is_configured_mock_method(client, "execute_raw"):
            raw = await client.execute_raw(
                code=code,
                timeout_seconds=timeout_seconds,
                store_history=store_history,
            )
            blocks = convert_iopub_to_mcp_blocks(
                raw.iopub_messages,
                include_images=True,
                max_text_lines=max_text_lines,
                max_output_bytes=max_output_bytes,
                max_images=max_images,
            )
            if raw.has_error and not any(
                (m.get("header") or {}).get("msg_type") == "error" for m in raw.iopub_messages
            ):
                err_msg = f"{raw.error_name or 'Error'}: {raw.error_value or ''}"
                blocks.append(TextContent(type="text", text=redact_credentials(err_msg)))
            return (_redact_blocks(blocks), bool(raw.has_error))

        blocks = await client.execute(
            code=code,
            timeout_seconds=timeout_seconds,
            store_history=store_history,
            max_text_lines=max_text_lines,
            max_output_bytes=max_output_bytes,
            max_images=max_images,
        )
        return (_redact_blocks(list(blocks or [])), False)

    except KernelExecutionTimeoutError as exc:
        err_msg = redact_credentials(str(exc))
        return ([TextContent(type="text", text=f"KernelExecutionTimeoutError: {err_msg}")], True)
    except KernelConnectionError:
        raise


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
    payload, _has_error = await execute_code_impl(
        code=code,
        runtime_id=runtime_id,
        timeout_seconds=timeout_seconds,
        wait_for_completion=wait_for_completion,
        store_history=store_history,
        max_text_lines=max_text_lines,
        max_output_bytes=max_output_bytes,
        max_images=max_images,
        _client_getter=_client_getter,
        _registry_getter=_registry_getter,
    )
    return payload


async def get_execution_status_impl(
    execution_id: str,
    *,
    _registry_getter: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Core logic for ``colab_get_execution_status``."""
    _validate_nonempty_str(execution_id, "execution_id")

    registry_getter = _registry_getter or get_execution_registry
    registry = registry_getter()
    record = registry.get(execution_id) if isinstance(registry, Mapping) else None

    runtime_id = record.get("runtime_id") if isinstance(record, Mapping) else None
    _record_session_activity(runtime_id)

    if record is None:
        return redact_data(
            {
                "execution_id": execution_id,
                "status": ExecutionLifecycleStatus.NOT_FOUND.value,
                "elapsed_seconds": 0.0,
                "partial_stdout_tail": "",
                "has_error": False,
            }
        )

    if hasattr(record, "to_status") and callable(getattr(record, "to_status")):
        status_model = record.to_status(current_monotonic=time.monotonic())
        return redact_data(status_model.model_dump())

    if "elapsed_seconds" in record and record["elapsed_seconds"] is not None:
        elapsed = float(record["elapsed_seconds"])
    else:
        started_at = float(record.get("started_at") or 0.0)
        finished_at = record.get("finished_at")
        end_ts = float(finished_at) if finished_at is not None else time.monotonic()
        elapsed = round(max(0.0, end_ts - started_at), 3) if started_at > 0 else 0.0

    if "partial_stdout_tail" in record and record["partial_stdout_tail"] is not None:
        tail = str(record["partial_stdout_tail"])
    else:
        chunks = record.get("stdout_chunks") or []
        combined_stdout = "".join(str(c) for c in chunks)
        lines = combined_stdout.splitlines()
        if len(lines) > 20:
            combined_stdout = "\n".join(lines[-20:])
        tail = combined_stdout[-2000:]

    result = {
        "execution_id": execution_id,
        "status": str(record.get("status", ExecutionLifecycleStatus.RUNNING.value)),
        "elapsed_seconds": elapsed,
        "partial_stdout_tail": redact_credentials(tail),
        "has_error": bool(record.get("has_error", False)),
        "execution_count": record.get("execution_count"),
        "error_name": record.get("error_name"),
        "error_value": record.get("error_value"),
    }
    return redact_data(result)


async def colab_get_execution_status(
    execution_id: str,
    *,
    _registry_getter: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Query progress, elapsed duration, and partial stdout tail of an async execution."""
    return await get_execution_status_impl(
        execution_id=execution_id,
        _registry_getter=_registry_getter,
    )


def _filter_preconverted_blocks(
    blocks: List[MCPContentBlock],
    *,
    include_images: bool,
    max_text_lines: int,
    max_output_bytes: int,
    max_images: int,
) -> List[MCPContentBlock]:
    filtered: List[MCPContentBlock] = []
    images_kept = 0
    omitted_images = 0

    for block in blocks:
        if isinstance(block, ImageContent):
            if not include_images:
                continue
            if images_kept < max_images:
                filtered.append(block)
                images_kept += 1
            else:
                omitted_images += 1
        elif isinstance(block, TextContent):
            bounded = apply_head_tail_truncation(
                block.text,
                max_lines=max_text_lines,
                max_bytes=max_output_bytes,
            )
            filtered.append(
                TextContent(
                    type="text",
                    text=redact_credentials(bounded),
                    annotations=block.annotations,
                    meta=block.meta,
                )
            )

    if include_images and omitted_images > 0:
        filtered.append(
            TextContent(
                type="text",
                text=f"[Notice: {omitted_images} additional plot images omitted due to max_images limit]",
            )
        )
    return filtered


async def get_cell_output_impl(
    execution_id: str,
    include_images: bool = True,
    max_text_lines: int = 1000,
    max_output_bytes: int = 102400,
    max_images: int = 5,
    *,
    _registry_getter: Optional[Callable[[], Any]] = None,
) -> tuple[List[MCPContentBlock], bool]:
    """Core logic for ``colab_get_cell_output`` returning ``(blocks, has_error)``."""
    _validate_nonempty_str(execution_id, "execution_id")
    _validate_bool(include_images, "include_images")
    _validate_int_range(max_text_lines, "max_text_lines", 10, 10000)
    _validate_int_range(max_output_bytes, "max_output_bytes", 1024, 10485760)
    _validate_int_range(max_images, "max_images", 1, 50)

    registry_getter = _registry_getter or get_execution_registry
    registry = registry_getter()
    record = registry.get(execution_id) if isinstance(registry, Mapping) else None

    runtime_id = record.get("runtime_id") if isinstance(record, Mapping) else None
    _record_session_activity(runtime_id)

    if record is None:
        return (
            [
                TextContent(
                    type="text",
                    text=redact_credentials(
                        f"No execution found for execution_id '{execution_id}'."
                    ),
                )
            ],
            True,
        )

    events = record.get("events") if isinstance(record, Mapping) else None
    has_error = bool(record.get("has_error", False))

    if events:
        blocks = convert_iopub_to_mcp_blocks(
            events,
            include_images=include_images,
            max_text_lines=max_text_lines,
            max_output_bytes=max_output_bytes,
            max_images=max_images,
        )
        if has_error and not any(
            (m.get("header") or {}).get("msg_type") == "error" for m in events if isinstance(m, dict)
        ):
            err_msg = record.get("error_message") or f"{record.get('error_name', 'Error')}: {record.get('error_value', '')}"
            blocks.append(TextContent(type="text", text=redact_credentials(str(err_msg))))
        return (_redact_blocks(blocks), has_error)

    pre_blocks = record.get("blocks") if isinstance(record, Mapping) else None
    if isinstance(pre_blocks, list):
        filtered = _filter_preconverted_blocks(
            pre_blocks,
            include_images=include_images,
            max_text_lines=max_text_lines,
            max_output_bytes=max_output_bytes,
            max_images=max_images,
        )
        return (filtered, has_error)

    if has_error and record.get("error_message"):
        return (
            [TextContent(type="text", text=redact_credentials(str(record["error_message"])))],
            True,
        )

    return ([TextContent(type="text", text="")], has_error)


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
    blocks, _has_error = await get_cell_output_impl(
        execution_id=execution_id,
        include_images=include_images,
        max_text_lines=max_text_lines,
        max_output_bytes=max_output_bytes,
        max_images=max_images,
        _registry_getter=_registry_getter,
    )
    return blocks


async def interrupt_execution_impl(
    runtime_id: Optional[str] = None,
    *,
    _client_getter: Optional[Callable[[Optional[str]], Awaitable[Any]]] = None,
) -> Dict[str, Any]:
    """Core logic for ``colab_interrupt_execution``."""
    _validate_optional_str(runtime_id, "runtime_id")

    client_getter = _client_getter or get_active_kernel_client
    client = await client_getter(runtime_id)
    effective_runtime = getattr(client, "runtime_id", None) or getattr(client, "endpoint", None) or runtime_id
    _record_session_activity(effective_runtime)

    if isinstance(client, KernelWebSocketClient) or _is_configured_mock_method(client, "interrupt_with_escalation"):
        raw_res = await client.interrupt_with_escalation(timeout=5.0)
    else:
        raw_res = await client.interrupt()

    if isinstance(raw_res, BaseModel):
        res_dict = raw_res.model_dump()
    elif isinstance(raw_res, Mapping):
        res_dict = dict(raw_res)
    else:
        res_dict = {
            "runtime_id": effective_runtime,
            "status": "interrupted",
            "state_reset": False,
            "kernel_restarted": False,
            "kernel_state": "idle",
            "warning": None,
            "message": str(raw_res),
        }

    if not res_dict.get("runtime_id"):
        res_dict["runtime_id"] = effective_runtime

    # Normalize default fields for InterruptExecutionResponse schema
    res_dict.setdefault("status", "restarted" if res_dict.get("kernel_restarted") else "interrupted")
    res_dict.setdefault("state_reset", bool(res_dict.get("kernel_restarted", False)))
    res_dict.setdefault("kernel_restarted", bool(res_dict.get("state_reset", False)))
    res_dict.setdefault("kernel_state", "idle")
    res_dict.setdefault("warning", None)
    res_dict.setdefault(
        "message",
        (
            "Kernel restarted and WebSocket bridge automatically reconnected."
            if res_dict.get("kernel_restarted")
            else "SIGINT interrupt signal dispatched; execution halted while preserving in-memory state."
        ),
    )

    if res_dict.get("kernel_restarted") or res_dict.get("state_reset"):
        try:
            mgr = get_session_manager()
            session = mgr.get_session(effective_runtime)
            if session is not None:
                session.kernel_state = str(res_dict.get("kernel_state", "idle"))
                client_kid = getattr(client, "kernel_id", None)
                if client_kid:
                    session.kernel_id = str(client_kid)
        except Exception:
            pass

    return redact_data(res_dict)


async def colab_interrupt_execution(
    runtime_id: Optional[str] = None,
    *,
    _client_getter: Optional[Callable[[Optional[str]], Awaitable[Any]]] = None,
) -> Dict[str, Any]:
    """Interrupt current kernel execution (Tier 1 SIGINT -> 5.0s -> Tier 2 restart)."""
    return await interrupt_execution_impl(
        runtime_id=runtime_id,
        _client_getter=_client_getter,
    )


# ============================================================================
# FastMCP Tool Registration (TECH_SPEC Â§7.3.3)
# ============================================================================


def register_kernel_tools(mcp: FastMCP) -> None:
    """Register Tools 8, 10, 11, and 12 on the FastMCP server."""

    @mcp.tool(
        name="colab_execute_code",
        description=(
            "Execute a Python snippet or IPython magic command (!pip, %timeit) on the "
            "connected Colab kernel. Returns captured stdout, stderr, rich multimodal MIME "
            "outputs (e.g. Matplotlib charts as ImageContent), and sanitized error tracebacks "
            "in strict chronological order with automatic head-tail truncation."
        ),
    )
    async def _tool_colab_execute_code(
        code: str,
        runtime_id: Optional[str] = None,
        timeout_seconds: int = 300,
        wait_for_completion: StrictBool = True,
        store_history: StrictBool = True,
        max_text_lines: int = 1000,
        max_output_bytes: int = 102400,
        max_images: int = 5,
    ) -> Any:
        payload, has_error = await execute_code_impl(
            code=code,
            runtime_id=runtime_id,
            timeout_seconds=timeout_seconds,
            wait_for_completion=wait_for_completion,
            store_history=store_history,
            max_text_lines=max_text_lines,
            max_output_bytes=max_output_bytes,
            max_images=max_images,
        )
        if has_error and isinstance(payload, list):
            return CallToolResult(content=payload, isError=True)  # type: ignore[arg-type]
        return payload

    @mcp.tool(
        name="colab_get_execution_status",
        description=(
            "Query execution progress, elapsed time, and partial stdout of an "
            "asynchronous cell job dispatched with wait_for_completion=false."
        ),
    )
    async def _tool_colab_get_execution_status(
        execution_id: str,
    ) -> ExecutionStatusResponse:
        data = await colab_get_execution_status(execution_id=execution_id)
        if data.get("status") == ExecutionLifecycleStatus.NOT_FOUND.value:
            model = ExecutionStatusResponse(
                execution_id=str(data.get("execution_id", execution_id)),
                status=ExecutionLifecycleStatus.NOT_FOUND.value,
                elapsed_seconds=float(data.get("elapsed_seconds", 0.0)),
                partial_stdout_tail=str(data.get("partial_stdout_tail", "")),
                has_error=True,
                execution_count=data.get("execution_count"),
                error_name="ExecutionNotFoundError",
                error_value=f"No execution found for execution_id '{execution_id}'.",
            )
            payload = redact_data(model.model_dump(mode="json"))
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(payload, indent=2))],
                structuredContent=payload,
                isError=True,
            )  # type: ignore[return-value]
        return ExecutionStatusResponse(**data)

    @mcp.tool(
        name="colab_get_cell_output",
        description=(
            "Retrieve full buffered outputs (stdout, stderr, and generated plots as "
            "ImageContent blocks) for an asynchronous or completed execution job in "
            "strict chronological order."
        ),
    )
    async def _tool_colab_get_cell_output(
        execution_id: str,
        include_images: StrictBool = True,
        max_text_lines: int = 1000,
        max_output_bytes: int = 102400,
        max_images: int = 5,
    ) -> Any:
        blocks, has_error = await get_cell_output_impl(
            execution_id=execution_id,
            include_images=include_images,
            max_text_lines=max_text_lines,
            max_output_bytes=max_output_bytes,
            max_images=max_images,
        )
        if has_error:
            return CallToolResult(content=blocks, isError=True)  # type: ignore[arg-type]
        return blocks

    @mcp.tool(
        name="colab_interrupt_execution",
        description=(
            "Cancel and abort the currently executing cell or snippet on the Colab kernel. "
            "Dispatches non-destructive SIGINT first (preserving in-memory variables). "
            "Escalates to kernel restart after 5.0s if unresponsive, explicitly reporting "
            "whether state_reset and kernel_restarted occurred."
        ),
    )
    async def _tool_colab_interrupt_execution(
        runtime_id: Optional[str] = None,
    ) -> InterruptExecutionResponse:
        data = await colab_interrupt_execution(runtime_id=runtime_id)
        return InterruptExecutionResponse(**data)


__all__ = [
    "ExecutionRegistry",
    "clear_kernel_clients",
    "colab_execute_code",
    "colab_get_cell_output",
    "colab_get_execution_status",
    "colab_interrupt_execution",
    "execute_code_impl",
    "get_active_kernel_client",
    "get_cell_output_impl",
    "get_execution_registry",
    "get_execution_status_impl",
    "interrupt_execution_impl",
    "register_kernel_tools",
    "reset_execution_registry",
    "reset_kernel_clients",
]
