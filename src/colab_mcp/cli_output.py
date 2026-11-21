"""
Rich terminal rendering, JSON output, image artifact saving, and semantic exit codes
for the ``colab-mcp`` CLI.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Sequence

import typer
from mcp.types import ImageContent, TextContent
from rich.console import Console
from rich.table import Table

from colab_mcp.auth.models import ColabAuthError
from colab_mcp.kernel.models import (
    ColabKernelError,
    KernelConnectionError,
    KernelNotFoundError,
)
from colab_mcp.runtime.models import (
    CloudCapacityError,
    ComputeUnitsExhaustedError,
    FreeTierQuotaExceededError,
    HighRamNotAvailableError,
    RuntimeDisconnectedError,
    TooManyAssignmentsError,
)
from colab_mcp.utils.logging import StderrProxy, unprotect_stdout
from colab_mcp.utils.redaction import redact_credentials, redact_data


class CLIExitCode(IntEnum):
    """Typed semantic exit codes for CLI scripting and automation."""

    OK = 0
    ERROR = 1
    USAGE_ERROR = 2
    AUTH_REQUIRED = 10
    QUOTA_OR_CAPACITY = 11
    RUNTIME_DISCONNECTED = 12
    EXECUTION_ERROR = 13


_MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


def ensure_cli_stdout() -> None:
    """Restore interactive CLI stdout if a lazy import activated StderrProxy."""
    if isinstance(sys.stdout, StderrProxy):
        unprotect_stdout()


def _stdout_console() -> Console:
    ensure_cli_stdout()
    return Console(file=sys.stdout, soft_wrap=True)


def _stderr_console() -> Console:
    return Console(file=sys.stderr, soft_wrap=True)


def classify_exception(exc: Exception) -> CLIExitCode:
    """Map domain exceptions to semantic CLI exit codes."""
    if isinstance(exc, ColabAuthError):
        return CLIExitCode.AUTH_REQUIRED
    if isinstance(
        exc,
        (
            HighRamNotAvailableError,
            TooManyAssignmentsError,
            ComputeUnitsExhaustedError,
            FreeTierQuotaExceededError,
            CloudCapacityError,
        ),
    ):
        return CLIExitCode.QUOTA_OR_CAPACITY
    if isinstance(exc, (RuntimeDisconnectedError, KernelConnectionError, KernelNotFoundError)):
        return CLIExitCode.RUNTIME_DISCONNECTED
    if isinstance(exc, ColabKernelError):
        return CLIExitCode.EXECUTION_ERROR
    if isinstance(exc, ValueError):
        return CLIExitCode.USAGE_ERROR
    return CLIExitCode.ERROR


def emit_json(payload: Any) -> None:
    """Redact credentials and write formatted JSON to stdout."""
    ensure_cli_stdout()
    safe_payload = redact_data(payload)
    typer.echo(json.dumps(safe_payload, indent=2, default=str))


def emit_error(exc: Exception, *, as_json: bool = False) -> NoReturn:
    """Render an exception cleanly to stderr with redaction and semantic exit code."""
    ensure_cli_stdout()
    code = classify_exception(exc)
    safe_msg = redact_credentials(str(exc))
    if as_json:
        err_payload = {
            "error": {
                "type": type(exc).__name__,
                "exit_code": int(code),
                "message": safe_msg,
            }
        }
        typer.echo(json.dumps(err_payload, indent=2), err=True)
    else:
        _stderr_console().print(f"[red]Error [{int(code)}]:[/red] {safe_msg}", highlight=False)
    raise typer.Exit(code=int(code)) from None


def save_image_blocks(blocks: Sequence[Any], output_dir: Path) -> List[Path]:
    """Decode base64 ImageContent blocks to numbered files inside ``output_dir``."""
    images = [b for b in blocks if isinstance(b, ImageContent) or getattr(b, "type", None) == "image"]
    if not images:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for idx, img in enumerate(images, start=1):
        mime = str(getattr(img, "mime_type", None) or getattr(img, "mimeType", None) or "image/png")
        ext = _MIME_TO_EXT.get(mime.lower(), ".png")
        raw_b64 = str(getattr(img, "data", "") or "")
        dest = output_dir / f"plot_{idx:03d}{ext}"
        tmp = output_dir / f".{dest.name}.tmp"
        tmp.write_bytes(base64.b64decode(raw_b64))
        os.replace(tmp, dest)
        saved.append(dest)
    return saved


def _build_table(title: str, columns: List[str]) -> Table:
    table = Table(title=title, show_lines=False)
    for col in columns:
        table.add_column(col, overflow="fold")
    return table


def render_key_values(title: str, rows: List[tuple[str, Any]]) -> None:
    """Render a compact two-column key/value summary table."""
    table = _build_table(title, ["Field", "Value"])
    for key, val in rows:
        table.add_row(str(key), redact_credentials(str(val)))
    _stdout_console().print(table)


def render_auth_status(payload: Dict[str, Any]) -> None:
    """Render auth status and optional active runtime summary."""
    rows: List[tuple[str, Any]] = [
        ("Authenticated", payload.get("is_authenticated")),
        ("User Index", payload.get("auth_user_index")),
        ("Email", payload.get("user_email") or "-"),
        ("Method", payload.get("auth_method") or "-"),
        ("Token Valid", payload.get("token_valid")),
        ("Account Tier", payload.get("account_tier") or "-"),
        ("Compute Units (CCU)", payload.get("compute_units_remaining") if payload.get("compute_units_remaining") is not None else "-"),
        ("Active Remote Sessions", payload.get("active_sessions_count", 0)),
    ]
    active = payload.get("active_runtime")
    if isinstance(active, dict):
        rows.append(("Active Runtime", f"{active.get('runtime_id')} ({active.get('kernel_state')})"))
    render_key_values("Google Colab Auth & Account Status", rows)


def render_runtimes(tier: str | None, items: Sequence[Any]) -> None:
    """Render accelerator catalog table."""
    table = _build_table(
        f"Colab Accelerators (Tier: {tier or 'UNKNOWN'})",
        ["Accelerator", "Variant", "Device", "VRAM", "RAM", "CCU/hr", "Available"],
    )
    for item in items:
        table.add_row(
            str(item.accelerator),
            str(item.variant),
            str(item.display_name),
            str(item.vram or "-"),
            str(item.system_ram),
            f"{item.ccu_cost_per_hour:.2f}",
            "yes" if item.available else "no",
        )
    _stdout_console().print(table)


def render_sessions(sessions: Sequence[Any], active_runtime_id: str | None) -> None:
    """Render active Colab sessions table."""
    if not sessions:
        _stdout_console().print("No active Colab sessions found.")
        return
    table = _build_table(
        f"Active Colab Sessions ({len(sessions)})",
        ["Runtime ID", "Accelerator", "Variant", "Kernel State", "Kernel ID", "Active CLI"],
    )
    for s in sessions:
        mark = "*" if s.runtime_id == active_runtime_id else ""
        table.add_row(
            str(s.runtime_id),
            str(s.accelerator),
            str(s.variant or "-"),
            str(s.kernel_state or "-"),
            str(s.kernel_id or "-"),
            mark,
        )
    _stdout_console().print(table)


def render_secrets(secrets_list: Sequence[Any]) -> None:
    """Render stored secret names and masked previews."""
    if not secrets_list:
        _stdout_console().print("No secrets stored in vault.")
        return
    table = _build_table(
        f"Encrypted Secret Vault ({len(secrets_list)})",
        ["Name", "Preview", "Length", "Updated (UTC)"],
    )
    for s in secrets_list:
        table.add_row(str(s.name), str(s.masked_preview), str(s.length), str(s.updated_at.isoformat()))
    _stdout_console().print(table)


def render_system_metrics(metrics: Any) -> None:
    """Render GPU VRAM/utilization, RAM, and disk metrics."""
    con = _stdout_console()
    if metrics.gpus:
        gpu_table = _build_table(
            f"GPU Telemetry ({metrics.runtime_id})",
            ["Idx", "Device", "Util %", "VRAM Used (MB)", "VRAM Total (MB)", "Temp (C)"],
        )
        for g in metrics.gpus:
            gpu_table.add_row(
                str(g.index),
                str(g.name),
                f"{g.gpu_utilization_percent:.1f}" if g.gpu_utilization_percent is not None else "-",
                f"{g.vram_used_mb:.0f}" if g.vram_used_mb is not None else "-",
                f"{g.vram_total_mb:.0f}" if g.vram_total_mb is not None else "-",
                f"{g.temperature_celsius:.0f}" if g.temperature_celsius is not None else "-",
            )
        con.print(gpu_table)
    rows = [
        ("Runtime ID", metrics.runtime_id),
        (
            "System RAM",
            f"{metrics.ram_used_mb} / {metrics.ram_total_mb} MB ({metrics.ram_utilization_percent}%)"
            if metrics.ram_total_mb is not None
            else "-",
        ),
        (
            f"Disk ({metrics.disk_path})",
            f"{metrics.disk_used_gb} / {metrics.disk_total_gb} GB free={metrics.disk_free_gb} GB ({metrics.disk_utilization_percent}%)"
            if metrics.disk_total_gb is not None
            else "-",
        ),
    ]
    render_key_values("Host Memory & Disk Telemetry", rows)


def render_execution_blocks(blocks: Sequence[Any], saved_images: Sequence[Path]) -> None:
    """Render TextContent blocks to stdout and report saved or inline image count."""
    ensure_cli_stdout()
    image_count = 0
    for block in blocks:
        if isinstance(block, TextContent) or getattr(block, "type", None) == "text":
            txt = redact_credentials(str(getattr(block, "text", "")))
            typer.echo(txt, nl=not txt.endswith("\n"))
        elif isinstance(block, ImageContent) or getattr(block, "type", None) == "image":
            image_count += 1
    if saved_images:
        for path in saved_images:
            typer.echo(f"[colab-mcp] Saved plot image: {path}")
    elif image_count:
        typer.echo(
            f"[colab-mcp] Output included {image_count} image(s). Pass --save-images-dir <dir> to save them."
        )


def render_notebook_summary(resp: Any) -> None:
    """Render notebook execution result table."""
    status_label = "failed" if resp.failed_cells > 0 else "completed"
    render_key_values(
        f"Notebook Execution ({status_label})",
        [
            ("Kernel Reset", resp.kernel_reset_performed),
            ("Executed Cells", f"{resp.executed_cells} / {resp.total_cells} (skipped={resp.skipped_cells})"),
            ("Successful / Failed", f"{resp.successful_cells} / {resp.failed_cells}"),
            ("Duration (s)", f"{resp.total_duration_seconds:.2f}"),
            ("Saved Notebook", resp.saved_to or "-"),
        ],
    )
    table = _build_table(
        "Per-Cell Summary",
        ["Cell", "Count", "Status", "Duration (s)", "Plots", "Error"],
    )
    for cell in resp.cell_summaries:
        err = f"{cell.error_name}: {cell.error_value}" if cell.error_name else "-"
        table.add_row(
            str(cell.cell_index),
            str(cell.execution_count if cell.execution_count is not None else "-"),
            str(cell.status),
            f"{cell.duration_seconds:.2f}",
            str(cell.plot_count),
            redact_credentials(err),
        )
    _stdout_console().print(table)


def render_remote_files(resp: Any) -> None:
    """Render Colab VM directory listing."""
    if not resp.entries:
        _stdout_console().print(f"No entries found at {resp.remote_path}.")
        return
    table = _build_table(
        f"Remote Files ({resp.runtime_id}:{resp.remote_path})",
        ["Type", "Name", "Size (bytes)", "Path"],
    )
    for entry in resp.entries:
        table.add_row(
            str(entry.type),
            str(entry.name),
            str(entry.size_bytes if entry.size_bytes is not None else "-"),
            str(entry.path),
        )
    _stdout_console().print(table)


def render_drive_files(resp: Any) -> None:
    """Render Google Drive listing."""
    if not resp.files:
        _stdout_console().print(f"No Drive files found in '{resp.folder}'.")
        return
    table = _build_table(
        f"Google Drive ({resp.folder})",
        ["Type", "Name", "Size (bytes)", "File ID", "Modified"],
    )
    for item in resp.files:
        table.add_row(
            "folder" if item.is_folder else "file",
            str(item.name),
            str(item.size_bytes if item.size_bytes is not None else "-"),
            str(item.id),
            str(item.modified_time or "-"),
        )
    _stdout_console().print(table)


__all__ = [
    "CLIExitCode",
    "classify_exception",
    "emit_error",
    "emit_json",
    "ensure_cli_stdout",
    "render_auth_status",
    "render_drive_files",
    "render_execution_blocks",
    "render_key_values",
    "render_notebook_summary",
    "render_remote_files",
    "render_runtimes",
    "render_secrets",
    "render_sessions",
    "render_system_metrics",
    "save_image_blocks",
]
