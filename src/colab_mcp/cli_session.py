"""
Token-free CLI session persistence and TFE runtime re-attachment.

Persists active Colab runtime routing metadata in ``~/.colabmcp/cli_session.json``
(without storing proxy tokens or OAuth credentials) so separate CLI invocations
seamlessly share the active cloud runtime.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from colab_mcp.auth import keystore as _keystore
from colab_mcp.runtime.client import ColabTFEClient
from colab_mcp.runtime.models import Assignment, RuntimeDisconnectedError
from colab_mcp.runtime.session_manager import get_session_manager
from colab_mcp.utils.logging import get_logger

logger = get_logger(__name__)

CLI_SESSION_FILENAME = "cli_session.json"


class CLISessionState(BaseModel):
    """Non-sensitive routing state persisted for cross-command CLI continuity."""

    runtime_id: str
    kernel_id: Optional[str] = None
    accelerator: str = "NONE"
    variant: str = "DEFAULT"
    high_ram: bool = False
    auth_user_index: int = Field(default=0, ge=0)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def get_cli_session_file() -> Path:
    """Return path to ``~/.colabmcp/cli_session.json``."""
    return _keystore.get_credentials_dir() / CLI_SESSION_FILENAME


def load_cli_session() -> Optional[CLISessionState]:
    """Load persisted CLI session metadata if present and valid."""
    filepath = get_cli_session_file()
    if not filepath.is_file():
        return None
    try:
        raw = json.loads(filepath.read_text(encoding="utf-8"))
        return CLISessionState.model_validate(raw)
    except Exception as exc:
        logger.debug("Ignoring unreadable CLI session pointer: %s", exc)
        return None


def save_cli_session(state: CLISessionState) -> Path:
    """Atomically write non-sensitive CLI session routing state to disk."""
    cred_dir = _keystore.get_credentials_dir()
    cred_dir.mkdir(parents=True, exist_ok=True)
    filepath = get_cli_session_file()
    temp_file = cred_dir / f"{filepath.name}.tmp"

    payload = state.model_copy(
        update={"updated_at": datetime.now(timezone.utc)}
    ).model_dump_json(indent=2)

    if sys.platform != "win32":
        fd = os.open(
            str(temp_file),
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            stat.S_IRUSR | stat.S_IWUSR,
        )
        with open(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
    else:
        temp_file.write_text(payload, encoding="utf-8")

    os.replace(temp_file, filepath)
    _keystore.enforce_file_permissions(filepath)
    return filepath


def clear_cli_session(runtime_id: Optional[str] = None) -> bool:
    """Remove persisted CLI session pointer if present (optionally matching runtime_id)."""
    filepath = get_cli_session_file()
    if not filepath.is_file():
        return False
    if runtime_id is not None:
        current = load_cli_session()
        if current is not None and current.runtime_id != runtime_id:
            return False
    try:
        filepath.unlink()
        return True
    except OSError:
        return False


def _match_assignment(
    assignments: list[Assignment],
    target_runtime_id: Optional[str],
) -> Optional[Assignment]:
    valid = [
        a
        for a in assignments
        if a.endpoint and a.runtime_proxy_url and a.runtime_proxy_token
    ]
    if not valid:
        return None
    if target_runtime_id is not None:
        for item in valid:
            if item.endpoint == target_runtime_id:
                return item
        return None
    return valid[0]


async def ensure_cli_runtime_attached(
    runtime_id: Optional[str] = None,
    auth_user_index: Optional[int] = None,
) -> tuple[str, int]:
    """
    Ensure the target Colab runtime is registered in the current CLI process's
    ``SessionManager``, re-attaching via TFE ``/assignments`` using the saved
    ``~/.colabmcp/cli_session.json`` pointer when needed.

    Returns ``(endpoint, auth_user_index)``.
    """
    saved = load_cli_session()
    target_id = runtime_id if runtime_id is not None else (saved.runtime_id if saved else None)
    effective_user = (
        auth_user_index
        if auth_user_index is not None
        else (saved.auth_user_index if saved else 0)
    )

    mgr = get_session_manager()
    in_mem = mgr.get_session(target_id)
    if (
        in_mem is not None
        and in_mem.assignment is not None
        and (target_id is None or in_mem.endpoint == target_id)
    ):
        return in_mem.endpoint, in_mem.auth_user_index

    async with ColabTFEClient(auth_user_index=effective_user, timeout=15.0) as tfe_client:
        assignments = await tfe_client.list_assignments()

    matched = _match_assignment(assignments, target_id)
    if matched is None or not matched.endpoint:
        if runtime_id is None or (saved is not None and saved.runtime_id == runtime_id):
            clear_cli_session()
        if target_id:
            raise RuntimeDisconnectedError(
                f"Runtime '{target_id}' is not active on this account. "
                "Run `colab-mcp connect-runtime` or `colab-mcp list-sessions`."
            )
        raise RuntimeDisconnectedError(
            "No active Colab runtime attached. Run `colab-mcp connect-runtime` first."
        )

    from colab_mcp.runtime.tools import _probe_kernel_status

    kernel_id: Optional[str] = saved.kernel_id if (saved and saved.runtime_id == matched.endpoint) else None
    kernel_state: Optional[str] = None
    if matched.runtime_proxy_url and matched.runtime_proxy_token:
        probed_id, probed_state = await _probe_kernel_status(
            matched.runtime_proxy_url,
            matched.runtime_proxy_token,
        )
        kernel_id = probed_id or kernel_id
        kernel_state = probed_state

    high_ram = (
        saved.high_ram
        if (saved and saved.runtime_id == matched.endpoint)
        else (str(matched.machine_shape or "").upper() in ("HM", "HIGH_RAM"))
    )
    acc = (matched.accelerator or (saved.accelerator if saved else "NONE") or "NONE").upper()
    var = (matched.variant or (saved.variant if saved else "DEFAULT") or "DEFAULT").upper()

    mgr.register_session(
        endpoint=matched.endpoint,
        auth_user_index=effective_user,
        assignment=matched,
        keepalive_daemon=None,
        high_ram=high_ram,
        kernel_id=kernel_id,
        kernel_state=kernel_state or "idle",
    )
    save_cli_session(
        CLISessionState(
            runtime_id=matched.endpoint,
            kernel_id=kernel_id,
            accelerator=acc,
            variant=var,
            high_ram=high_ram,
            auth_user_index=effective_user,
        )
    )
    return matched.endpoint, effective_user


async def cleanup_cli_runtime_resources() -> None:
    """Close local WebSockets and cancel local background tasks without unassigning the VM."""
    try:
        from colab_mcp.kernel.tools import clear_kernel_clients, reset_execution_registry

        await clear_kernel_clients()
        reset_execution_registry()
    except Exception:
        pass
    try:
        await get_session_manager().shutdown()
    except Exception:
        pass


__all__ = [
    "CLI_SESSION_FILENAME",
    "CLISessionState",
    "cleanup_cli_runtime_resources",
    "clear_cli_session",
    "ensure_cli_runtime_attached",
    "get_cli_session_file",
    "load_cli_session",
    "save_cli_session",
]
