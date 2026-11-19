"""
Jupyter Messaging Protocol v5.3/v5.4 Wire Envelopes, Execution Models, and Domain Exceptions.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ============================================================================
# Domain Exceptions
# ============================================================================


class ColabKernelError(RuntimeError):
    """Base exception for Colab Jupyter kernel communication and execution errors."""


KernelError = ColabKernelError


class KernelConnectionError(ColabKernelError, ConnectionError):
    """Raised on kernel discovery failure, WebSocket handshake failure, or reconnect exhaustion."""


class KernelExecutionTimeoutError(ColabKernelError, TimeoutError):
    """Raised when kernel execution exceeds timeout_seconds and auto-SIGINT is dispatched."""


KernelTimeoutError = KernelExecutionTimeoutError


class KernelExecutionError(ColabKernelError):
    """Raised when a kernel execution fails unexpectedly."""


class KernelInterruptedError(ColabKernelError):
    """Raised when an active execution is interrupted by SIGINT or kernel restart."""


class KernelNotFoundError(ColabKernelError):
    """Raised when a requested Jupyter kernel ID does not exist (HTTP 404)."""


class ExecutionNotFoundError(ColabKernelError, KeyError):
    """Raised when an execution_id is missing from the ExecutionRegistry."""


# ============================================================================
# Jupyter Messaging Protocol v5.3/v5.4 Wire Models & Builders
# ============================================================================


JupyterChannel = Literal["shell", "iopub", "stdin", "control"]


class KernelState(str, Enum):
    """Lifecycle execution states for a Colab Jupyter kernel."""

    STARTING = "starting"
    IDLE = "idle"
    BUSY = "busy"
    RESTARTING = "restarting"
    AUTORESTARTING = "autorestarting"
    DEAD = "dead"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class JupyterHeader(BaseModel):
    """Jupyter Messaging Protocol v5.3/v5.4 frame header."""

    model_config = ConfigDict(extra="allow")

    msg_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str = "colab-mcp"
    session: str = ""
    msg_type: str
    version: str = "5.3"
    date: str = Field(default_factory=_utc_iso_now)


class JupyterMessage(BaseModel):
    """Jupyter Messaging Protocol v5.3/v5.4 multiplexed WebSocket frame."""

    model_config = ConfigDict(extra="allow")

    header: JupyterHeader
    parent_header: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    content: Dict[str, Any] = Field(default_factory=dict)
    channel: str = "shell"
    buffers: List[Any] = Field(default_factory=list)


def build_jupyter_header(
    msg_type: str,
    session_id: str = "",
    username: str = "colab-mcp",
    msg_id: Optional[str] = None,
    version: str = "5.3",
) -> Dict[str, Any]:
    """Construct a JSON-serializable Jupyter v5.3 header dictionary."""
    return {
        "msg_id": msg_id or str(uuid.uuid4()),
        "username": username,
        "session": session_id,
        "msg_type": msg_type,
        "version": version,
        "date": _utc_iso_now(),
    }


def build_jupyter_message(
    msg_type: str,
    content: Dict[str, Any],
    session_id: str = "",
    channel: JupyterChannel | str = "shell",
    parent_header: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    username: str = "colab-mcp",
    msg_id: Optional[str] = None,
    version: str = "5.3",
    buffers: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Construct a JSON-serializable Jupyter v5.3 wire message dictionary."""
    return {
        "header": build_jupyter_header(
            msg_type=msg_type,
            session_id=session_id,
            username=username,
            msg_id=msg_id,
            version=version,
        ),
        "parent_header": dict(parent_header) if parent_header is not None else {},
        "metadata": dict(metadata) if metadata is not None else {},
        "content": dict(content),
        "channel": channel,
        "buffers": list(buffers) if buffers is not None else [],
    }


def build_execute_request(
    code: str,
    session_id: str = "",
    silent: bool = False,
    store_history: bool = True,
    user_expressions: Optional[Dict[str, Any]] = None,
    allow_stdin: bool = False,
    stop_on_error: bool = True,
    username: str = "colab-mcp",
    msg_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a Jupyter shell ``execute_request`` frame (TECH_SPEC §5.3)."""
    return build_jupyter_message(
        msg_type="execute_request",
        content={
            "code": code,
            "silent": silent,
            "store_history": store_history,
            "user_expressions": dict(user_expressions) if user_expressions is not None else {},
            "allow_stdin": allow_stdin,
            "stop_on_error": stop_on_error,
        },
        session_id=session_id,
        channel="shell",
        username=username,
        msg_id=msg_id,
    )


def build_interrupt_request(
    session_id: str = "",
    username: str = "colab-mcp",
    msg_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a Jupyter control ``interrupt_request`` frame."""
    return build_jupyter_message(
        msg_type="interrupt_request",
        content={},
        session_id=session_id,
        channel="control",
        username=username,
        msg_id=msg_id,
    )


def build_shutdown_request(
    session_id: str = "",
    restart: bool = False,
    username: str = "colab-mcp",
    msg_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a Jupyter control ``shutdown_request`` frame."""
    return build_jupyter_message(
        msg_type="shutdown_request",
        content={"restart": restart},
        session_id=session_id,
        channel="control",
        username=username,
        msg_id=msg_id,
    )


def build_kernel_info_request(
    session_id: str = "",
    username: str = "colab-mcp",
    msg_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a Jupyter shell ``kernel_info_request`` frame."""
    return build_jupyter_message(
        msg_type="kernel_info_request",
        content={},
        session_id=session_id,
        channel="shell",
        username=username,
        msg_id=msg_id,
    )


def build_input_reply(
    value: str,
    parent_header: Dict[str, Any],
    session_id: str = "",
    username: str = "colab-mcp",
    msg_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a Jupyter stdin ``input_reply`` frame."""
    return build_jupyter_message(
        msg_type="input_reply",
        content={"value": value},
        session_id=session_id,
        channel="stdin",
        parent_header=parent_header,
        username=username,
        msg_id=msg_id,
    )


# ============================================================================
# Execution Lifecycle & MCP Tool Response Models
# ============================================================================


class ExecutionLifecycleStatus(str, Enum):
    """Execution lifecycle status values."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"
    NOT_FOUND = "not_found"


class ExecutionEvent(BaseModel):
    """Streaming execution event yielded by ``execute_snippet`` (TECH_SPEC §2.4 Contract 3)."""

    event_type: Literal["started", "stream", "display_data", "execute_result", "completed", "error"]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stream_name: Optional[Literal["stdout", "stderr"]] = None
    text: Optional[str] = None
    mime_type: Optional[str] = None
    data_base64: Optional[str] = None
    execution_count: Optional[int] = None
    error_name: Optional[str] = None
    error_value: Optional[str] = None
    sanitized_traceback: Optional[str] = None
    raw_message: Optional[Dict[str, Any]] = None


class RawExecutionResult(BaseModel):
    """Raw structured execution capture consumed by Tool 8 and M4 NotebookRunner."""

    execution_id: Optional[str] = None
    msg_id: str = ""
    status: Literal["ok", "error", "abort", "timed_out", "interrupted"] = "ok"
    execution_count: Optional[int] = None
    duration_seconds: float = 0.0
    iopub_messages: List[Dict[str, Any]] = Field(default_factory=list)
    reply_message: Optional[Dict[str, Any]] = None
    has_error: bool = False
    error_name: Optional[str] = None
    error_value: Optional[str] = None
    traceback: Optional[List[str]] = None


class _DictCompatMixin:
    """Provides dict-like .get() and __getitem__() access for Pydantic response models."""

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


class ExecutionStatusResponse(_DictCompatMixin, BaseModel):
    """Response schema for ``colab_get_execution_status`` (TECH_SPEC §7.3.3)."""

    execution_id: str
    status: str
    elapsed_seconds: float = 0.0
    partial_stdout_tail: str = ""
    has_error: bool = False
    execution_count: Optional[int] = None
    error_name: Optional[str] = None
    error_value: Optional[str] = None


ExecutionStatus = ExecutionStatusResponse


class AsyncExecutionResponse(_DictCompatMixin, BaseModel):
    """Immediate response returned when ``colab_execute_code(..., wait_for_completion=False)``."""

    execution_id: str
    status: str = "running"
    runtime_id: Optional[str] = None
    message: str = "Execution dispatched asynchronously."


class InterruptExecutionResponse(_DictCompatMixin, BaseModel):
    """Response schema for ``colab_interrupt_execution`` (TECH_SPEC §7.3.3)."""

    runtime_id: Optional[str] = None
    status: str = "interrupted"
    state_reset: bool = False
    kernel_restarted: bool = False
    kernel_state: str = "idle"
    warning: Optional[str] = None
    message: str = (
        "SIGINT interrupt signal dispatched; execution halted while preserving in-memory state."
    )


class ExecutionRecord(BaseModel):
    """In-memory execution record tracked by ExecutionRegistry."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    execution_id: str
    runtime_id: Optional[str] = None
    msg_id: str = ""
    code: str = ""
    status: str = ExecutionLifecycleStatus.QUEUED.value
    started_at: float = 0.0
    completed_at: Optional[float] = None
    execution_count: Optional[int] = None
    iopub_messages: List[Dict[str, Any]] = Field(default_factory=list)
    reply_message: Optional[Dict[str, Any]] = None
    stdout_buffer: str = ""
    has_error: bool = False
    error_name: Optional[str] = None
    error_value: Optional[str] = None
    traceback: Optional[List[str]] = None
    task: Optional[Any] = Field(default=None, exclude=True)

    def to_status(self, tail_chars: int = 1000, current_monotonic: Optional[float] = None) -> ExecutionStatusResponse:
        """Build an ExecutionStatusResponse snapshot from this record."""
        if self.completed_at is not None:
            elapsed = max(0.0, self.completed_at - self.started_at)
        elif current_monotonic is not None and self.started_at > 0:
            elapsed = max(0.0, current_monotonic - self.started_at)
        else:
            elapsed = 0.0
        tail = self.stdout_buffer[-tail_chars:] if tail_chars > 0 else ""
        return ExecutionStatusResponse(
            execution_id=self.execution_id,
            status=self.status,
            elapsed_seconds=round(elapsed, 3),
            partial_stdout_tail=tail,
            has_error=self.has_error,
            execution_count=self.execution_count,
            error_name=self.error_name,
            error_value=self.error_value,
        )


__all__ = [
    "AsyncExecutionResponse",
    "ColabKernelError",
    "ExecutionEvent",
    "ExecutionLifecycleStatus",
    "ExecutionNotFoundError",
    "ExecutionRecord",
    "ExecutionStatus",
    "ExecutionStatusResponse",
    "InterruptExecutionResponse",
    "JupyterChannel",
    "JupyterHeader",
    "JupyterMessage",
    "KernelConnectionError",
    "KernelError",
    "KernelExecutionError",
    "KernelExecutionTimeoutError",
    "KernelInterruptedError",
    "KernelNotFoundError",
    "KernelState",
    "KernelTimeoutError",
    "RawExecutionResult",
    "build_execute_request",
    "build_input_reply",
    "build_interrupt_request",
    "build_jupyter_header",
    "build_jupyter_message",
    "build_kernel_info_request",
    "build_shutdown_request",
]
