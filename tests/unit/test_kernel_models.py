"""
Unit tests for colab_mcp.kernel.models (Jupyter v5.3 wire frames, ExecutionEvent,
status/interrupt response models, ExecutionRecord, and kernel exception hierarchy).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pytest

from colab_mcp.kernel.models import (
    AsyncExecutionResponse,
    ColabKernelError,
    ExecutionEvent,
    ExecutionLifecycleStatus,
    ExecutionNotFoundError,
    ExecutionRecord,
    ExecutionStatus,
    ExecutionStatusResponse,
    InterruptExecutionResponse,
    JupyterHeader,
    JupyterMessage,
    KernelConnectionError,
    KernelError,
    KernelExecutionError,
    KernelExecutionTimeoutError,
    KernelInterruptedError,
    KernelNotFoundError,
    KernelState,
    KernelTimeoutError,
    RawExecutionResult,
    build_execute_request,
    build_input_reply,
    build_interrupt_request,
    build_jupyter_header,
    build_jupyter_message,
    build_kernel_info_request,
    build_shutdown_request,
)


def test_build_jupyter_header_defaults_and_utc_date():
    hdr = build_jupyter_header("execute_request", session_id="sess-123")
    assert hdr["msg_type"] == "execute_request"
    assert hdr["session"] == "sess-123"
    assert hdr["username"] == "colab-mcp"
    assert hdr["version"] == "5.3"
    assert isinstance(hdr["msg_id"], str) and len(hdr["msg_id"]) > 10
    assert hdr["date"].endswith("Z")
    parsed_hdr = JupyterHeader(**hdr)
    assert parsed_hdr.msg_id == hdr["msg_id"]


def test_build_execute_request_wire_envelope():
    frame = build_execute_request(
        code="print('hello')",
        session_id="sess-abc",
        silent=False,
        store_history=True,
        allow_stdin=False,
        stop_on_error=True,
        msg_id="custom-msg-1",
    )
    assert frame["channel"] == "shell"
    assert frame["parent_header"] == {}
    assert frame["metadata"] == {}
    assert frame["buffers"] == []
    assert frame["header"]["msg_id"] == "custom-msg-1"
    assert frame["header"]["msg_type"] == "execute_request"
    assert frame["header"]["version"] == "5.3"
    assert frame["content"] == {
        "code": "print('hello')",
        "silent": False,
        "store_history": True,
        "user_expressions": {},
        "allow_stdin": False,
        "stop_on_error": True,
    }
    # Ensure JSON serializable and valid JupyterMessage
    roundtrip = json.loads(json.dumps(frame))
    model = JupyterMessage(**roundtrip)
    assert model.channel == "shell"
    assert model.header.msg_id == "custom-msg-1"


def test_build_control_and_stdin_frames():
    interrupt = build_interrupt_request(session_id="s1", msg_id="int-1")
    assert interrupt["channel"] == "control"
    assert interrupt["header"]["msg_type"] == "interrupt_request"
    assert interrupt["content"] == {}

    shutdown = build_shutdown_request(session_id="s1", restart=True)
    assert shutdown["channel"] == "control"
    assert shutdown["header"]["msg_type"] == "shutdown_request"
    assert shutdown["content"] == {"restart": True}

    kinfo = build_kernel_info_request(session_id="s1")
    assert kinfo["channel"] == "shell"
    assert kinfo["header"]["msg_type"] == "kernel_info_request"

    reply = build_input_reply("42", parent_header={"msg_id": "req-9"}, session_id="s1")
    assert reply["channel"] == "stdin"
    assert reply["header"]["msg_type"] == "input_reply"
    assert reply["parent_header"] == {"msg_id": "req-9"}
    assert reply["content"] == {"value": "42"}


@pytest.mark.parametrize(
    "msg_type,channel,content",
    [
        ("stream", "iopub", {"name": "stdout", "text": "ok\n"}),
        ("display_data", "iopub", {"data": {"image/png": "abc"}, "metadata": {}}),
        ("execute_result", "iopub", {"execution_count": 3, "data": {"text/plain": "3"}}),
        ("error", "iopub", {"ename": "ValueError", "evalue": "bad", "traceback": ["tb"]}),
        ("status", "iopub", {"execution_state": "idle"}),
        ("execute_reply", "shell", {"status": "ok", "execution_count": 3}),
    ],
)
def test_jupyter_message_parses_all_wire_frame_types(msg_type, channel, content):
    raw = build_jupyter_message(
        msg_type=msg_type,
        content=content,
        session_id="s-wire",
        channel=channel,
        parent_header={"msg_id": "parent-1"},
    )
    parsed = JupyterMessage(**raw)
    assert parsed.header.msg_type == msg_type
    assert parsed.channel == channel
    assert parsed.parent_header["msg_id"] == "parent-1"
    assert parsed.content == content


def test_execution_event_literals_and_serialization():
    events = [
        ExecutionEvent(event_type="started"),
        ExecutionEvent(event_type="stream", stream_name="stdout", text="hello\n"),
        ExecutionEvent(event_type="stream", stream_name="stderr", text="warn\n"),
        ExecutionEvent(event_type="display_data", mime_type="image/png", data_base64="AAAA"),
        ExecutionEvent(event_type="execute_result", mime_type="text/plain", text="42", execution_count=1),
        ExecutionEvent(event_type="completed", execution_count=1),
        ExecutionEvent(
            event_type="error",
            error_name="ZeroDivisionError",
            error_value="division by zero",
            sanitized_traceback="ZeroDivisionError: division by zero",
        ),
    ]
    for ev in events:
        assert ev.timestamp.tzinfo is not None
        dumped = json.loads(ev.model_dump_json())
        assert dumped["event_type"] == ev.event_type


def test_response_models_and_dict_compatibility():
    async_resp = AsyncExecutionResponse(execution_id="exec-1", runtime_id="ep-1")
    assert async_resp.status == "running"
    assert async_resp["execution_id"] == "exec-1"
    assert async_resp.get("runtime_id") == "ep-1"

    tier1 = InterruptExecutionResponse(runtime_id="ep-1")
    assert tier1.state_reset is False
    assert tier1.kernel_restarted is False
    assert tier1["state_reset"] is False
    assert tier1.get("warning") is None

    tier2 = InterruptExecutionResponse(
        runtime_id="ep-1",
        status="restarted",
        state_reset=True,
        kernel_restarted=True,
        warning="Variables cleared",
    )
    assert tier2["kernel_restarted"] is True
    assert tier2.get("warning") == "Variables cleared"

    status_resp = ExecutionStatusResponse(
        execution_id="exec-1",
        status=ExecutionLifecycleStatus.RUNNING.value,
        elapsed_seconds=2.5,
        partial_stdout_tail="step 1",
        has_error=False,
    )
    assert ExecutionStatus is ExecutionStatusResponse
    assert status_resp["status"] == "running"
    assert status_resp.get("missing", "fallback") == "fallback"


def test_execution_record_to_status_transitions():
    start = time.monotonic() - 1.5
    rec = ExecutionRecord(
        execution_id="exec-99",
        runtime_id="ep-9",
        msg_id="m-99",
        code="print('hi')",
        status=ExecutionLifecycleStatus.RUNNING.value,
        started_at=start,
        stdout_buffer="A" * 1500,
    )
    st_running = rec.to_status(tail_chars=200, current_monotonic=start + 1.5)
    assert st_running.elapsed_seconds == pytest.approx(1.5, abs=0.05)
    assert len(st_running.partial_stdout_tail) == 200

    rec.status = ExecutionLifecycleStatus.COMPLETED.value
    rec.completed_at = start + 3.25
    st_done = rec.to_status()
    assert st_done.status == "completed"
    assert st_done.elapsed_seconds == pytest.approx(3.25, abs=0.01)


def test_raw_execution_result_defaults():
    res = RawExecutionResult(msg_id="m-1", execution_count=4)
    assert res.status == "ok"
    assert res.has_error is False
    assert res.iopub_messages == []


def test_kernel_exception_hierarchy_and_mro():
    assert KernelError is ColabKernelError
    assert KernelTimeoutError is KernelExecutionTimeoutError

    conn_err = KernelConnectionError("ws down")
    assert isinstance(conn_err, ColabKernelError)
    assert isinstance(conn_err, ConnectionError)
    assert isinstance(conn_err, RuntimeError)

    to_err = KernelExecutionTimeoutError("timeout")
    assert isinstance(to_err, ColabKernelError)
    assert isinstance(to_err, TimeoutError)
    assert isinstance(to_err, RuntimeError)

    not_found = ExecutionNotFoundError("exec-404")
    assert isinstance(not_found, ColabKernelError)
    assert isinstance(not_found, KeyError)

    for cls in (KernelExecutionError, KernelInterruptedError, KernelNotFoundError):
        assert issubclass(cls, ColabKernelError)

    assert KernelState.IDLE.value == "idle"
    assert KernelState.RECONNECTING.value == "reconnecting"
