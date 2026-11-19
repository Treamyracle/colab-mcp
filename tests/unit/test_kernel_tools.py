"""
Unit tests for colab_mcp.kernel.tools (Tools 8, 10, 11, 12), ExecutionRegistry,
parameter bounds, StrictBool enforcement, SessionManager integration, and egress redaction.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, ImageContent, TextContent

from colab_mcp.kernel.mime_converter import convert_iopub_to_mcp_blocks
from colab_mcp.kernel.models import (
    InterruptExecutionResponse,
    KernelExecutionTimeoutError,
    RawExecutionResult,
)
from colab_mcp.kernel.tools import (
    ExecutionRegistry,
    clear_kernel_clients,
    colab_execute_code,
    colab_get_cell_output,
    colab_get_execution_status,
    colab_interrupt_execution,
    get_active_kernel_client,
    get_execution_registry,
    register_kernel_tools,
    reset_execution_registry,
    reset_kernel_clients,
)
from colab_mcp.runtime.models import Assignment, RuntimeDisconnectedError
from colab_mcp.runtime.session_manager import get_session_manager, reset_session_manager

SAMPLE_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _extract_result_dict(tool_result: Any) -> Dict[str, Any]:
    if isinstance(tool_result, CallToolResult):
        if tool_result.structured_content is not None:
            return dict(tool_result.structured_content)
        if tool_result.content:
            raw_text = getattr(tool_result.content[0], "text", "")
            try:
                parsed = json.loads(raw_text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            return {"text": raw_text}
    if isinstance(tool_result, tuple) and len(tool_result) > 1 and isinstance(tool_result[1], dict):
        return tool_result[1]
    blocks = tool_result[0] if isinstance(tool_result, tuple) else tool_result
    if isinstance(blocks, list) and blocks and hasattr(blocks[0], "text"):
        return json.loads(blocks[0].text)
    raise AssertionError(f"Unexpected tool_result shape: {tool_result!r}")


@pytest.fixture(autouse=True)
async def isolated_kernel_state():
    reset_session_manager()
    reset_execution_registry()
    reset_kernel_clients()
    yield
    await clear_kernel_clients()
    reset_execution_registry()
    reset_session_manager()


@pytest.fixture
def registered_runtime():
    mgr = get_session_manager()
    assignment = Assignment(
        endpoint="colab-ep-test-1",
        accelerator="T4",
        variant="GPU",
        runtime_proxy_url="https://mock-proxy.local/tun/m/colab-ep-test-1",
        runtime_proxy_token="proxy-jwt-secret",
        token_expires_in=3600,
    )
    record = mgr.register_session(
        endpoint="colab-ep-test-1",
        assignment=assignment,
        kernel_id="kernel-uuid-1",
        kernel_state="idle",
    )
    return mgr, record


@pytest.fixture
def kernel_mcp_server() -> MCPServer:
    mcp = MCPServer("kernel-tools-unit-test")
    register_kernel_tools(mcp)
    return mcp


# ============================================================================
# Tool 8: colab_execute_code
# ============================================================================


@pytest.mark.asyncio
async def test_execute_code_sync_chronological_and_activity_reset(registered_runtime):
    mgr, record = registered_runtime
    record.last_activity_time = 1.0

    mock_client = AsyncMock()
    mock_client.runtime_id = record.endpoint
    mock_client.execute.return_value = [
        TextContent(type="text", text="Epoch 1"),
        ImageContent(type="image", data=SAMPLE_PNG_B64, mimeType="image/png"),
        TextContent(type="text", text="Done"),
    ]

    with patch("colab_mcp.kernel.tools.get_active_kernel_client", return_value=mock_client):
        blocks = await colab_execute_code(code="print('Epoch 1')")

    assert len(blocks) == 3
    assert isinstance(blocks[0], TextContent)
    assert isinstance(blocks[1], ImageContent)
    assert isinstance(blocks[2], TextContent)
    assert record.last_activity_time > 1.0


@pytest.mark.asyncio
async def test_execute_code_async_dispatch_and_fastmcp_no_wrapper(
    registered_runtime, kernel_mcp_server: MCPServer
):
    _, record = registered_runtime

    async def slow_execute(**kwargs):
        await asyncio.sleep(0.02)
        return [TextContent(type="text", text="async finished\n")]

    mock_client = AsyncMock()
    mock_client.runtime_id = record.endpoint
    mock_client.execute.side_effect = slow_execute

    with patch("colab_mcp.kernel.tools.get_active_kernel_client", return_value=mock_client):
        res = await kernel_mcp_server.call_tool(
            "colab_execute_code",
            {"code": "import time; time.sleep(0.02)", "wait_for_completion": False},
        )
        data = _extract_result_dict(res)
        assert "result" not in data  # Must NOT be FastMCP wrapped
        assert data["execution_id"].startswith("exec-")
        assert data["status"] == "running"

        # Wait for background task completion
        reg = get_execution_registry()
        task = reg[data["execution_id"]]["task"]
        await task

        status = await colab_get_execution_status(data["execution_id"])
        assert status["status"] == "completed"
        assert status["has_error"] is False


@pytest.mark.asyncio
async def test_execute_code_unconnected_runtime_raises():
    with pytest.raises(RuntimeDisconnectedError):
        await colab_execute_code(code="print(1)")


@pytest.mark.asyncio
async def test_execute_code_timeout_returns_error_call_tool_result(
    registered_runtime, kernel_mcp_server: MCPServer
):
    _, record = registered_runtime
    mock_client = AsyncMock()
    mock_client.runtime_id = record.endpoint
    mock_client.execute.side_effect = KernelExecutionTimeoutError("Timed out after 5s")

    with patch("colab_mcp.kernel.tools.get_active_kernel_client", return_value=mock_client):
        res = await kernel_mcp_server.call_tool(
            "colab_execute_code",
            {"code": "while True: pass", "timeout_seconds": 5},
        )
    assert isinstance(res, CallToolResult)
    assert res.is_error is True
    assert "Timed out" in res.content[0].text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args",
    [
        {"code": "pass", "timeout_seconds": 4},
        {"code": "pass", "timeout_seconds": 3601},
        {"code": "pass", "timeout_seconds": True},
        {"code": "pass", "max_text_lines": 9},
        {"code": "pass", "max_text_lines": 10001},
        {"code": "pass", "max_output_bytes": 1023},
        {"code": "pass", "max_output_bytes": 10485761},
        {"code": "pass", "max_images": 0},
        {"code": "pass", "max_images": 51},
        {"code": "pass", "wait_for_completion": "yes"},
        {"code": "pass", "store_history": "no"},
        {},
    ],
)
async def test_execute_code_bounds_and_type_validation(kernel_mcp_server: MCPServer, args):
    with pytest.raises((ToolError, ValueError, TypeError)):
        await kernel_mcp_server.call_tool("colab_execute_code", args)


# ============================================================================
# Tool 10 & 11: colab_get_execution_status & colab_get_cell_output
# ============================================================================


@pytest.mark.asyncio
async def test_execution_status_and_cell_output_filtering_and_truncation(
    registered_runtime, kernel_mcp_server: MCPServer
):
    _, record = registered_runtime
    record.last_activity_time = 2.0

    reg = get_execution_registry()
    exec_id = "exec-unit-001"
    reg.create_execution(exec_id, record.endpoint, "print('hi')")
    for i in range(60):
        reg.record_iopub_message(
            exec_id,
            {
                "header": {"msg_type": "stream"},
                "content": {"name": "stdout", "text": f"Batch line {i} token=ya29.secret_tok_99\n"},
            },
        )
    for _ in range(3):
        reg.record_iopub_message(
            exec_id,
            {
                "header": {"msg_type": "display_data"},
                "content": {"data": {"image/png": SAMPLE_PNG_B64}},
            },
        )
    reg.mark_completed(exec_id, status="completed", has_error=False)

    # Poll status via FastMCP
    st_res = await kernel_mcp_server.call_tool(
        "colab_get_execution_status", {"execution_id": exec_id}
    )
    st_data = _extract_result_dict(st_res)
    assert st_data["execution_id"] == exec_id
    assert st_data["status"] == "completed"
    assert st_data["elapsed_seconds"] >= 0.0
    assert "ya29.secret_tok_99" not in st_data["partial_stdout_tail"]
    assert "[REDACTED]" in st_data["partial_stdout_tail"]
    assert record.last_activity_time > 2.0

    # Retrieve cell output with include_images=False and tight line limit
    out_no_img = await colab_get_cell_output(
        execution_id=exec_id,
        include_images=False,
        max_text_lines=20,
        max_output_bytes=2048,
    )
    assert all(isinstance(b, TextContent) for b in out_no_img)
    combined = "\n".join(b.text for b in out_no_img)
    assert "[OUTPUT TRUNCATED" in combined
    assert "ya29.secret_tok_99" not in combined

    # Retrieve cell output with max_images=1
    out_limited = await colab_get_cell_output(
        execution_id=exec_id,
        include_images=True,
        max_images=1,
    )
    imgs = [b for b in out_limited if isinstance(b, ImageContent)]
    txts = [b for b in out_limited if isinstance(b, TextContent)]
    assert len(imgs) == 1
    assert imgs[0].data == SAMPLE_PNG_B64
    assert any("2 additional plot images omitted" in t.text for t in txts)


@pytest.mark.asyncio
async def test_status_and_output_unknown_id_and_strict_bool(kernel_mcp_server: MCPServer):
    st = await colab_get_execution_status("exec-does-not-exist")
    assert st["status"] == "not_found"

    out = await colab_get_cell_output("exec-does-not-exist")
    assert len(out) == 1 and "No execution found" in out[0].text

    with pytest.raises((ToolError, ValueError, TypeError)):
        await kernel_mcp_server.call_tool(
            "colab_get_cell_output",
            {"execution_id": "exec-abc", "include_images": "yes"},
        )

    with pytest.raises((ToolError, ValueError, TypeError)):
        await kernel_mcp_server.call_tool(
            "colab_get_execution_status",
            {"execution_id": ""},
        )


# ============================================================================
# Tool 12: colab_interrupt_execution
# ============================================================================


@pytest.mark.asyncio
async def test_interrupt_execution_tier1_and_tier2_session_sync(
    registered_runtime, kernel_mcp_server: MCPServer
):
    _, record = registered_runtime

    # Tier 1 SIGINT
    mock_client = AsyncMock()
    mock_client.runtime_id = record.endpoint
    mock_client.kernel_id = "kernel-uuid-1"
    mock_client.interrupt_with_escalation.return_value = InterruptExecutionResponse(
        runtime_id=record.endpoint,
        status="interrupted",
        state_reset=False,
        kernel_restarted=False,
        kernel_state="idle",
    )

    with patch("colab_mcp.kernel.tools.get_active_kernel_client", return_value=mock_client):
        res = await kernel_mcp_server.call_tool("colab_interrupt_execution", {})
    data = _extract_result_dict(res)
    assert data["status"] == "interrupted"
    assert data["state_reset"] is False
    assert data["kernel_restarted"] is False

    # Tier 2 escalated restart updates SessionManager kernel_id and kernel_state
    mock_client.kernel_id = "kernel-uuid-restarted"
    mock_client.interrupt_with_escalation.return_value = InterruptExecutionResponse(
        runtime_id=record.endpoint,
        status="restarted",
        state_reset=True,
        kernel_restarted=True,
        kernel_state="idle",
        warning="SIGINT did not halt kernel within 5.0s.",
    )
    with patch("colab_mcp.kernel.tools.get_active_kernel_client", return_value=mock_client):
        res2 = await colab_interrupt_execution(runtime_id=record.endpoint)
    assert res2["state_reset"] is True
    assert res2["kernel_restarted"] is True
    assert res2["warning"] is not None
    assert record.kernel_id == "kernel-uuid-restarted"
    assert record.kernel_state == "idle"

    with pytest.raises((ToolError, ValueError, TypeError)):
        await kernel_mcp_server.call_tool("colab_interrupt_execution", {"runtime_id": False})


# ============================================================================
# Egress Redaction & HTML / MIME Edge Cases
# ============================================================================


def test_mime_converter_redaction_and_html_conversion():
    messages = [
        {
            "header": {"msg_type": "stream"},
            "content": {
                "name": "stdout",
                "text": "oauth=ya29.abc123 refresh=1//0xyz999 cookie=SAPISID=secret_val\n",
            },
        },
        {
            "header": {"msg_type": "display_data"},
            "content": {
                "data": {
                    "text/html": "<style>.x{}</style><div><b>Accuracy</b>: 99.1%</div><script>alert(1)</script>"
                }
            },
        },
        {
            "header": {"msg_type": "execute_result"},
            "content": {"data": {"image/png": SAMPLE_PNG_B64}},
        },
    ]
    blocks = convert_iopub_to_mcp_blocks(messages)
    assert len(blocks) == 2
    assert isinstance(blocks[0], TextContent)
    assert "ya29.abc123" not in blocks[0].text
    assert "1//0xyz999" not in blocks[0].text
    assert "secret_val" not in blocks[0].text
    assert "[REDACTED]" in blocks[0].text
    assert "Accuracy : 99.1%" in blocks[0].text or "Accuracy" in blocks[0].text
    assert "alert(1)" not in blocks[0].text
    assert isinstance(blocks[1], ImageContent)
    assert blocks[1].data == SAMPLE_PNG_B64


@pytest.mark.asyncio
async def test_fastmcp_tool_registration_schemas(kernel_mcp_server: MCPServer):
    tools = await kernel_mcp_server.list_tools()
    by_name = {t.name: t for t in tools}
    assert set(by_name.keys()) == {
        "colab_execute_code",
        "colab_get_execution_status",
        "colab_get_cell_output",
        "colab_interrupt_execution",
    }
    assert by_name["colab_execute_code"].input_schema["properties"]["wait_for_completion"]["type"] == "boolean"
    assert by_name["colab_get_cell_output"].input_schema["properties"]["include_images"]["type"] == "boolean"


@pytest.mark.asyncio
async def test_get_active_kernel_client_reuses_and_replaces_on_kernel_change(registered_runtime):
    _, record = registered_runtime
    fake_ws = AsyncMock()
    fake_ws.closed = False
    fake_ws.recv = AsyncMock(return_value=AsyncMock())  # causes _reader_loop to exit cleanly

    with patch("colab_mcp.kernel.client.connect_colab_kernel_ws", return_value=fake_ws):
        c1 = await get_active_kernel_client(record.endpoint)
        c2 = await get_active_kernel_client(record.endpoint)
        assert c1 is c2

        # Change kernel_id -> client must be replaced
        record.kernel_id = "kernel-uuid-2"
        c3 = await get_active_kernel_client(record.endpoint)
        assert c3 is not c1
        assert c3.kernel_id == "kernel-uuid-2"
