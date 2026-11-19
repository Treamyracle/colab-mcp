"""
Live Local WebSocket Mock Integration Tests for colab_mcp.kernel.client.

Uses a real loopback ``websockets.serve`` server on ``127.0.0.1:0`` and
``httpx.MockTransport`` for REST calls (zero real Colab network traffic).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import patch

import httpx
import pytest
import websockets
from mcp.types import ImageContent, TextContent

from colab_mcp.kernel.client import KernelWebSocketClient, connect_colab_kernel_ws
from colab_mcp.kernel.models import KernelExecutionTimeoutError

SAMPLE_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _reply_frame(
    msg_type: str,
    parent_msg_id: str,
    channel: str,
    content: Dict[str, Any],
) -> str:
    return json.dumps(
        {
            "header": {
                "msg_id": f"srv-{parent_msg_id}-{msg_type}",
                "username": "kernel",
                "session": "srv-session",
                "msg_type": msg_type,
                "version": "5.3",
            },
            "parent_header": {"msg_id": parent_msg_id},
            "metadata": {},
            "content": content,
            "channel": channel,
            "buffers": [],
        }
    )


@pytest.mark.asyncio
async def test_live_ws_handshake_headers_and_absent_subprotocol():
    """Verify real websockets==16.0 upgrade omits Sec-WebSocket-Protocol and sends auth headers."""
    captured_headers: Dict[str, str] = {}

    async def handler(conn):
        captured_headers.update({k.lower(): v for k, v in conn.request.headers.items()})
        await conn.wait_closed()

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        proxy_url = f"http://127.0.0.1:{port}"
        ws = await connect_colab_kernel_ws(
            proxy_url=proxy_url,
            proxy_token="live-mock-token-777",
            kernel_id="kern-1",
            client_agent="colab-mcp",
        )
        await ws.close()

    assert "sec-websocket-protocol" not in captured_headers
    assert captured_headers.get("x-colab-runtime-proxy-token") == "live-mock-token-777"
    assert "colab-mcp" in captured_headers.get("x-colab-client-agent", "").lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_before_idle", [True, False])
async def test_live_ws_execute_dual_completion_both_orders(reply_before_idle: bool):
    """
    Verify execute() produces chronological [TextContent, ImageContent, TextContent]
    and waits for BOTH execute_reply and status: idle regardless of arrival order.
    """

    async def handler(conn):
        async for raw in conn:
            req = json.loads(raw)
            msg_id = req["header"]["msg_id"]
            await conn.send(_reply_frame("status", msg_id, "iopub", {"execution_state": "busy"}))
            await conn.send(_reply_frame("stream", msg_id, "iopub", {"name": "stdout", "text": "Step 1\n"}))
            await conn.send(
                _reply_frame(
                    "display_data",
                    msg_id,
                    "iopub",
                    {"data": {"image/png": SAMPLE_PNG_B64, "text/plain": "<Figure>"}},
                )
            )
            await conn.send(_reply_frame("stream", msg_id, "iopub", {"name": "stdout", "text": "Step 2\n"}))
            if reply_before_idle:
                await conn.send(_reply_frame("execute_reply", msg_id, "shell", {"status": "ok", "execution_count": 7}))
                await asyncio.sleep(0.02)
                await conn.send(_reply_frame("status", msg_id, "iopub", {"execution_state": "idle"}))
            else:
                await conn.send(_reply_frame("status", msg_id, "iopub", {"execution_state": "idle"}))
                await asyncio.sleep(0.02)
                await conn.send(_reply_frame("execute_reply", msg_id, "shell", {"status": "ok", "execution_count": 7}))

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = KernelWebSocketClient(
            proxy_url=f"http://127.0.0.1:{port}",
            proxy_token="tok-1",
            kernel_id="kern-1",
        )
        await client.connect()
        try:
            blocks = await client.execute("print('Step 1'); plot(); print('Step 2')")
            assert len(blocks) == 3
            assert isinstance(blocks[0], TextContent) and "Step 1" in blocks[0].text
            assert isinstance(blocks[1], ImageContent) and blocks[1].mime_type == "image/png"
            assert isinstance(blocks[2], TextContent) and "Step 2" in blocks[2].text
        finally:
            await client.disconnect()


@pytest.mark.asyncio
async def test_live_ws_execute_snippet_and_execute_raw_for_m4():
    """Verify streaming execute_snippet() events and raw AST capture via execute_raw()."""

    async def handler(conn):
        async for raw in conn:
            req = json.loads(raw)
            msg_id = req["header"]["msg_id"]
            code = req["content"]["code"]
            await conn.send(_reply_frame("status", msg_id, "iopub", {"execution_state": "busy"}))
            if "fail" in code:
                await conn.send(
                    _reply_frame(
                        "error",
                        msg_id,
                        "iopub",
                        {
                            "ename": "ZeroDivisionError",
                            "evalue": "division by zero",
                            "traceback": ["\x1b[31mZeroDivisionError\x1b[0m: division by zero"],
                        },
                    )
                )
                await conn.send(
                    _reply_frame(
                        "execute_reply",
                        msg_id,
                        "shell",
                        {"status": "error", "execution_count": 2, "ename": "ZeroDivisionError", "evalue": "division by zero"},
                    )
                )
            else:
                await conn.send(_reply_frame("stream", msg_id, "iopub", {"name": "stdout", "text": "hello\n"}))
                await conn.send(
                    _reply_frame(
                        "execute_result",
                        msg_id,
                        "iopub",
                        {"execution_count": 1, "data": {"text/plain": "42"}},
                    )
                )
                await conn.send(_reply_frame("execute_reply", msg_id, "shell", {"status": "ok", "execution_count": 1}))
            await conn.send(_reply_frame("status", msg_id, "iopub", {"execution_state": "idle"}))

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = KernelWebSocketClient(
            proxy_url=f"http://127.0.0.1:{port}",
            proxy_token="tok-2",
            kernel_id="kern-2",
        )
        await client.connect()
        try:
            events = []
            async for ev in client.execute_snippet("print('hello'); 42"):
                events.append(ev)
            assert [e.event_type for e in events] == ["started", "stream", "execute_result", "completed"]
            assert events[1].text == "hello\n"
            assert events[2].text == "42"
            assert events[3].execution_count == 1

            raw_res = await client.execute_raw("fail()")
            assert raw_res.has_error is True
            assert raw_res.status == "error"
            assert raw_res.error_name == "ZeroDivisionError"
            assert raw_res.execution_count == 2
            assert len(raw_res.iopub_messages) == 1
        finally:
            await client.disconnect()


@pytest.mark.asyncio
async def test_live_ws_concurrent_multiplexing_zero_crosstalk():
    """Run two concurrent executions over the same live WS with interleaved frames."""

    async def handler(conn):
        async def handle_one(req: Dict[str, Any]):
            msg_id = req["header"]["msg_id"]
            label = "A" if "code_a" in req["content"]["code"] else "B"
            delay = 0.03 if label == "A" else 0.01
            await conn.send(_reply_frame("stream", msg_id, "iopub", {"name": "stdout", "text": f"{label}-1\n"}))
            await asyncio.sleep(delay)
            await conn.send(_reply_frame("stream", msg_id, "iopub", {"name": "stdout", "text": f"{label}-2\n"}))
            await conn.send(_reply_frame("execute_reply", msg_id, "shell", {"status": "ok", "execution_count": 1}))
            await conn.send(_reply_frame("status", msg_id, "iopub", {"execution_state": "idle"}))

        tasks = []
        async for raw in conn:
            tasks.append(asyncio.create_task(handle_one(json.loads(raw))))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = KernelWebSocketClient(
            proxy_url=f"http://127.0.0.1:{port}",
            proxy_token="tok-mux",
            kernel_id="kern-mux",
        )
        await client.connect()
        try:
            res_a, res_b = await asyncio.gather(
                client.execute("code_a()"),
                client.execute("code_b()"),
            )
            text_a = "\n".join(b.text for b in res_a if isinstance(b, TextContent))
            text_b = "\n".join(b.text for b in res_b if isinstance(b, TextContent))
            assert "A-1" in text_a and "A-2" in text_a and "B-" not in text_a
            assert "B-1" in text_b and "B-2" in text_b and "A-" not in text_b
        finally:
            await client.disconnect()


@pytest.mark.asyncio
async def test_live_ws_timeout_dispatches_auto_sigint():
    """Slow execution exceeding timeout_seconds sends interrupt_request and raises KernelExecutionTimeoutError."""
    received_types: List[str] = []

    async def handler(conn):
        async for raw in conn:
            req = json.loads(raw)
            received_types.append(req["header"]["msg_type"])

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = KernelWebSocketClient(
            proxy_url=f"http://127.0.0.1:{port}",
            proxy_token="tok-to",
            kernel_id="kern-to",
        )
        await client.connect()
        try:
            with patch("httpx.AsyncClient", return_value=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda req: httpx.Response(204))
            )):
                with pytest.raises(KernelExecutionTimeoutError):
                    await client.execute_raw("import time; time.sleep(5)", timeout_seconds=0.1)  # type: ignore[arg-type]
            await asyncio.sleep(0.05)
            assert "execute_request" in received_types
            assert "interrupt_request" in received_types
        finally:
            await client.disconnect()


@pytest.mark.asyncio
async def test_live_ws_abnormal_1006_drop_triggers_6stage_reconnect():
    """Abrupt TCP drop (1006) triggers auto-reconnection polling /api/kernels and resumes execution."""
    connection_count = 0

    async def handler(conn):
        nonlocal connection_count
        connection_count += 1
        current = connection_count
        if current == 1:
            # Abruptly drop TCP transport without close frame (code 1006)
            conn.transport.abort()
            return
        async for raw in conn:
            req = json.loads(raw)
            msg_id = req["header"]["msg_id"]
            await conn.send(_reply_frame("stream", msg_id, "iopub", {"name": "stdout", "text": "recovered!\n"}))
            await conn.send(_reply_frame("execute_reply", msg_id, "shell", {"status": "ok", "execution_count": 1}))
            await conn.send(_reply_frame("status", msg_id, "iopub", {"execution_state": "idle"}))

    poll_calls = 0

    async def rest_handler(request: httpx.Request) -> httpx.Response:
        nonlocal poll_calls
        if request.url.path.endswith("/api/kernels"):
            poll_calls += 1
            state = "starting" if poll_calls == 1 else "idle"
            return httpx.Response(200, json=[{"id": "kern-1006", "execution_state": state}])
        return httpx.Response(200)

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = KernelWebSocketClient(
            proxy_url=f"http://127.0.0.1:{port}",
            proxy_token="tok-rec",
            kernel_id="kern-1006",
        )
        orig_async_client = httpx.AsyncClient
        with patch(
            "httpx.AsyncClient",
            side_effect=lambda *a, **kw: orig_async_client(
                transport=httpx.MockTransport(rest_handler)
            ),
        ):
            await client.connect()
            try:
                # Wait briefly for drop + background auto-reconnect
                for _ in range(40):
                    if connection_count >= 2 and client.is_connected:
                        break
                    await asyncio.sleep(0.05)
                assert connection_count >= 2
                assert poll_calls >= 2
                blocks = await client.execute("print('after reconnect')")
                assert any(isinstance(b, TextContent) and "recovered!" in b.text for b in blocks)
            finally:
                await client.disconnect()


@pytest.mark.asyncio
async def test_live_ws_interrupt_with_escalation_tier1_and_tier2():
    """Verify interrupt_with_escalation returns Tier 1 on idle and escalates to Tier 2 on hang."""

    async def handler(conn):
        async for raw in conn:
            req = json.loads(raw)
            if req["header"]["msg_type"] == "execute_request":
                msg_id = req["header"]["msg_id"]
                await conn.send(_reply_frame("status", msg_id, "iopub", {"execution_state": "busy"}))

    async def rest_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/interrupt"):
            return httpx.Response(204)
        if request.url.path.endswith("/restart"):
            return httpx.Response(200, json={"id": "kern-esc", "execution_state": "idle"})
        if request.url.path.endswith("/api/kernels"):
            return httpx.Response(200, json=[{"id": "kern-esc", "execution_state": "idle"}])
        return httpx.Response(200)

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = KernelWebSocketClient(
            proxy_url=f"http://127.0.0.1:{port}",
            proxy_token="tok-esc",
            kernel_id="kern-esc",
        )
        await client.connect()
        try:
            # Tier 1 when already idle
            with patch(
                "httpx.AsyncClient",
                return_value=httpx.AsyncClient(transport=httpx.MockTransport(rest_handler)),
            ):
                t1 = await client.interrupt_with_escalation(timeout=0.2)
            assert t1.state_reset is False
            assert t1.kernel_restarted is False

            # Simulate busy state -> kernel unresponsive -> Tier 2 restart
            client._kernel_state = "busy"
            client._idle_event.clear()
            with patch(
                "httpx.AsyncClient",
                return_value=httpx.AsyncClient(transport=httpx.MockTransport(rest_handler)),
            ):
                t2 = await client.interrupt_with_escalation(timeout=0.05)
            assert t2.state_reset is True
            assert t2.kernel_restarted is True
            assert client.is_connected
        finally:
            await client.disconnect()
