"""
Integration tests for interactive Jupyter stdin (input_request/input_reply),
colab_send_input, colab_request Secret resolution, and KernelWebSocketClient
headless bootstrapping over a real loopback WebSocket.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest
import websockets

from colab_mcp.auth.secrets import set_secret
from colab_mcp.kernel.client import KernelWebSocketClient
from colab_mcp.kernel.tools import (
    clear_kernel_clients,
    colab_execute_code,
    colab_get_execution_status,
    colab_send_input,
    get_execution_registry,
    reset_execution_registry,
    reset_kernel_clients,
)
from colab_mcp.notebook.runner import NotebookRunner
from colab_mcp.runtime.models import Assignment
from colab_mcp.runtime.session_manager import get_session_manager, reset_session_manager
import nbformat


def _frame(msg_type: str, parent_msg_id: str, channel: str, content: Dict[str, Any], msg_id: str | None = None) -> str:
    return json.dumps(
        {
            "header": {
                "msg_id": msg_id or f"srv-{parent_msg_id}-{msg_type}",
                "username": "kernel",
                "session": "srv-sess",
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


@pytest.fixture(autouse=True)
async def _isolated_state():
    reset_session_manager()
    reset_execution_registry()
    reset_kernel_clients()
    yield
    await clear_kernel_clients()
    reset_execution_registry()
    reset_session_manager()


@pytest.mark.asyncio
async def test_pre_supplied_stdin_inputs_and_secret_interpolation(
    mock_keystore_dir,
    mock_keyring,
    mock_machine_id,
):
    set_secret("API_KEY", "secret_token_98765")
    received_inputs: List[str] = []

    async def handler(conn):
        async for raw in conn:
            req = json.loads(raw)
            msg_type = req["header"]["msg_type"]
            msg_id = req["header"]["msg_id"]
            if msg_type == "execute_request":
                code = req["content"]["code"]
                await conn.send(_frame("status", msg_id, "iopub", {"execution_state": "busy"}))
                if "_colab_mcp_install_shims" in code:
                    await conn.send(_frame("execute_reply", msg_id, "shell", {"status": "ok", "execution_count": 0}))
                    await conn.send(_frame("status", msg_id, "iopub", {"execution_state": "idle"}))
                    continue

                assert req["content"]["allow_stdin"] is True
                await conn.send(
                    _frame("input_request", msg_id, "stdin", {"prompt": "Name: ", "password": False}, msg_id="inp-1")
                )
                reply1 = json.loads(await conn.recv())
                assert reply1["header"]["msg_type"] == "input_reply"
                assert reply1["parent_header"]["msg_id"] == "inp-1"
                received_inputs.append(reply1["content"]["value"])

                await conn.send(
                    _frame("input_request", msg_id, "stdin", {"prompt": "Key: ", "password": True}, msg_id="inp-2")
                )
                reply2 = json.loads(await conn.recv())
                received_inputs.append(reply2["content"]["value"])

                await conn.send(
                    _frame(
                        "stream",
                        msg_id,
                        "iopub",
                        {"name": "stdout", "text": f"Hello {received_inputs[0]}, key={received_inputs[1]}\n"},
                    )
                )
                await conn.send(_frame("execute_reply", msg_id, "shell", {"status": "ok", "execution_count": 1}))
                await conn.send(_frame("status", msg_id, "iopub", {"execution_state": "idle"}))

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = KernelWebSocketClient(
            proxy_url=f"http://127.0.0.1:{port}",
            proxy_token="tok-1",
            kernel_id="kern-1",
            runtime_id="ep-stdin-1",
        )
        blocks = await colab_execute_code(
            code="name = input('Name: '); key = input('Key: '); print(name, key)",
            stdin_inputs=["alice", "$SECRET:API_KEY"],
            _client_getter=AsyncMock(return_value=client),
        )
        await client.disconnect()

    assert received_inputs == ["alice", "secret_token_98765"]
    text = "\n".join(getattr(b, "text", "") for b in blocks)
    assert "Hello alice" in text
    assert "secret_token_98765" not in text
    assert "[REDACTED]" in text


@pytest.mark.asyncio
async def test_sync_input_request_without_stdin_inputs_fails_fast():
    async def handler(conn):
        async for raw in conn:
            req = json.loads(raw)
            msg_type = req["header"]["msg_type"]
            msg_id = req["header"]["msg_id"]
            if msg_type == "execute_request":
                await conn.send(_frame("status", msg_id, "iopub", {"execution_state": "busy"}))
                if "_colab_mcp_install_shims" in req["content"]["code"]:
                    await conn.send(_frame("execute_reply", msg_id, "shell", {"status": "ok", "execution_count": 0}))
                    await conn.send(_frame("status", msg_id, "iopub", {"execution_state": "idle"}))
                    continue
                await conn.send(
                    _frame("input_request", msg_id, "stdin", {"prompt": "Enter choice: ", "password": False}, msg_id="inp-sync")
                )
            elif msg_type == "interrupt_request":
                break

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = KernelWebSocketClient(
            proxy_url=f"http://127.0.0.1:{port}",
            proxy_token="tok-1",
            kernel_id="kern-1",
            runtime_id="ep-sync-fail",
        )
        client.interrupt = AsyncMock()
        blocks = await colab_execute_code(
            code="input('Enter choice: ')",
            timeout_seconds=10,
            _client_getter=AsyncMock(return_value=client),
        )
        await client.disconnect()

    text = "\n".join(getattr(b, "text", "") for b in blocks)
    assert "StdinInputRequiredError" in text
    assert "Enter choice:" in text


@pytest.mark.asyncio
async def test_async_waiting_for_input_and_colab_send_input_tool(
    mock_keystore_dir,
    mock_keyring,
    mock_machine_id,
):
    set_secret("HF_TOKEN", "hf_async_secret_999")
    received_reply: List[str] = []

    async def handler(conn):
        async for raw in conn:
            req = json.loads(raw)
            if req["header"]["msg_type"] != "execute_request":
                continue
            msg_id = req["header"]["msg_id"]
            await conn.send(_frame("status", msg_id, "iopub", {"execution_state": "busy"}))
            if "_colab_mcp_install_shims" in req["content"]["code"]:
                await conn.send(_frame("execute_reply", msg_id, "shell", {"status": "ok", "execution_count": 0}))
                await conn.send(_frame("status", msg_id, "iopub", {"execution_state": "idle"}))
                continue

            await conn.send(
                _frame("input_request", msg_id, "stdin", {"prompt": "Token: ", "password": True}, msg_id="inp-async")
            )
            reply = json.loads(await conn.recv())
            received_reply.append(reply["content"]["value"])
            await conn.send(_frame("stream", msg_id, "iopub", {"name": "stdout", "text": "Logged in!\n"}))
            await conn.send(_frame("execute_reply", msg_id, "shell", {"status": "ok", "execution_count": 1}))
            await conn.send(_frame("status", msg_id, "iopub", {"execution_state": "idle"}))

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = KernelWebSocketClient(
            proxy_url=f"http://127.0.0.1:{port}",
            proxy_token="tok-1",
            kernel_id="kern-1",
            runtime_id="ep-async-stdin",
        )
        dispatch = await colab_execute_code(
            code="import getpass; tok = getpass.getpass('Token: '); print('Logged in!')",
            wait_for_completion=False,
            _client_getter=AsyncMock(return_value=client),
        )
        exec_id = dispatch["execution_id"]

        # Poll until status enters waiting_for_input
        for _ in range(50):
            st = await colab_get_execution_status(exec_id)
            if st["status"] == "waiting_for_input":
                break
            await asyncio.sleep(0.02)

        assert st["status"] == "waiting_for_input"
        assert st["pending_input_prompt"] == "Token: "
        assert st["pending_input_password"] is True

        send_resp = await colab_send_input(exec_id, "$SECRET:HF_TOKEN")
        assert send_resp.status == "input_delivered"

        reg = get_execution_registry()
        await reg[exec_id]["task"]
        final_st = await colab_get_execution_status(exec_id)
        assert final_st["status"] == "completed"
        await client.disconnect()

    assert received_reply == ["hf_async_secret_999"]


@pytest.mark.asyncio
async def test_colab_request_secret_and_modal_fallback(
    mock_keystore_dir,
    mock_keyring,
    mock_machine_id,
):
    set_secret("MY_KEY", "my_secret_value_abc")
    colab_replies: List[Dict[str, Any]] = []

    async def handler(conn):
        async for raw in conn:
            req = json.loads(raw)
            if req["header"]["msg_type"] != "execute_request":
                continue
            msg_id = req["header"]["msg_id"]
            await conn.send(_frame("status", msg_id, "iopub", {"execution_state": "busy"}))
            await conn.send(
                _frame(
                    "colab_request",
                    msg_id,
                    "shell",
                    {"request_type": "GetSecret", "key": "MY_KEY", "colab_msg_id": 1},
                )
            )
            colab_replies.append(json.loads(await conn.recv()))
            await conn.send(
                _frame(
                    "colab_request",
                    msg_id,
                    "shell",
                    {"request_type": "dfs_ephemeral", "colab_msg_id": 2},
                )
            )
            colab_replies.append(json.loads(await conn.recv()))
            await conn.send(_frame("execute_reply", msg_id, "shell", {"status": "ok", "execution_count": 1}))
            await conn.send(_frame("status", msg_id, "iopub", {"execution_state": "idle"}))

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = KernelWebSocketClient(
            proxy_url=f"http://127.0.0.1:{port}",
            proxy_token="tok-1",
            kernel_id="kern-1",
        )
        await client.connect()
        try:
            res = await client.execute_raw("userdata.get('MY_KEY')", timeout_seconds=10)
            assert res.status == "ok"
        finally:
            await client.disconnect()

    assert len(colab_replies) == 2
    assert colab_replies[0]["header"]["msg_type"] == "colab_reply"
    assert colab_replies[0]["content"]["value"] == {"exists": True, "payload": "my_secret_value_abc"}
    assert colab_replies[1]["header"]["msg_type"] == "colab_reply"
    assert "error" in colab_replies[1]["content"]


@pytest.mark.asyncio
async def test_notebook_runner_rebootstraps_after_reset_and_forwards_stdin(
    mock_keystore_dir,
    mock_keyring,
    mock_machine_id,
):
    set_secret("NB_SECRET", "nb_secret_val_42")
    saw_bootstrap: List[str] = []
    received_stdin: List[str] = []

    async def handler(conn):
        async for raw in conn:
            req = json.loads(raw)
            if req["header"]["msg_type"] != "execute_request":
                continue
            msg_id = req["header"]["msg_id"]
            code = req["content"]["code"]
            await conn.send(_frame("status", msg_id, "iopub", {"execution_state": "busy"}))
            if "_colab_mcp_install_shims" in code:
                saw_bootstrap.append(code)
                await conn.send(_frame("execute_reply", msg_id, "shell", {"status": "ok", "execution_count": 0}))
                await conn.send(_frame("status", msg_id, "iopub", {"execution_state": "idle"}))
                continue

            await conn.send(
                _frame("input_request", msg_id, "stdin", {"prompt": "Answer: ", "password": False}, msg_id=f"inp-{msg_id}")
            )
            reply = json.loads(await conn.recv())
            received_stdin.append(reply["content"]["value"])
            await conn.send(_frame("stream", msg_id, "iopub", {"name": "stdout", "text": f"got={reply['content']['value']}\n"}))
            await conn.send(_frame("execute_reply", msg_id, "shell", {"status": "ok", "execution_count": len(received_stdin)}))
            await conn.send(_frame("status", msg_id, "iopub", {"execution_state": "idle"}))

    nb = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("print(input('Answer: '))"),
            nbformat.v4.new_code_cell("print(input('Answer: '))"),
        ]
    )

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = KernelWebSocketClient(
            proxy_url=f"http://127.0.0.1:{port}",
            proxy_token="tok-1",
            kernel_id="kern-1",
            runtime_id="ep-nb-stdin",
        )
        client.restart = AsyncMock()
        runner = NotebookRunner(client=client, runtime_id="ep-nb-stdin")
        summary = await runner.run(
            nb,
            reset_kernel_before_run=True,
            stdin_inputs=["first_answer", "$SECRET:NB_SECRET"],
            env={"NB_CUSTOM_ENV": "yes"},
        )
        await client.disconnect()

    assert summary["successful_cells"] == 2
    assert len(saw_bootstrap) == 1
    assert "NB_SECRET" in saw_bootstrap[0]
    assert "NB_CUSTOM_ENV" in saw_bootstrap[0]
    assert received_stdin == ["first_answer", "nb_secret_val_42"]
    assert "nb_secret_val_42" not in json.dumps(summary)

