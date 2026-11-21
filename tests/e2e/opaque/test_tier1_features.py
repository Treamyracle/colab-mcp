"""
Tier 1: Opaque-Box Feature Coverage Tests (Happy Path).

Covers:
- T0: Local Colab Twin Self-Checks (3 tests)
- FA1: Authentication & Credential Management (6 tests)
- FA2: Runtime Provisioning & Lifecycle (6 tests)
- FA3: Kernel Bridge, Code Execution & Output Streaming (6 tests)
- FA4: Execution Interrupt, Restart & Reconnection (5 tests)
- FA5: Notebook Validation & Execution Engine (5 tests)
- FA6: MCP Server Transports, Security & Typer CLI (5 tests)

ZERO imports from `colab_mcp.kernel` or `colab_mcp.notebook`.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Dict

import httpx
import pytest
import websockets

from colab_mcp.cli import app as cli_app

from .colab_twin import ColabTwin
from .helpers import (
    extract_blocks_text,
    extract_image_blocks,
    get_registered_cli_commands,
    make_code_cell,
    make_markdown_cell,
    make_notebook,
    make_raw_cell,
    parse_tool_blocks,
    parse_tool_json,
    require_cli_command,
    require_tools,
    write_notebook,
)

ALL_12_TOOLS = [
    "colab_login",
    "colab_get_auth_status",
    "colab_logout",
    "colab_list_runtime_types",
    "colab_connect_runtime",
    "colab_get_runtime_status",
    "colab_disconnect_runtime",
    "colab_execute_code",
    "colab_execute_notebook",
    "colab_get_execution_status",
    "colab_get_cell_output",
    "colab_interrupt_execution",
]


async def _authenticate_via_mcp(
    session: Any,
    twin: ColabTwin,
    auth_user_index: int = 0,
    tier: str = "FREE",
    ccu: float = 0.0,
) -> None:
    twin.state.user_tiers[auth_user_index] = tier
    twin.state.ccu_balances[auth_user_index] = ccu
    code = twin.issue_auth_code(f"twin-code-u{auth_user_index}-{uuid.uuid4().hex[:6]}", auth_user_index=auth_user_index)
    res = await session.call_tool(
        "colab_login",
        {"method": "oauth_remote", "auth_user_index": auth_user_index, "auth_code": code},
    )
    data = parse_tool_json(res)
    assert data["status"] == "authenticated"


# ============================================================================
# T0: Local Colab Twin Self-Checks (3 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_t1_twin_http_tfe_two_step_assign_and_unassign(colab_twin: ColabTwin) -> None:
    """T0.1: Twin performs 2-step XSRF /assign, lists /assignments, and /unassign."""
    async with httpx.AsyncClient() as client:
        tok_resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={"grant_type": "authorization_code", "code": colab_twin.issue_auth_code("t0-code")},
        )
        assert tok_resp.status_code == 200
        access_token = tok_resp.json()["access_token"]
        headers = {
            "Authorization": f"Bearer {access_token}",
            "X-Goog-AuthUser": "0",
            "X-Colab-Client-Agent": "colab-mcp",
        }
        nbh = "a" * 44
        step1 = await client.get(
            f"https://colab.research.google.com/tun/m/assign?authuser=0&nbh={nbh}&variant=GPU&accelerator=T4",
            headers=headers,
        )
        assert step1.status_code == 200
        xsrf = step1.json()["token"]
        step2 = await client.post(
            f"https://colab.research.google.com/tun/m/assign?authuser=0&nbh={nbh}&variant=GPU&accelerator=T4",
            headers={**headers, "X-Goog-Colab-Token": xsrf},
        )
        assert step2.status_code == 200
        assigned = step2.json()
        ep = assigned["endpoint"]
        assert assigned["runtimeProxyInfo"]["url"].startswith("https://colab.research.google.com/tun/m/")
        assert assigned["runtimeProxyInfo"]["tokenExpiresInSeconds"] == 3600

        listed = await client.get("https://colab.research.google.com/tun/m/assignments?authuser=0", headers=headers)
        assert listed.status_code == 200
        assert any(item["endpoint"] == ep for item in listed.json())

        u1 = await client.get(f"https://colab.research.google.com/tun/m/unassign/{ep}?authuser=0", headers=headers)
        assert u1.status_code == 200
        u2 = await client.post(
            f"https://colab.research.google.com/tun/m/unassign/{ep}?authuser=0",
            headers={**headers, "X-Goog-Colab-Token": u1.json()["token"]},
        )
        assert u2.status_code == 200
        assert ep in colab_twin.state.unassigned_endpoints


@pytest.mark.asyncio
async def test_t1_twin_oauth_token_and_userinfo(colab_twin: ColabTwin) -> None:
    """T0.2: Twin exchanges OAuth auth code, refreshes token, and serves /userinfo."""
    colab_twin.issue_auth_code("t0-oauth-code", auth_user_index=1, email="researcher1@example.com")
    async with httpx.AsyncClient() as client:
        ex = await client.post(
            "https://oauth2.googleapis.com/token",
            data={"grant_type": "authorization_code", "code": "t0-oauth-code"},
        )
        assert ex.status_code == 200
        tokens = ex.json()
        assert tokens["access_token"].startswith("ya29.")
        assert tokens["refresh_token"].startswith("1//0")

        ui = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert ui.status_code == 200
        assert ui.json()["email"] == "researcher1@example.com"

        ref = await client.post(
            "https://oauth2.googleapis.com/token",
            data={"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"]},
        )
        assert ref.status_code == 200
        assert ref.json()["access_token"].startswith("ya29.")


@pytest.mark.asyncio
async def test_t1_twin_ws_jupyter_v53_stateful_exec_and_restart(colab_twin: ColabTwin) -> None:
    """T0.3: Twin real WS server executes stateful Python v5.3 frames & wipes state on /restart."""
    async with httpx.AsyncClient() as client:
        tok = (
            await client.post(
                "https://oauth2.googleapis.com/token",
                data={"grant_type": "authorization_code", "code": colab_twin.issue_auth_code("t0-ws")},
            )
        ).json()["access_token"]
        headers = {"Authorization": f"Bearer {tok}", "X-Goog-AuthUser": "0"}
        nbh = "b" * 44
        s1 = (await client.get(f"https://colab.research.google.com/tun/m/assign?authuser=0&nbh={nbh}&variant=DEFAULT&accelerator=NONE", headers=headers)).json()
        assign = (await client.post(
            f"https://colab.research.google.com/tun/m/assign?authuser=0&nbh={nbh}&variant=DEFAULT&accelerator=NONE",
            headers={**headers, "X-Goog-Colab-Token": s1["token"]},
        )).json()
        proxy_url = assign["runtimeProxyInfo"]["url"]
        proxy_tok = assign["runtimeProxyInfo"]["token"]
        kernels = (await client.get(f"{proxy_url}/api/kernels", headers={"X-Colab-Runtime-Proxy-Token": proxy_tok})).json()
        kid = kernels[0]["id"]

        ws_uri = proxy_url.replace("https://", "wss://") + f"/api/kernels/{kid}/channels"
        ws_headers = {"X-Colab-Runtime-Proxy-Token": proxy_tok, "X-Colab-Client-Agent": "colab-mcp"}

        async def _exec(ws_conn: Any, code: str) -> list[dict[str, Any]]:
            msg_id = str(uuid.uuid4())
            await ws_conn.send(json.dumps({
                "header": {"msg_id": msg_id, "username": "colab-mcp", "session": "s1", "msg_type": "execute_request", "version": "5.3"},
                "parent_header": {},
                "metadata": {},
                "content": {"code": code, "silent": False, "store_history": True, "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
                "channel": "shell",
                "buffers": [],
            }))
            frames = []
            got_idle = False
            got_reply = False
            while not (got_idle and got_reply):
                frame = json.loads(await asyncio.wait_for(ws_conn.recv(), timeout=5.0))
                if frame.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                frames.append(frame)
                mtype = frame["header"]["msg_type"]
                if mtype == "status" and frame["content"].get("execution_state") == "idle":
                    got_idle = True
                if mtype == "execute_reply":
                    got_reply = True
            return frames

        async with websockets.connect(ws_uri, subprotocols=None, additional_headers=ws_headers) as ws:
            f1 = await _exec(ws, "twin_var = 99\nprint('ok-twin')")
            assert any(f["header"]["msg_type"] == "stream" and "ok-twin" in f["content"]["text"] for f in f1)
            f2 = await _exec(ws, "twin_var + 1")
            assert any(f["header"]["msg_type"] == "execute_result" and f["content"]["data"]["text/plain"] == "100" for f in f2)

        restart_resp = await client.post(
            f"{proxy_url}/api/kernels/{kid}/restart",
            headers={"X-Colab-Runtime-Proxy-Token": proxy_tok},
        )
        assert restart_resp.status_code == 200

        async with websockets.connect(ws_uri, subprotocols=None, additional_headers=ws_headers) as ws:
            f3 = await _exec(ws, "twin_var")
            assert any(f["header"]["msg_type"] == "error" and f["content"]["ename"] == "NameError" for f in f3)


# ============================================================================
# FA1: Authentication & Credential Management (6 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_t1_fa1_oauth_remote_step1_consent_url(mcp_session: Any) -> None:
    """FA1.1: colab_login(oauth_remote) without auth_code returns authorization_required & Google URL."""
    async with mcp_session("memory") as session:
        res = await session.call_tool("colab_login", {"method": "oauth_remote", "auth_user_index": 0})
        data = parse_tool_json(res)
        assert data["status"] == "authorization_required"
        assert data["auth_method"] == "oauth_remote"
        assert data["auth_url"].startswith("https://accounts.google.com/o/oauth2/")
        assert "prompt=consent" in data["auth_url"]
        assert "token_usage=remote" in data["auth_url"]


@pytest.mark.asyncio
async def test_t1_fa1_oauth_remote_step2_code_exchange(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA1.2: colab_login(oauth_remote, auth_code=...) exchanges code via twin and authenticates."""
    code = colab_twin.issue_auth_code("fa1-remote-code", auth_user_index=0, email="ml-dev@example.com")
    async with mcp_session("memory") as session:
        step1 = parse_tool_json(await session.call_tool("colab_login", {"method": "oauth_remote"}))
        assert step1["status"] == "authorization_required"

        step2 = parse_tool_json(
            await session.call_tool("colab_login", {"method": "oauth_remote", "auth_code": code})
        )
        assert step2["status"] == "authenticated"
        assert step2["user_email"] == "ml-dev@example.com"
        assert step2["has_refresh_token"] is True

        status = parse_tool_json(await session.call_tool("colab_get_auth_status", {"auth_user_index": 0}))
        assert status["is_authenticated"] is True
        assert status["token_valid"] is True
        assert status["user_email"] == "ml-dev@example.com"


@pytest.mark.asyncio
async def test_t1_fa1_cookie_import_and_auth_status(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA1.3: colab_login(cookie_import) stores cookies and colab_get_auth_status reports authenticated."""
    async with mcp_session("memory") as session:
        res = await session.call_tool(
            "colab_login",
            {
                "method": "cookie_import",
                "cookies": {
                    "SAPISID": "twin_sapisid_abc",
                    "SID": "twin_sid_xyz",
                    "OSID": "twin_osid_123",
                },
            },
        )
        data = parse_tool_json(res)
        assert data["status"] == "authenticated"
        assert data["auth_method"] == "cookie_import"

        status = parse_tool_json(await session.call_tool("colab_get_auth_status", {}))
        assert status["is_authenticated"] is True
        assert status["token_valid"] is True
        assert status["account_tier"] == "FREE"


@pytest.mark.asyncio
async def test_t1_fa1_oauth_adc_ingestion(
    mcp_session: Any,
    colab_twin: ColabTwin,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FA1.4: colab_login(oauth_adc) discovers GOOGLE_APPLICATION_CREDENTIALS and authenticates via twin."""
    adc_path = tmp_path / "adc.json"
    adc_path.write_text(
        json.dumps(
            {
                "type": "authorized_user",
                "client_id": "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com",
                "client_secret": "d-FL95Q19q7MQmFpd7hHD0Ty",
                "refresh_token": "1//0twin_adc_refresh_token",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(adc_path))
    async with mcp_session("memory") as session:
        data = parse_tool_json(await session.call_tool("colab_login", {"method": "oauth_adc"}))
        assert data["status"] == "authenticated"
        assert data["auth_method"] == "oauth_adc"
        assert data["user_email"] == "twin-user0@example.com"


@pytest.mark.asyncio
async def test_t1_fa1_get_auth_status_proactive_force_refresh(
    mcp_session: Any, colab_twin: ColabTwin
) -> None:
    """FA1.5: colab_get_auth_status(force_refresh=True) refreshes token via OAuth /token."""
    async with mcp_session("memory") as session:
        await _authenticate_via_mcp(session, colab_twin, auth_user_index=0, tier="PRO", ccu=42.5)
        before_count = sum(1 for r in colab_twin.state.http_requests if r["path"] == "/token")

        status = parse_tool_json(
            await session.call_tool("colab_get_auth_status", {"auth_user_index": 0, "force_refresh": True})
        )
        after_count = sum(1 for r in colab_twin.state.http_requests if r["path"] == "/token")
        assert after_count > before_count
        assert status["is_authenticated"] is True
        assert status["token_valid"] is True
        assert status["account_tier"] == "PRO"
        assert status["compute_units_remaining"] == 42.5


@pytest.mark.asyncio
async def test_t1_fa1_logout_purges_keystore(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA1.6: colab_logout purges credentials and flips colab_get_auth_status to unauthenticated."""
    async with mcp_session("memory") as session:
        await _authenticate_via_mcp(session, colab_twin, auth_user_index=0)
        logout_data = parse_tool_json(
            await session.call_tool("colab_logout", {"auth_user_index": 0, "terminate_active_runtimes": True})
        )
        assert logout_data["status"] == "logged_out"
        assert logout_data["keystore_purged"] is True

        status = parse_tool_json(await session.call_tool("colab_get_auth_status", {"auth_user_index": 0}))
        assert status["is_authenticated"] is False
        assert status["token_valid"] is False


# ============================================================================
# FA2: Runtime Provisioning & Lifecycle (6 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_t1_fa2_list_runtime_types_reflects_pro_tier(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA2.1: colab_list_runtime_types returns CPU/GPU/TPU catalog and twin account tier."""
    async with mcp_session("memory") as session:
        await _authenticate_via_mcp(session, colab_twin, auth_user_index=0, tier="PRO", ccu=80.0)
        data = parse_tool_json(await session.call_tool("colab_list_runtime_types", {"auth_user_index": 0}))
        assert data["account_tier"] == "PRO"
        accs = {item["accelerator"]: item for item in data["available_accelerators"]}
        assert {"NONE", "T4", "L4", "A100", "H100", "V5E1", "V6E1"}.issubset(accs.keys())
        assert accs["T4"]["ccu_cost_per_hour"] == 1.96
        assert accs["T4"]["available"] is True


@pytest.mark.asyncio
async def test_t1_fa2_connect_runtime_cpu_two_step_assign(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA2.2: colab_connect_runtime(NONE) completes 2-step /assign and reports connected + kernel_id."""
    async with mcp_session("memory") as session:
        await _authenticate_via_mcp(session, colab_twin, auth_user_index=0)
        data = parse_tool_json(
            await session.call_tool("colab_connect_runtime", {"accelerator": "NONE", "high_ram": False})
        )
        assert data["status"] == "connected"
        assert data["accelerator"] == "NONE"
        assert data["variant"] == "DEFAULT"
        assert data["high_ram"] is False
        assert data["kernel_id"] is not None
        assert data["heartbeat_interval_seconds"] == 35
        assign_methods = [r["method"] for r in colab_twin.state.http_requests if r["path"] == "/tun/m/assign"]
        assert "GET" in assign_methods and "POST" in assign_methods


@pytest.mark.asyncio
async def test_t1_fa2_connect_runtime_t4_gpu(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA2.3: colab_connect_runtime(T4) provisions GPU runtime with VRAM specs."""
    async with mcp_session("memory") as session:
        await _authenticate_via_mcp(session, colab_twin, auth_user_index=0, tier="PRO", ccu=50.0)
        data = parse_tool_json(
            await session.call_tool("colab_connect_runtime", {"accelerator": "T4", "high_ram": True})
        )
        assert data["status"] == "connected"
        assert data["accelerator"] == "T4"
        assert data["variant"] == "GPU"
        assert data["high_ram"] is True
        assert data["resources"]["vram"] == "16 GB GDDR6"


@pytest.mark.asyncio
async def test_t1_fa2_connect_runtime_reuse_existing(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA2.4: colab_connect_runtime(reuse_existing=True) re-attaches via /assignments without new /assign."""
    async with mcp_session("memory") as session:
        await _authenticate_via_mcp(session, colab_twin, auth_user_index=0)
        first = parse_tool_json(
            await session.call_tool("colab_connect_runtime", {"accelerator": "NONE", "reuse_existing": False})
        )
        assign_count_before = sum(1 for r in colab_twin.state.http_requests if r["path"] == "/tun/m/assign")

        second = parse_tool_json(
            await session.call_tool("colab_connect_runtime", {"accelerator": "NONE", "reuse_existing": True})
        )
        assign_count_after = sum(1 for r in colab_twin.state.http_requests if r["path"] == "/tun/m/assign")
        assert second["runtime_id"] == first["runtime_id"]
        assert second["reused_existing"] is True
        assert assign_count_after == assign_count_before


@pytest.mark.asyncio
async def test_t1_fa2_get_runtime_status_active_metrics(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA2.5: colab_get_runtime_status reports active runtime metrics, kernel state, and keepalive health."""
    async with mcp_session("memory") as session:
        await _authenticate_via_mcp(session, colab_twin, auth_user_index=0)
        conn = parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "T4"}))
        await asyncio.sleep(0.05)

        status = parse_tool_json(
            await session.call_tool("colab_get_runtime_status", {"runtime_id": conn["runtime_id"]})
        )
        assert status["status"] == "active"
        assert status["runtime_id"] == conn["runtime_id"]
        assert status["kernel_state"] == "idle"
        assert status["uptime_seconds"] >= 0.0
        assert status["idle_seconds"] >= 0.0
        assert status["keep_alive_healthy"] is True
        assert status["metrics"]["accelerator"] == "T4"


@pytest.mark.asyncio
async def test_t1_fa2_disconnect_runtime_two_step_unassign(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA2.6: colab_disconnect_runtime releases VM via 2-step /unassign and reports terminated."""
    async with mcp_session("memory") as session:
        await _authenticate_via_mcp(session, colab_twin, auth_user_index=0)
        conn = parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))
        ep = conn["runtime_id"]

        disc = parse_tool_json(await session.call_tool("colab_disconnect_runtime", {"runtime_id": ep}))
        assert disc["status"] == "terminated"
        assert disc["compute_resources_released"] is True
        assert disc["already_terminated"] is False
        assert ep in colab_twin.state.unassigned_endpoints

        status = parse_tool_json(await session.call_tool("colab_get_runtime_status", {"runtime_id": ep}))
        assert status["status"] == "disconnected"


# ============================================================================
# FA3: Kernel Bridge, Code Execution & Output Streaming (6 tests — auto-activates on M3)
# ============================================================================


@pytest.mark.asyncio
async def test_t1_fa3_execute_code_sync_stdout_and_expr(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA3.1: colab_execute_code sync executes Python and returns stdout + expression result TextContent."""
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_code"])
        await _authenticate_via_mcp(session, colab_twin)
        conn = parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))

        res = await session.call_tool(
            "colab_execute_code",
            {"code": "print('hello-colab-twin')\n21 * 2", "runtime_id": conn["runtime_id"]},
        )
        blocks = parse_tool_blocks(res)
        text = extract_blocks_text(blocks)
        assert "hello-colab-twin" in text
        assert "42" in text


@pytest.mark.asyncio
async def test_t1_fa3_execute_code_chronological_multimodal_output(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA3.2: colab_execute_code preserves chronological [TextContent, ImageContent, TextContent] order."""
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_code"])
        await _authenticate_via_mcp(session, colab_twin)
        conn = parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "T4"}))

        code = (
            "import matplotlib.pyplot as plt\n"
            "print('BEFORE_PLOT_LOG')\n"
            "plt.plot([1, 2, 3])\n"
            "plt.show()\n"
            "print('AFTER_PLOT_LOG')\n"
        )
        res = await session.call_tool("colab_execute_code", {"code": code, "runtime_id": conn["runtime_id"]})
        blocks = parse_tool_blocks(res)
        types_seq = [getattr(b, "type", None) for b in blocks]
        assert "image" in types_seq
        first_img_idx = types_seq.index("image")
        before_text = extract_blocks_text(blocks[:first_img_idx])
        after_text = extract_blocks_text(blocks[first_img_idx + 1 :])
        assert "BEFORE_PLOT_LOG" in before_text
        assert "AFTER_PLOT_LOG" in after_text
        imgs = extract_image_blocks(blocks)
        assert imgs[0].mime_type == "image/png"
        assert len(imgs[0].data) > 20


@pytest.mark.asyncio
async def test_t1_fa3_execute_code_state_persistence(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA3.3: Kernel variables and imports persist across sequential colab_execute_code calls."""
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_code"])
        await _authenticate_via_mcp(session, colab_twin)
        conn = parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))

        parse_tool_blocks(
            await session.call_tool(
                "colab_execute_code",
                {"code": "acc_list = [10, 20, 30]\nimport math", "runtime_id": conn["runtime_id"]},
            )
        )
        res2 = await session.call_tool(
            "colab_execute_code",
            {"code": "print(f'sum={sum(acc_list)} sqrt16={int(math.sqrt(16))}')", "runtime_id": conn["runtime_id"]},
        )
        text = extract_blocks_text(parse_tool_blocks(res2))
        assert "sum=60 sqrt16=4" in text


@pytest.mark.asyncio
async def test_t1_fa3_execute_code_stderr_prefix(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA3.4: Kernel stderr stream is formatted with '[stderr] ' prefix."""
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_code"])
        await _authenticate_via_mcp(session, colab_twin)
        conn = parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))

        res = await session.call_tool(
            "colab_execute_code",
            {"code": "import sys\nsys.stderr.write('warning-msg\\n')", "runtime_id": conn["runtime_id"]},
        )
        text = extract_blocks_text(parse_tool_blocks(res))
        assert "[stderr]" in text
        assert "warning-msg" in text


@pytest.mark.asyncio
async def test_t1_fa3_async_execution_and_status_polling(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA3.5: wait_for_completion=False returns execution_id and colab_get_execution_status tracks completion."""
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_code", "colab_get_execution_status"])
        await _authenticate_via_mcp(session, colab_twin)
        conn = parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))

        dispatch = parse_tool_json(
            await session.call_tool(
                "colab_execute_code",
                {
                    "code": "import time\nprint('async-step-1')\ntime.sleep(0.15)\nprint('async-step-2')",
                    "runtime_id": conn["runtime_id"],
                    "wait_for_completion": False,
                },
            )
        )
        exec_id = dispatch["execution_id"]
        assert dispatch["status"] in ("queued", "running", "completed", "ok")

        status_data: Dict[str, Any] = {}
        for _ in range(40):
            status_data = parse_tool_json(
                await session.call_tool("colab_get_execution_status", {"execution_id": exec_id})
            )
            if status_data["status"] in ("completed", "ok"):
                break
            await asyncio.sleep(0.05)

        assert status_data["execution_id"] == exec_id
        assert status_data["status"] in ("completed", "ok")
        assert status_data["has_error"] is False
        assert "async-step-2" in status_data.get("partial_stdout_tail", "")


@pytest.mark.asyncio
async def test_t1_fa3_get_cell_output_with_and_without_images(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA3.6: colab_get_cell_output returns buffered chronological blocks and respects include_images."""
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_code", "colab_get_execution_status", "colab_get_cell_output"])
        await _authenticate_via_mcp(session, colab_twin)
        conn = parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))

        dispatch = parse_tool_json(
            await session.call_tool(
                "colab_execute_code",
                {
                    "code": "import matplotlib.pyplot as plt\nprint('chart-title')\nplt.plot([1,2])\nplt.show()",
                    "runtime_id": conn["runtime_id"],
                    "wait_for_completion": False,
                },
            )
        )
        exec_id = dispatch["execution_id"]
        for _ in range(40):
            st = parse_tool_json(await session.call_tool("colab_get_execution_status", {"execution_id": exec_id}))
            if st["status"] in ("completed", "ok"):
                break
            await asyncio.sleep(0.05)

        with_imgs = parse_tool_blocks(
            await session.call_tool("colab_get_cell_output", {"execution_id": exec_id, "include_images": True})
        )
        assert "chart-title" in extract_blocks_text(with_imgs)
        assert len(extract_image_blocks(with_imgs)) == 1

        no_imgs = parse_tool_blocks(
            await session.call_tool("colab_get_cell_output", {"execution_id": exec_id, "include_images": False})
        )
        assert "chart-title" in extract_blocks_text(no_imgs)
        assert len(extract_image_blocks(no_imgs)) == 0


# ============================================================================
# FA4: Execution Interrupt, Restart & Reconnection (5 tests — auto-activates on M3)
# ============================================================================


@pytest.mark.asyncio
async def test_t1_fa4_tier1_sigint_interrupt_preserves_state(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA4.1: Tier-1 SIGINT halts slow loop with state_reset=False and kernel_restarted=False."""
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_code", "colab_interrupt_execution"])
        await _authenticate_via_mcp(session, colab_twin)
        conn = parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))
        rid = conn["runtime_id"]

        parse_tool_blocks(
            await session.call_tool("colab_execute_code", {"code": "keep_me = 777", "runtime_id": rid})
        )
        loop_task = asyncio.create_task(
            session.call_tool(
                "colab_execute_code",
                {"code": "import time\nwhile True:\n    time.sleep(0.05)", "runtime_id": rid, "timeout_seconds": 60},
            )
        )
        await asyncio.sleep(0.15)
        intr = parse_tool_json(await session.call_tool("colab_interrupt_execution", {"runtime_id": rid}))
        assert intr["status"] == "interrupted"
        assert intr["state_reset"] is False
        assert intr["kernel_restarted"] is False
        await asyncio.gather(loop_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_t1_fa4_post_sigint_variable_continuity(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA4.2: Kernel remains usable after SIGINT and preserves variables created before interrupt."""
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_code", "colab_interrupt_execution"])
        await _authenticate_via_mcp(session, colab_twin)
        conn = parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))
        rid = conn["runtime_id"]

        parse_tool_blocks(
            await session.call_tool("colab_execute_code", {"code": "preserved_val = 'alive'", "runtime_id": rid})
        )
        loop_task = asyncio.create_task(
            session.call_tool(
                "colab_execute_code",
                {"code": "import time\nwhile True:\n    time.sleep(0.05)", "runtime_id": rid},
            )
        )
        await asyncio.sleep(0.15)
        parse_tool_json(await session.call_tool("colab_interrupt_execution", {"runtime_id": rid}))
        await asyncio.gather(loop_task, return_exceptions=True)

        check = await session.call_tool(
            "colab_execute_code",
            {"code": "print(f'status={preserved_val}')", "runtime_id": rid},
        )
        assert "status=alive" in extract_blocks_text(parse_tool_blocks(check))


@pytest.mark.asyncio
async def test_t1_fa4_tier2_escalated_restart_on_hung_sigint(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA4.3: Unresponsive kernel ignoring SIGINT escalates after 5.0s to Tier-2 /restart."""
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_code", "colab_interrupt_execution"])
        await _authenticate_via_mcp(session, colab_twin)
        conn = parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))
        rid = conn["runtime_id"]

        parse_tool_blocks(await session.call_tool("colab_execute_code", {"code": "doomed = 123", "runtime_id": rid}))
        colab_twin.faults.hang_on_sigint = True
        hung_task = asyncio.create_task(
            session.call_tool(
                "colab_execute_code",
                {"code": "import time\nwhile True:\n    time.sleep(0.05)", "runtime_id": rid},
            )
        )
        await asyncio.sleep(0.15)
        intr = parse_tool_json(await session.call_tool("colab_interrupt_execution", {"runtime_id": rid}))
        assert intr["status"] == "restarted"
        assert intr["state_reset"] is True
        assert intr["kernel_restarted"] is True
        assert intr["warning"] is not None
        assert len(colab_twin.state.restarted_kernels) >= 1
        colab_twin.faults.hang_on_sigint = False
        await asyncio.gather(hung_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_t1_fa4_post_restart_ws_reconnect_clean_namespace(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA4.4: After Tier-2 /restart, WS reconnects automatically with wiped namespace."""
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_code", "colab_interrupt_execution"])
        await _authenticate_via_mcp(session, colab_twin)
        conn = parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))
        rid = conn["runtime_id"]

        parse_tool_blocks(await session.call_tool("colab_execute_code", {"code": "wiped_var = 42", "runtime_id": rid}))
        colab_twin.faults.hang_on_sigint = True
        hung_task = asyncio.create_task(
            session.call_tool(
                "colab_execute_code",
                {"code": "import time\nwhile True:\n    time.sleep(0.05)", "runtime_id": rid},
            )
        )
        await asyncio.sleep(0.15)
        intr = parse_tool_json(await session.call_tool("colab_interrupt_execution", {"runtime_id": rid}))
        assert intr["kernel_restarted"] is True
        colab_twin.faults.hang_on_sigint = False
        await asyncio.gather(hung_task, return_exceptions=True)

        check_err = await session.call_tool("colab_execute_code", {"code": "print(wiped_var)", "runtime_id": rid})
        assert check_err.is_error is True
        assert "NameError" in extract_blocks_text(check_err)

        ok = await session.call_tool("colab_execute_code", {"code": "print('reconnected-ok')", "runtime_id": rid})
        assert "reconnected-ok" in extract_blocks_text(parse_tool_blocks(ok))


@pytest.mark.asyncio
async def test_t1_fa4_auto_reconnect_after_ws_1006(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA4.5: Client auto-reconnects seamlessly after genuine WS 1006 transport abort."""
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_code"])
        await _authenticate_via_mcp(session, colab_twin)
        conn = parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))
        rid = conn["runtime_id"]

        parse_tool_blocks(await session.call_tool("colab_execute_code", {"code": "x_persist = 55", "runtime_id": rid}))
        dropped = await colab_twin.drop_active_ws_1006()
        assert dropped >= 1
        await asyncio.sleep(0.05)

        res = await session.call_tool("colab_execute_code", {"code": "print(f'after-1006={x_persist}')", "runtime_id": rid})
        assert "after-1006=55" in extract_blocks_text(parse_tool_blocks(res))


# ============================================================================
# FA5: Notebook Validation & Execution Engine (5 tests — auto-activates on M4)
# ============================================================================


@pytest.mark.asyncio
async def test_t1_fa5_execute_notebook_from_path_skips_markdown_and_raw(
    mcp_session: Any,
    colab_twin: ColabTwin,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FA5.1: colab_execute_notebook(notebook_path=...) executes code cells sequentially and skips markdown/raw."""
    monkeypatch.chdir(tmp_path)
    nb_path = write_notebook(
        tmp_path / "sequential.ipynb",
        [
            make_markdown_cell("# Header"),
            make_code_cell("a = 10\nprint('cell1')"),
            make_raw_cell("raw block"),
            make_code_cell("print(f'cell3={a * 2}')"),
        ],
    )
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_notebook"])
        await _authenticate_via_mcp(session, colab_twin)
        parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))

        data = parse_tool_json(
            await session.call_tool(
                "colab_execute_notebook",
                {"notebook_path": str(nb_path), "reset_kernel_before_run": True},
            )
        )
        assert data["total_cells"] == 4
        assert data["executed_cells"] == 2
        assert data["successful_cells"] == 2
        assert data["failed_cells"] == 0
        assert data["skipped_cells"] == 2


@pytest.mark.asyncio
async def test_t1_fa5_execute_notebook_from_content(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA5.2: colab_execute_notebook(notebook_content=...) executes inline nbformat v4 JSON."""
    nb = make_notebook([make_code_cell("val = 7\nprint('inline-nb-ok')"), make_code_cell("val * 6")])
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_notebook"])
        await _authenticate_via_mcp(session, colab_twin)
        parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))

        data = parse_tool_json(
            await session.call_tool(
                "colab_execute_notebook",
                {"notebook_content": json.dumps(nb), "reset_kernel_before_run": False},
            )
        )
        assert data["total_cells"] == 2
        assert data["executed_cells"] == 2
        assert data["successful_cells"] == 2


@pytest.mark.asyncio
async def test_t1_fa5_execute_notebook_selective_cell_indices(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA5.3: 0-based cell_indices executes only targeted code cells and skips markdown cleanly."""
    nb = make_notebook([
        make_code_cell("step0 = 's0'"),
        make_markdown_cell("## Doc"),
        make_code_cell("step2 = 'should-not-run'"),
        make_code_cell("print(f'got={step0}')"),
    ])
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_notebook"])
        await _authenticate_via_mcp(session, colab_twin)
        parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))

        data = parse_tool_json(
            await session.call_tool(
                "colab_execute_notebook",
                {
                    "notebook_content": json.dumps(nb),
                    "cell_indices": [0, 1, 3],
                    "reset_kernel_before_run": True,
                },
            )
        )
        assert data["total_cells"] == 4
        assert data["executed_cells"] == 2
        assert data["successful_cells"] == 2
        statuses = {s["cell_index"]: s["status"] for s in data["cell_summaries"]}
        assert statuses[0] == "success"
        assert statuses[3] == "success"


@pytest.mark.asyncio
async def test_t1_fa5_reset_kernel_before_run_true_vs_false(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA5.4: reset_kernel_before_run=False preserves session state while True resets kernel."""
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_code", "colab_execute_notebook"])
        await _authenticate_via_mcp(session, colab_twin)
        parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))

        parse_tool_blocks(await session.call_tool("colab_execute_code", {"code": "shared_seed = 33"}))

        nb_preserve = make_notebook([make_code_cell("print(shared_seed + 9)")])
        res_keep = parse_tool_json(
            await session.call_tool(
                "colab_execute_notebook",
                {"notebook_content": json.dumps(nb_preserve), "reset_kernel_before_run": False},
            )
        )
        assert res_keep["kernel_reset_performed"] is False
        assert res_keep["successful_cells"] == 1

        res_reset = parse_tool_json(
            await session.call_tool(
                "colab_execute_notebook",
                {"notebook_content": json.dumps(nb_preserve), "reset_kernel_before_run": True},
            )
        )
        assert res_reset["kernel_reset_performed"] is True
        assert res_reset["failed_cells"] == 1
        assert res_reset["cell_summaries"][0]["error_name"] == "NameError"


@pytest.mark.asyncio
async def test_t1_fa5_atomic_output_save_path_ast_nodes(
    mcp_session: Any,
    colab_twin: ColabTwin,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FA5.5: output_save_path writes valid nbformat v4 .ipynb with stream, display_data, and execute_result."""
    monkeypatch.chdir(tmp_path)
    out_path = tmp_path / "saved_results.ipynb"
    nb = make_notebook([
        make_code_cell("import matplotlib.pyplot as plt\nprint('stream-line')\nplt.plot([1, 2])\nplt.show()\n'final-expr'"),
    ])
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_notebook"])
        await _authenticate_via_mcp(session, colab_twin)
        parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))

        data = parse_tool_json(
            await session.call_tool(
                "colab_execute_notebook",
                {"notebook_content": json.dumps(nb), "output_save_path": str(out_path)},
            )
        )
        assert data["saved_to"] is not None
        assert out_path.exists()
        saved = json.loads(out_path.read_text(encoding="utf-8"))
        assert saved["nbformat"] == 4
        cell0 = saved["cells"][0]
        assert cell0["execution_count"] is not None
        out_types = {o["output_type"] for o in cell0["outputs"]}
        assert "stream" in out_types
        assert "display_data" in out_types
        assert data["cell_summaries"][0]["has_plots"] is True


# ============================================================================
# FA6: MCP Server Transports, Security & Typer CLI (5 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_t1_fa6_list_all_12_tools_with_valid_schemas(mcp_session: Any) -> None:
    """FA6.1: Server exposes all 12 MCP tools with object JSON schemas once M3-M5 complete."""
    async with mcp_session("memory") as session:
        await require_tools(session, ALL_12_TOOLS)
        resp = await session.list_tools()
        by_name = {t.name: t for t in resp.tools}
        assert set(ALL_12_TOOLS).issubset(by_name.keys())
        for name in ALL_12_TOOLS:
            t = by_name[name]
            assert t.description and len(t.description) > 10
            assert t.input_schema.get("type") == "object"


@pytest.mark.asyncio
async def test_t1_fa6_sse_client_executes_tools_end_to_end(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA6.2: Real sse_client over local uvicorn executes MCP tools end-to-end."""
    code = colab_twin.issue_auth_code("sse-code", auth_user_index=0, email="sse@example.com")
    async with mcp_session("sse") as session:
        login_res = parse_tool_json(
            await session.call_tool("colab_login", {"method": "oauth_remote", "auth_code": code})
        )
        assert login_res["status"] == "authenticated"
        assert login_res["user_email"] == "sse@example.com"

        rt_res = parse_tool_json(await session.call_tool("colab_list_runtime_types", {}))
        assert len(rt_res["available_accelerators"]) >= 7


@pytest.mark.asyncio
async def test_t1_fa6_stdio_client_subprocess_clean_jsonrpc(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA6.3: Real stdio_client subprocess (`python -m colab_mcp run --transport stdio`) executes tools cleanly."""
    code = colab_twin.issue_auth_code("stdio-code", auth_user_index=0, email="stdio@example.com")
    async with mcp_session("stdio") as session:
        login_res = parse_tool_json(
            await session.call_tool("colab_login", {"method": "oauth_remote", "auth_code": code})
        )
        assert login_res["status"] == "authenticated"
        status_res = parse_tool_json(await session.call_tool("colab_get_auth_status", {}))
        assert status_res["is_authenticated"] is True


def test_t1_fa6_cli_help_and_version(cli_runner: Any) -> None:
    """FA6.4: Typer CLI --help and version succeed with exit code 0."""
    help_res = cli_runner.invoke(cli_app, ["--help"])
    assert help_res.exit_code == 0
    assert "colab-mcp" in help_res.stdout.lower()

    cmds = get_registered_cli_commands()
    ver_args = ["version"] if len(cmds) > 1 else []
    ver_res = cli_runner.invoke(cli_app, ver_args)
    assert ver_res.exit_code == 0
    assert "version" in ver_res.stdout.lower()


def test_t1_fa6_cli_lifecycle_commands_against_twin(cli_runner: Any, colab_twin: ColabTwin) -> None:
    """FA6.5: Typer CLI login, status, list-runtimes, and logout work offline against ColabTwin."""
    for cmd in ("login", "status", "list-runtimes", "logout"):
        require_cli_command(cmd)

    code = colab_twin.issue_auth_code("cli-code", auth_user_index=0, email="cli@example.com")
    login_help = cli_runner.invoke(cli_app, ["login", "--help"])
    assert login_help.exit_code == 0

    login_res = cli_runner.invoke(
        cli_app,
        ["login", "--method", "oauth_remote", "--auth-code", code],
    )
    assert login_res.exit_code == 0

    status_res = cli_runner.invoke(cli_app, ["status"])
    assert status_res.exit_code == 0

    list_res = cli_runner.invoke(cli_app, ["list-runtimes"])
    assert list_res.exit_code == 0
    assert "T4" in list_res.stdout

    logout_res = cli_runner.invoke(cli_app, ["logout"])
    assert logout_res.exit_code == 0


def test_t1_fa6_cli_stateful_runtime_exec_and_disconnect_against_twin(
    cli_runner: Any,
    colab_twin: ColabTwin,
    tmp_path: Path,
) -> None:
    """FA6.6: Stateful CLI flow: login -> connect-runtime -> execute-code (with plot) -> system-metrics -> disconnect."""
    code = colab_twin.issue_auth_code("cli-full-code", auth_user_index=0, email="clifull@example.com")
    assert cli_runner.invoke(cli_app, ["login", "--method", "oauth_remote", "--auth-code", code]).exit_code == 0

    conn_res = cli_runner.invoke(cli_app, ["connect-runtime", "--accelerator", "T4", "--json"])
    assert conn_res.exit_code == 0
    conn_data = json.loads(conn_res.stdout)
    assert conn_data["status"] == "connected"
    rid = conn_data["runtime_id"]

    plots_dir = tmp_path / "cli_plots"
    exec_res = cli_runner.invoke(
        cli_app,
        [
            "execute-code",
            "import matplotlib.pyplot as plt\nprint('cli-exec-ok')\nplt.plot([1, 2], [3, 4])\nplt.show()",
            "--save-images-dir",
            str(plots_dir),
            "--json",
        ],
    )
    assert exec_res.exit_code == 0
    exec_data = json.loads(exec_res.stdout)
    assert exec_data["has_error"] is False
    assert (plots_dir / "plot_001.png").is_file()

    metrics_res = cli_runner.invoke(cli_app, ["system-metrics", "--json"])
    assert metrics_res.exit_code == 0
    assert json.loads(metrics_res.stdout)["runtime_id"] == rid

    disc_res = cli_runner.invoke(cli_app, ["disconnect", "--json"])
    assert disc_res.exit_code == 0
    assert json.loads(disc_res.stdout)["terminated_count"] >= 1

    assert cli_runner.invoke(cli_app, ["logout"]).exit_code == 0
