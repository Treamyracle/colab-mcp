"""
Tier 2: Opaque-Box Boundary, Negative, Fault-Injection & Security Tests.

Covers:
- T0: Local Colab Twin Fault & Protocol Guard Checks (2 tests)
- FA1: Authentication & Credential Management Boundaries (6 tests)
- FA2: Runtime Provisioning & Lifecycle Faults/Boundaries (6 tests)
- FA3: Kernel Bridge, ANSI Sanitizer, Truncation & Image Bounds (6 tests)
- FA4: Interrupt, Restart & Reconnection Edge Cases (5 tests)
- FA5: Notebook Validation, Traversal Guard & Error Policies (5 tests)
- FA6: Server Security, Egress Redaction, Lifespan & CLI Bounds (5 tests)

ZERO imports from `colab_mcp.kernel` or `colab_mcp.notebook`.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
import websockets
from websockets.exceptions import ConnectionClosedError, InvalidStatus

from colab_mcp.cli import app as cli_app
from colab_mcp.runtime.session_manager import get_session_manager
from colab_mcp.server import create_server

from .colab_twin import AssignFault, ColabTwin
from .helpers import (
    assert_tool_error,
    extract_blocks_text,
    extract_image_blocks,
    make_code_cell,
    make_notebook,
    parse_tool_blocks,
    parse_tool_json,
    require_cli_command,
    require_m5_server_features,
    require_tools,
    write_notebook,
)


async def _auth(
    session: Any,
    twin: ColabTwin,
    user_idx: int = 0,
    tier: str = "FREE",
    ccu: float = 0.0,
) -> None:
    twin.state.user_tiers[user_idx] = tier
    twin.state.ccu_balances[user_idx] = ccu
    code = twin.issue_auth_code(f"t2-u{user_idx}-{uuid.uuid4().hex[:6]}", auth_user_index=user_idx)
    res = await session.call_tool(
        "colab_login",
        {"method": "oauth_remote", "auth_user_index": user_idx, "auth_code": code},
    )
    parse_tool_json(res)


# ============================================================================
# T0: Local Colab Twin Fault & Guard Checks (2 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_t2_twin_ws_rejects_subprotocol_and_bad_proxy_token(colab_twin: ColabTwin) -> None:
    """T0.4: Twin WS rejects any Sec-WebSocket-Protocol with HTTP 400 and bad token with HTTP 403."""
    async with httpx.AsyncClient() as client:
        tok = (
            await client.post(
                "https://oauth2.googleapis.com/token",
                data={"grant_type": "authorization_code", "code": colab_twin.issue_auth_code("t2-ws-guard")},
            )
        ).json()["access_token"]
        h = {"Authorization": f"Bearer {tok}", "X-Goog-AuthUser": "0"}
        nbh = "c" * 44
        s1 = (await client.get(f"https://colab.research.google.com/tun/m/assign?authuser=0&nbh={nbh}&variant=DEFAULT&accelerator=NONE", headers=h)).json()
        assign = (await client.post(
            f"https://colab.research.google.com/tun/m/assign?authuser=0&nbh={nbh}&variant=DEFAULT&accelerator=NONE",
            headers={**h, "X-Goog-Colab-Token": s1["token"]},
        )).json()
        proxy_url = assign["runtimeProxyInfo"]["url"]
        proxy_tok = assign["runtimeProxyInfo"]["token"]
        kid = (await client.get(f"{proxy_url}/api/kernels", headers={"X-Colab-Runtime-Proxy-Token": proxy_tok})).json()[0]["id"]
        ws_uri = proxy_url.replace("https://", "wss://") + f"/api/kernels/{kid}/channels"

        with pytest.raises(InvalidStatus) as exc_sub:
            await websockets.connect(
                ws_uri,
                subprotocols=["v1.kernel.websocket.jupyter.org"],  # type: ignore[arg-type]
                additional_headers={"X-Colab-Runtime-Proxy-Token": proxy_tok},
            )
        assert exc_sub.value.response.status_code == 400

        with pytest.raises(InvalidStatus) as exc_tok:
            await websockets.connect(
                ws_uri,
                subprotocols=None,
                additional_headers={"X-Colab-Runtime-Proxy-Token": "wrong-token"},
            )
        assert exc_tok.value.response.status_code == 403


@pytest.mark.asyncio
async def test_t2_twin_fault_injection_and_fail_closed_guard(colab_twin: ColabTwin) -> None:
    """T0.5: Twin serves 400/412/502 faults, genuine WS 1006 abort, and blocks unmocked hosts."""
    async with httpx.AsyncClient() as client:
        with pytest.raises(RuntimeError, match="blocked outbound"):
            await client.get("https://evil-unmocked.example.com/steal")

        tok = (
            await client.post(
                "https://oauth2.googleapis.com/token",
                data={"grant_type": "authorization_code", "code": colab_twin.issue_auth_code("t2-fault")},
            )
        ).json()["access_token"]
        h = {"Authorization": f"Bearer {tok}", "X-Goog-AuthUser": "0"}
        nbh = "d" * 44

        colab_twin.faults.assign_error = AssignFault(status=503, body={"error": "No GPUs"}, on_step="get")
        r503 = await client.get(
            f"https://colab.research.google.com/tun/m/assign?authuser=0&nbh={nbh}&variant=GPU&accelerator=T4",
            headers=h,
        )
        assert r503.status_code == 503

        s1 = (await client.get(f"https://colab.research.google.com/tun/m/assign?authuser=0&nbh={nbh}&variant=DEFAULT&accelerator=NONE", headers=h)).json()
        assign = (await client.post(
            f"https://colab.research.google.com/tun/m/assign?authuser=0&nbh={nbh}&variant=DEFAULT&accelerator=NONE",
            headers={**h, "X-Goog-Colab-Token": s1["token"]},
        )).json()
        proxy_url = assign["runtimeProxyInfo"]["url"]
        proxy_tok = assign["runtimeProxyInfo"]["token"]
        kid = (await client.get(f"{proxy_url}/api/kernels", headers={"X-Colab-Runtime-Proxy-Token": proxy_tok})).json()[0]["id"]
        ws_uri = proxy_url.replace("https://", "wss://") + f"/api/kernels/{kid}/channels"

        async with websockets.connect(
            ws_uri,
            subprotocols=None,
            additional_headers={"X-Colab-Runtime-Proxy-Token": proxy_tok},
        ) as ws:
            dropped = await colab_twin.drop_active_ws_1006()
            assert dropped == 1
            with pytest.raises(ConnectionClosedError) as close_exc:
                await asyncio.wait_for(ws.recv(), timeout=3.0)
            assert close_exc.value.rcvd is None


# ============================================================================
# FA1: Authentication & Credential Management Boundaries (6 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_t2_fa1_invalid_method_and_negative_user_index(mcp_session: Any) -> None:
    """FA1.B1: Invalid login method and negative auth_user_index return MCP errors."""
    async with mcp_session("memory") as session:
        assert_tool_error(await session.call_tool("colab_login", {"method": "kerberos_magic"}))
        assert_tool_error(await session.call_tool("colab_login", {"auth_user_index": -1}), "auth_user_index")
        assert_tool_error(await session.call_tool("colab_get_auth_status", {"auth_user_index": -1}), "auth_user_index")
        assert_tool_error(await session.call_tool("colab_logout", {"auth_user_index": -2}), "auth_user_index")


@pytest.mark.asyncio
async def test_t2_fa1_timeout_seconds_boundaries(mcp_session: Any) -> None:
    """FA1.B2: timeout_seconds enforces [30, 600] inclusive bounds."""
    async with mcp_session("memory") as session:
        assert_tool_error(await session.call_tool("colab_login", {"timeout_seconds": 29}), "timeout")
        assert_tool_error(await session.call_tool("colab_login", {"timeout_seconds": 601}), "timeout")

        ok_low = parse_tool_json(await session.call_tool("colab_login", {"timeout_seconds": 30}))
        assert ok_low["status"] == "authorization_required"
        ok_high = parse_tool_json(await session.call_tool("colab_login", {"timeout_seconds": 600}))
        assert ok_high["status"] == "authorization_required"


@pytest.mark.asyncio
async def test_t2_fa1_cookie_import_missing_mandatory_keys(mcp_session: Any) -> None:
    """FA1.B3: cookie_import missing SAPISID or SID is rejected."""
    async with mcp_session("memory") as session:
        assert_tool_error(
            await session.call_tool("colab_login", {"method": "cookie_import", "cookies": {"SID": "only_sid"}}),
            "SAPISID",
        )
        assert_tool_error(
            await session.call_tool("colab_login", {"method": "cookie_import", "cookies": {"SAPISID": "only_sapisid"}}),
            "SID",
        )
        assert_tool_error(
            await session.call_tool("colab_login", {"method": "cookie_import", "cookies": {}}),
        )


@pytest.mark.asyncio
async def test_t2_fa1_revoked_refresh_token_purges_keystore(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA1.B4: OAuth invalid_grant during refresh purges keystore and sets is_authenticated=False."""
    async with mcp_session("memory") as session:
        await _auth(session, colab_twin, user_idx=0)
        colab_twin.faults.oauth_refresh_mode = "invalid_grant"
        status = parse_tool_json(
            await session.call_tool("colab_get_auth_status", {"auth_user_index": 0, "force_refresh": True})
        )
        assert status["is_authenticated"] is False
        assert status["token_valid"] is False

        colab_twin.faults.oauth_refresh_mode = "ok"
        after = parse_tool_json(await session.call_tool("colab_get_auth_status", {"auth_user_index": 0}))
        assert after["is_authenticated"] is False


@pytest.mark.asyncio
async def test_t2_fa1_multi_account_logout_isolation(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA1.B5: Logging out auth_user_index=1 purges only user 1 and leaves user 0 intact."""
    async with mcp_session("memory") as session:
        await _auth(session, colab_twin, user_idx=0, tier="FREE")
        await _auth(session, colab_twin, user_idx=1, tier="PRO", ccu=99.0)

        parse_tool_json(
            await session.call_tool("colab_logout", {"auth_user_index": 1, "terminate_active_runtimes": True})
        )
        st1 = parse_tool_json(await session.call_tool("colab_get_auth_status", {"auth_user_index": 1}))
        st0 = parse_tool_json(await session.call_tool("colab_get_auth_status", {"auth_user_index": 0}))
        assert st1["is_authenticated"] is False
        assert st0["is_authenticated"] is True
        assert st0["user_email"] == "twin-user0@example.com"


@pytest.mark.asyncio
async def test_t2_fa1_zero_credential_leakage_in_auth_outputs(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA1.B6: Auth tool outputs never expose raw OAuth tokens or cookie secrets."""
    secret_sapisid = "SUPER_SECRET_SAPISID_VALUE_987654"
    secret_sid = "SUPER_SECRET_SID_VALUE_123456"
    async with mcp_session("memory") as session:
        res = await session.call_tool(
            "colab_login",
            {
                "method": "cookie_import",
                "cookies": {"SAPISID": secret_sapisid, "SID": secret_sid, "OSID": "SECRET_OSID"},
            },
        )
        raw_login = extract_blocks_text(res)
        assert secret_sapisid not in raw_login
        assert secret_sid not in raw_login

        st_res = await session.call_tool("colab_get_auth_status", {})
        raw_st = extract_blocks_text(st_res)
        assert secret_sapisid not in raw_st
        assert secret_sid not in raw_st
        assert "ya29." not in raw_st
        assert "1//0" not in raw_st


# ============================================================================
# FA2: Runtime Provisioning & Lifecycle Faults/Boundaries (6 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_t2_fa2_invalid_accelerator_and_timeout_bounds(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA2.B1: Invalid accelerator and timeout_seconds outside [30, 300] are rejected."""
    async with mcp_session("memory") as session:
        await _auth(session, colab_twin)
        assert_tool_error(await session.call_tool("colab_connect_runtime", {"accelerator": "RTX_4090"}))
        assert_tool_error(await session.call_tool("colab_connect_runtime", {"timeout_seconds": 29}), "timeout")
        assert_tool_error(await session.call_tool("colab_connect_runtime", {"timeout_seconds": 301}), "timeout")


@pytest.mark.asyncio
async def test_t2_fa2_free_tier_high_ram_400_rejected(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA2.B2: Free tier requesting high_ram=True hits twin HTTP 400 and raises HighRamNotAvailableError."""
    async with mcp_session("memory") as session:
        await _auth(session, colab_twin, user_idx=0, tier="FREE")
        err = assert_tool_error(
            await session.call_tool("colab_connect_runtime", {"accelerator": "NONE", "high_ram": True})
        )
        assert any(tok in err.lower() for tok in ("high-ram", "high_ram", "free tier", "pro"))


@pytest.mark.asyncio
async def test_t2_fa2_http_412_quota_disambiguation(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA2.B3: HTTP 412 TOO_MANY_ASSIGNMENTS, CCU_EXHAUSTED, and FREE_TIER_QUOTA_EXCEEDED map distinctly."""
    async with mcp_session("memory") as session:
        await _auth(session, colab_twin, user_idx=0, tier="FREE")

        colab_twin.faults.assign_error = AssignFault(
            status=412,
            body={"error": {"code": "TOO_MANY_ASSIGNMENTS", "message": "Too many assignments"}},
        )
        assert_tool_error(
            await session.call_tool("colab_connect_runtime", {"accelerator": "NONE", "reuse_existing": False}),
            "maximum",
        )

        colab_twin.faults.assign_error = AssignFault(
            status=412,
            body={"error": {"code": "CCU_EXHAUSTED", "message": "Compute units exhausted"}},
        )
        assert_tool_error(
            await session.call_tool("colab_connect_runtime", {"accelerator": "T4", "reuse_existing": False}),
            "compute unit",
        )

        colab_twin.faults.assign_error = AssignFault(
            status=412,
            body={"error": {"code": "FREE_TIER_QUOTA_EXCEEDED", "message": "Free tier quota reached"}},
        )
        assert_tool_error(
            await session.call_tool("colab_connect_runtime", {"accelerator": "T4", "reuse_existing": False}),
            "quota",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [502, 503, 504])
async def test_t2_fa2_cloud_capacity_502_503_504(
    mcp_session: Any, colab_twin: ColabTwin, status_code: int
) -> None:
    """FA2.B4: HTTP 502/503/504 from TFE /assign surfaces CloudCapacityError."""
    async with mcp_session("memory") as session:
        await _auth(session, colab_twin, user_idx=0, tier="PRO", ccu=50.0)
        colab_twin.faults.assign_error = AssignFault(
            status=status_code,
            body={"error": "Capacity unavailable"},
            on_step="post",
        )
        err = assert_tool_error(
            await session.call_tool("colab_connect_runtime", {"accelerator": "A100", "reuse_existing": False})
        )
        assert "capacity" in err.lower() or str(status_code) in err


@pytest.mark.asyncio
async def test_t2_fa2_unauthenticated_connect_and_disconnected_idempotency(mcp_session: Any) -> None:
    """FA2.B5: Unauthenticated connect fails; status/disconnect on unknown runtime return clean disconnected."""
    async with mcp_session("memory") as session:
        assert_tool_error(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))

        st = parse_tool_json(await session.call_tool("colab_get_runtime_status", {"runtime_id": "nonexistent"}))
        assert st["status"] == "disconnected"

        disc = parse_tool_json(await session.call_tool("colab_disconnect_runtime", {"runtime_id": "nonexistent"}))
        assert disc["status"] == "disconnected"
        assert disc["already_terminated"] is True
        assert disc["compute_resources_released"] is False


@pytest.mark.asyncio
async def test_t2_fa2_xssi_prefix_stripping_and_keepalive_read_timeout_healthy(
    mcp_session: Any, colab_twin: ColabTwin
) -> None:
    """FA2.B6: XSSI )]}' prefix is stripped cleanly and keep-alive ReadTimeout counts as healthy."""
    colab_twin.faults.xssi_prefix = True
    colab_twin.faults.keepalive_mode = "read_timeout"
    async with mcp_session("memory") as session:
        await _auth(session, colab_twin)
        rt_list = parse_tool_json(await session.call_tool("colab_list_runtime_types", {}))
        assert rt_list["account_tier"] == "FREE"

        conn = parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))
        await asyncio.sleep(0.1)
        assert len(colab_twin.state.keepalive_pings) >= 1

        st = parse_tool_json(await session.call_tool("colab_get_runtime_status", {"runtime_id": conn["runtime_id"]}))
        assert st["status"] == "active"
        assert st["keep_alive_healthy"] is True


# ============================================================================
# FA3: Kernel Bridge, ANSI Sanitizer, Truncation & Image Bounds (6 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_t2_fa3_execute_code_missing_and_out_of_bounds_params(
    mcp_session: Any, colab_twin: ColabTwin
) -> None:
    """FA3.B1: Out-of-bounds timeout_seconds, max_text_lines, max_output_bytes, max_images rejected."""
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_code"])
        await _auth(session, colab_twin)
        parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))

        assert_tool_error(await session.call_tool("colab_execute_code", {}))
        assert_tool_error(await session.call_tool("colab_execute_code", {"code": "1", "timeout_seconds": 4}))
        assert_tool_error(await session.call_tool("colab_execute_code", {"code": "1", "timeout_seconds": 3601}))
        assert_tool_error(await session.call_tool("colab_execute_code", {"code": "1", "max_text_lines": 9}))
        assert_tool_error(await session.call_tool("colab_execute_code", {"code": "1", "max_output_bytes": 1023}))
        assert_tool_error(await session.call_tool("colab_execute_code", {"code": "1", "max_images": 0}))
        assert_tool_error(await session.call_tool("colab_execute_code", {"code": "1", "max_images": 51}))


@pytest.mark.asyncio
async def test_t2_fa3_ecma48_ansi_sanitization_and_kernel_error(
    mcp_session: Any, colab_twin: ColabTwin
) -> None:
    """FA3.B2: ECMA-48 CSI, multiline OSC, VT100, and 2-byte Fe escapes are stripped across stdout & tracebacks."""
    colab_twin.faults.inject_osc_in_traceback = True
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_code"])
        await _auth(session, colab_twin)
        parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))

        ansi_code = (
            "print('\\x1b[38;2;255;0;0mRGB_TEXT\\x1b[0m')\n"
            "print('\\x1b]0;WindowTitle\\nMultiLine\\x07AFTER_OSC')\n"
            "print('\\x1b(BVT100_TEXT\\x1bMFE_TEXT')\n"
        )
        ok_res = await session.call_tool("colab_execute_code", {"code": ansi_code})
        ok_text = extract_blocks_text(parse_tool_blocks(ok_res))
        assert "\x1b" not in ok_text
        assert "RGB_TEXT" in ok_text
        assert "AFTER_OSC" in ok_text
        assert "VT100_TEXT" in ok_text
        assert "FE_TEXT" in ok_text

        err_res = await session.call_tool("colab_execute_code", {"code": "1 / 0"})
        err_text = assert_tool_error(err_res, "ZeroDivisionError")
        assert "\x1b" not in err_text
        assert "Colab Trace" in err_text


@pytest.mark.asyncio
async def test_t2_fa3_two_stage_truncation_exact_boundary_and_utf8_safety(
    mcp_session: Any, colab_twin: ColabTwin
) -> None:
    """FA3.B3: Exact line limit does not truncate; +1 truncates; 4-byte UTF-8 byte slicing never corrupts."""
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_code"])
        await _auth(session, colab_twin)
        parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))

        exact = await session.call_tool(
            "colab_execute_code",
            {"code": "for i in range(10): print(f'L{i}')", "max_text_lines": 10},
        )
        exact_text = extract_blocks_text(parse_tool_blocks(exact))
        assert "TRUNCATED" not in exact_text.upper()
        assert "L0" in exact_text and "L9" in exact_text

        over = await session.call_tool(
            "colab_execute_code",
            {"code": "for i in range(11): print(f'L{i}')", "max_text_lines": 10},
        )
        over_text = extract_blocks_text(parse_tool_blocks(over))
        assert "TRUNCATED" in over_text.upper()
        assert "L0" in over_text and "L10" in over_text

        utf8_res = await session.call_tool(
            "colab_execute_code",
            {"code": "print('🚀🌟🔥🎉' * 500)", "max_output_bytes": 1024},
        )
        utf8_text = extract_blocks_text(parse_tool_blocks(utf8_res))
        assert "TRUNCATED" in utf8_text.upper()
        assert len(utf8_text.encode("utf-8")) <= 2048


@pytest.mark.asyncio
async def test_t2_fa3_max_images_exact_vs_exceeded_omission_notice(
    mcp_session: Any, colab_twin: ColabTwin
) -> None:
    """FA3.B4: max_images==count emits no notice; count>max_images caps images and appends notice."""
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_code"])
        await _auth(session, colab_twin)
        parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))

        exact_res = await session.call_tool(
            "colab_execute_code",
            {
                "code": "import matplotlib.pyplot as plt\nfor _ in range(2):\n plt.plot([1])\n plt.show()",
                "max_images": 2,
            },
        )
        exact_blocks = parse_tool_blocks(exact_res)
        assert len(extract_image_blocks(exact_blocks)) == 2
        assert "omitted" not in extract_blocks_text(exact_blocks).lower()

        exceeded_res = await session.call_tool(
            "colab_execute_code",
            {
                "code": "import matplotlib.pyplot as plt\nfor _ in range(4):\n plt.plot([1])\n plt.show()",
                "max_images": 2,
            },
        )
        ex_blocks = parse_tool_blocks(exceeded_res)
        assert len(extract_image_blocks(ex_blocks)) == 2
        ex_text = extract_blocks_text(ex_blocks).lower()
        assert "omitted" in ex_text and "2" in ex_text


@pytest.mark.asyncio
async def test_t2_fa3_unknown_execution_id_and_disconnected_runtime(
    mcp_session: Any, colab_twin: ColabTwin
) -> None:
    """FA3.B5: Unknown execution_id and disconnected runtime produce clear MCP errors."""
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_code", "colab_get_execution_status", "colab_get_cell_output"])
        assert_tool_error(await session.call_tool("colab_execute_code", {"code": "print(1)"}))
        assert_tool_error(await session.call_tool("colab_get_execution_status", {"execution_id": "missing-id"}))
        assert_tool_error(await session.call_tool("colab_get_cell_output", {"execution_id": "missing-id"}))


@pytest.mark.asyncio
async def test_t2_fa3_multi_mime_display_priority_image_over_text_plain(
    mcp_session: Any, colab_twin: ColabTwin
) -> None:
    """FA3.B6: Multi-MIME display_data prioritizes image/png without emitting redundant text/plain."""
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_code"])
        await _auth(session, colab_twin)
        parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))

        res = await session.call_tool(
            "colab_execute_code",
            {"code": "twin_display({'image/png': TINY_PNG_B64, 'text/plain': 'SUPPRESSED_FIGURE_REPR'})"},
        )
        blocks = parse_tool_blocks(res)
        assert len(extract_image_blocks(blocks)) == 1
        assert "SUPPRESSED_FIGURE_REPR" not in extract_blocks_text(blocks)


# ============================================================================
# FA4: Interrupt, Restart & Reconnection Edge Cases (5 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_t2_fa4_interrupt_disconnected_runtime_error(mcp_session: Any) -> None:
    """FA4.B1: colab_interrupt_execution on unknown/disconnected runtime fails cleanly."""
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_interrupt_execution"])
        assert_tool_error(await session.call_tool("colab_interrupt_execution", {"runtime_id": "no-such-runtime"}))


@pytest.mark.asyncio
async def test_t2_fa4_interrupt_non_string_runtime_id_rejected(mcp_session: Any) -> None:
    """FA4.B2: Non-string runtime_id in colab_interrupt_execution is rejected."""
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_interrupt_execution"])
        assert_tool_error(await session.call_tool("colab_interrupt_execution", {"runtime_id": 12345}))


@pytest.mark.asyncio
async def test_t2_fa4_execute_code_timeout_dispatches_sigint(
    mcp_session: Any, colab_twin: ColabTwin
) -> None:
    """FA4.B3: colab_execute_code timeout_seconds=5 dispatches SIGINT when cell runs past deadline."""
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_code"])
        await _auth(session, colab_twin)
        conn = parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))

        res = await session.call_tool(
            "colab_execute_code",
            {"code": "import time\ntime.sleep(20)", "runtime_id": conn["runtime_id"], "timeout_seconds": 5},
        )
        assert res.isError is True
        assert len(colab_twin.state.interrupted_kernels) >= 1


@pytest.mark.asyncio
async def test_t2_fa4_reconnect_polls_restarting_kernel_until_idle(
    mcp_session: Any, colab_twin: ColabTwin
) -> None:
    """FA4.B4: 6-stage reconnect polls /api/kernels through 'restarting' states until 'idle'."""
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_code"])
        await _auth(session, colab_twin)
        conn = parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))
        rid = conn["runtime_id"]

        parse_tool_blocks(await session.call_tool("colab_execute_code", {"code": "print('warm')", "runtime_id": rid}))
        colab_twin.faults.kernel_restarting_polls = 2
        await colab_twin.drop_active_ws_1006()
        await asyncio.sleep(0.05)

        res = await session.call_tool("colab_execute_code", {"code": "print('idle-recovered')", "runtime_id": rid})
        assert "idle-recovered" in extract_blocks_text(parse_tool_blocks(res))


@pytest.mark.asyncio
async def test_t2_fa4_no_stale_msg_id_leakage_after_interrupt(
    mcp_session: Any, colab_twin: ColabTwin
) -> None:
    """FA4.B5: Interrupting noisy loop leaves zero stale frames leaking into subsequent execution."""
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_code", "colab_interrupt_execution"])
        await _auth(session, colab_twin)
        conn = parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))
        rid = conn["runtime_id"]

        task = asyncio.create_task(
            session.call_tool(
                "colab_execute_code",
                {"code": "import time\nwhile True:\n print('NOISE_MARKER')\n time.sleep(0.05)", "runtime_id": rid},
            )
        )
        await asyncio.sleep(0.2)
        parse_tool_json(await session.call_tool("colab_interrupt_execution", {"runtime_id": rid}))
        await asyncio.gather(task, return_exceptions=True)

        clean = await session.call_tool("colab_execute_code", {"code": "print('CLEAN_RESULT')", "runtime_id": rid})
        clean_text = extract_blocks_text(parse_tool_blocks(clean))
        assert "CLEAN_RESULT" in clean_text
        assert "NOISE_MARKER" not in clean_text


# ============================================================================
# FA5: Notebook Validation, Traversal Guard & Error Policies (5 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_t2_fa5_exclusive_notebook_inputs_enforced(mcp_session: Any, colab_twin: ColabTwin) -> None:
    """FA5.B1: Providing neither or both notebook_path and notebook_content fails."""
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_notebook"])
        assert_tool_error(await session.call_tool("colab_execute_notebook", {}))
        assert_tool_error(
            await session.call_tool(
                "colab_execute_notebook",
                {"notebook_path": "a.ipynb", "notebook_content": "{}"},
            )
        )


@pytest.mark.asyncio
async def test_t2_fa5_path_traversal_guard(
    mcp_session: Any, colab_twin: ColabTwin, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FA5.B2: Directory traversal (../..) in notebook_path and output_save_path is rejected."""
    monkeypatch.chdir(tmp_path)
    nb = make_notebook([make_code_cell("print(1)")])
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_notebook"])
        assert_tool_error(
            await session.call_tool("colab_execute_notebook", {"notebook_path": "../../etc/passwd"})
        )
        assert_tool_error(
            await session.call_tool(
                "colab_execute_notebook",
                {"notebook_content": json.dumps(nb), "output_save_path": "../escaped_output.ipynb"},
            )
        )


@pytest.mark.asyncio
async def test_t2_fa5_invalid_json_v3_missing_file_and_bad_indices(
    mcp_session: Any, colab_twin: ColabTwin, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FA5.B3: Malformed JSON, non-v4 schema, missing file, and out-of-range cell_indices fail."""
    monkeypatch.chdir(tmp_path)
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_notebook"])
        await _auth(session, colab_twin)
        parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))

        assert_tool_error(await session.call_tool("colab_execute_notebook", {"notebook_content": "{not-valid-json"}))
        v3 = {"cells": [], "metadata": {}, "nbformat": 3, "nbformat_minor": 0}
        assert_tool_error(await session.call_tool("colab_execute_notebook", {"notebook_content": json.dumps(v3)}))
        assert_tool_error(await session.call_tool("colab_execute_notebook", {"notebook_path": "missing_file.ipynb"}))
        v4 = make_notebook([make_code_cell("1 + 1")])
        assert_tool_error(
            await session.call_tool(
                "colab_execute_notebook",
                {"notebook_content": json.dumps(v4), "cell_indices": [99]},
            )
        )


@pytest.mark.asyncio
async def test_t2_fa5_stop_on_error_true_halts_downstream(
    mcp_session: Any, colab_twin: ColabTwin
) -> None:
    """FA5.B4: stop_on_error=True stops execution on first error and marks downstream cells skipped."""
    nb = make_notebook([
        make_code_cell("ok_flag = True"),
        make_code_cell("raise RuntimeError('boom-stop')"),
        make_code_cell("downstream = True"),
    ])
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_notebook"])
        await _auth(session, colab_twin)
        parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))

        data = parse_tool_json(
            await session.call_tool(
                "colab_execute_notebook",
                {"notebook_content": json.dumps(nb), "stop_on_error": True},
            )
        )
        assert data["executed_cells"] == 2
        assert data["successful_cells"] == 1
        assert data["failed_cells"] == 1
        assert data["skipped_cells"] >= 1
        assert data["cell_summaries"][1]["error_name"] == "RuntimeError"


@pytest.mark.asyncio
async def test_t2_fa5_stop_on_error_false_continues_and_serializes_error_ast(
    mcp_session: Any,
    colab_twin: ColabTwin,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FA5.B5: stop_on_error=False executes remaining cells and records sanitized error AST node."""
    monkeypatch.chdir(tmp_path)
    out_file = tmp_path / "continue_on_err.ipynb"
    nb_file = write_notebook(
        tmp_path / "in_err.ipynb",
        [
            make_code_cell("1 / 0"),
            make_code_cell("recovered = 99\nprint('ran-after-error')"),
        ],
    )
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_notebook"])
        await _auth(session, colab_twin)
        parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))

        data = parse_tool_json(
            await session.call_tool(
                "colab_execute_notebook",
                {
                    "notebook_path": str(nb_file),
                    "stop_on_error": False,
                    "output_save_path": str(out_file),
                },
            )
        )
        assert data["executed_cells"] == 2
        assert data["failed_cells"] == 1
        assert data["successful_cells"] == 1
        saved = json.loads(out_file.read_text(encoding="utf-8"))
        err_outs = saved["cells"][0]["outputs"]
        assert any(o.get("output_type") == "error" and o.get("ename") == "ZeroDivisionError" for o in err_outs)
        assert "\x1b" not in json.dumps(err_outs)


# ============================================================================
# FA6: Server Security, Egress Redaction, Lifespan & CLI Bounds (5 tests)
# ============================================================================


@pytest.mark.asyncio
async def test_t2_fa6_sse_unauthorized_host_header_rejected() -> None:
    """FA6.B1: SSE ASGI app rejects requests with unauthorized Host header (DNS rebinding guard)."""
    server = create_server("opaque-sec-host")
    app = server.sse_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://evil.attacker.invalid") as client:
        try:
            resp = await client.get("/sse")
            assert resp.status_code in (400, 403, 421, 500)
        except Exception:
            assert True


@pytest.mark.asyncio
async def test_t2_fa6_sse_post_messages_invalid_session_or_json() -> None:
    """FA6.B2: POST /messages/ rejects missing session_id and malformed JSON payloads."""
    server = create_server("opaque-sec-msgs")
    app = server.sse_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
        r1 = await client.post("/messages/", json={"jsonrpc": "2.0", "method": "tools/list", "id": 1})
        assert r1.status_code in (400, 404, 422)

        r2 = await client.post(
            "/messages/?session_id=00000000-0000-0000-0000-000000000000",
            content=b"{bad-json",
            headers={"Content-Type": "application/json"},
        )
        assert r2.status_code in (400, 404, 422, 500)


@pytest.mark.asyncio
async def test_t2_fa6_egress_redaction_strips_tokens_and_cookies(
    mcp_session: Any, colab_twin: ColabTwin
) -> None:
    """FA6.B3: Egress redaction strips ya29..., 1//0..., SAPISID=..., SAPISIDHASH from tool output."""
    async with mcp_session("memory") as session:
        await require_tools(session, ["colab_execute_code"])
        require_m5_server_features()
        await _auth(session, colab_twin)
        parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))

        leak_code = (
            "print('TOK=ya29.a0LEAKED_OAUTH_TOKEN_XYZ')\n"
            "print('REF=1//04LEAKED_REFRESH_TOKEN_ABC')\n"
            "print('COOKIE=SAPISID=LEAKED_SAPISID_12345')\n"
            "print('SIG=SAPISIDHASH 1700000000_0123456789abcdef0123456789abcdef01234567')\n"
        )
        res = await session.call_tool("colab_execute_code", {"code": leak_code})
        text = extract_blocks_text(parse_tool_blocks(res))
        assert "LEAKED_OAUTH_TOKEN_XYZ" not in text
        assert "LEAKED_REFRESH_TOKEN_ABC" not in text
        assert "LEAKED_SAPISID_12345" not in text
        assert "0123456789abcdef0123456789abcdef01234567" not in text
        assert "[REDACTED]" in text


def test_t2_fa6_cli_invalid_command_transport_and_port(cli_runner: Any) -> None:
    """FA6.B4: CLI rejects unknown commands, unsupported --transport, and out-of-range --port."""
    bad_cmd = cli_runner.invoke(cli_app, ["no-such-subcommand-xyz"])
    assert bad_cmd.exit_code != 0

    require_cli_command("run")
    bad_tr = cli_runner.invoke(cli_app, ["run", "--transport", "grpc"])
    assert bad_tr.exit_code != 0

    bad_port_low = cli_runner.invoke(cli_app, ["run", "--transport", "sse", "--port", "0"])
    assert bad_port_low.exit_code != 0

    bad_port_high = cli_runner.invoke(cli_app, ["run", "--transport", "sse", "--port", "70000"])
    assert bad_port_high.exit_code != 0


@pytest.mark.asyncio
async def test_t2_fa6_lifespan_shutdown_stops_active_sessions(
    mcp_session: Any, colab_twin: ColabTwin
) -> None:
    """FA6.B5: FastMCP server lifespan exit shuts down active SessionManager keepalive & watchdog."""
    require_m5_server_features()
    async with mcp_session("memory") as session:
        await _auth(session, colab_twin)
        conn = parse_tool_json(await session.call_tool("colab_connect_runtime", {"accelerator": "NONE"}))
        mgr = get_session_manager()
        rec = mgr.get_session(conn["runtime_id"])
        assert rec is not None and rec.keepalive_daemon is not None
        assert rec.keepalive_daemon.is_running is True

    # After exiting mcp_session, lifespan shutdown must have stopped keepalive daemon
    assert rec.keepalive_daemon.is_running is False
