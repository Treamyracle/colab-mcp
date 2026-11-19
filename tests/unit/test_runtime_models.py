"""
Unit and behavioral tests for Milestone 2 runtime models, TFE auth integration,
SessionManager kernel connection params, and runtime MCP tools.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from colab_mcp.auth.keystore import get_credentials, store_credentials
from colab_mcp.auth.models import AuthenticationRequiredError, AuthMethod, GoogleCredentials
from colab_mcp.auth.tools import colab_logout_impl
from colab_mcp.runtime.client import ColabTFEClient, _strip_xssi_prefix
from colab_mcp.runtime.keepalive import KeepAliveDaemon
from colab_mcp.runtime.models import (
    Accelerator,
    Assignment,
    RuntimeDisconnectedError,
    RuntimeProxyInfo,
    Shape,
    Variant,
    get_accelerator_catalog,
    resolve_variant,
)
from colab_mcp.runtime.session_manager import SessionManager, get_session_manager, reset_session_manager
from colab_mcp.runtime.tools import (
    colab_connect_runtime,
    colab_disconnect_runtime,
    colab_get_runtime_status,
    colab_list_runtime_types,
)


@pytest.fixture(autouse=True)
def clean_singleton_session_manager():
    reset_session_manager()
    yield
    reset_session_manager()


def test_resolve_variant_mapping_and_invalid_raises():
    assert resolve_variant(Accelerator.NONE) == Variant.DEFAULT
    assert resolve_variant("t4") == Variant.GPU
    assert resolve_variant(Accelerator.L4) == Variant.GPU
    assert resolve_variant(Accelerator.A100) == Variant.GPU
    assert resolve_variant(Accelerator.H100) == Variant.GPU
    assert resolve_variant(Accelerator.V5E1) == Variant.TPU
    assert resolve_variant("v6e1") == Variant.TPU

    with pytest.raises(ValueError, match="Unsupported accelerator"):
        resolve_variant("RTX_4090")


def test_accelerator_catalog_tiers():
    free_catalog = {item.accelerator: item for item in get_accelerator_catalog("FREE")}
    assert free_catalog["NONE"].available is True
    assert free_catalog["T4"].available is True
    assert free_catalog["T4"].ccu_cost_per_hour == 1.96
    assert free_catalog["L4"].available is False
    assert free_catalog["A100"].available is False

    pro_catalog = {item.accelerator: item for item in get_accelerator_catalog("PRO")}
    assert pro_catalog["A100"].available is True
    assert pro_catalog["H100"].available is False

    pro_plus_catalog = {item.accelerator: item for item in get_accelerator_catalog("PRO_PLUS")}
    assert pro_plus_catalog["H100"].available is True
    assert pro_plus_catalog["V6E1"].available is True


def test_assignment_dual_schema_and_proxy_info_dict_compat():
    camel = Assignment.model_validate(
        {
            "endpoint": "ep-camel",
            "accelerator": "T4",
            "runtimeProxyInfo": {
                "url": "https://colab.research.google.com/tun/m/ep-camel",
                "token": "tok-camel",
                "tokenExpiresInSeconds": 1800,
            },
        }
    )
    assert camel.variant == "GPU"
    assert camel.runtime_proxy_url == "https://colab.research.google.com/tun/m/ep-camel"
    assert camel.runtime_proxy_token == "tok-camel"
    assert camel.token_expires_in == 1800
    assert camel.runtimeProxyInfo is not None
    assert camel.runtimeProxyInfo.get("token") == "tok-camel"
    assert camel.runtimeProxyInfo["url"] == "https://colab.research.google.com/tun/m/ep-camel"

    snake = Assignment.model_validate(
        {
            "endpoint": "ep-snake",
            "accelerator": "NONE",
            "runtime_proxy_url": "https://colab.research.google.com/tun/m/ep-snake",
            "runtime_proxy_token": "tok-snake",
            "token_expires_in": 3600,
        }
    )
    assert snake.variant == "DEFAULT"
    assert snake.runtimeProxyInfo is not None
    assert snake.runtimeProxyInfo.token == "tok-snake"


@pytest.mark.asyncio
async def test_tfe_client_uses_stored_oauth_and_sapisid_headers_and_strips_xssi(
    mock_keystore_dir, mock_keyring, mock_machine_id, sample_credentials, mock_cookie_credentials
):
    # 1. Unauthenticated raises AuthenticationRequiredError genuinely
    unauth_client = ColabTFEClient(
        auth_user_index=0,
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=[])),
    )
    with pytest.raises(AuthenticationRequiredError):
        await unauth_client.list_assignments()
    await unauth_client.close()

    # 2. OAuth stored creds -> Bearer header + XSSI prefix stripping
    store_credentials(0, sample_credentials)
    seen_headers: List[Dict[str, str]] = []

    def oauth_handler(req: httpx.Request) -> httpx.Response:
        seen_headers.append(dict(req.headers))
        return httpx.Response(
            200,
            text=")]}'\n"
            + '[{"endpoint": "ep-oauth", "accelerator": "NONE", "runtime_proxy_url": "u", "runtime_proxy_token": "t"}]',
        )

    async with ColabTFEClient(auth_user_index=0, transport=httpx.MockTransport(oauth_handler)) as client:
        assignments = await client.list_assignments()
    assert len(assignments) == 1
    assert assignments[0].endpoint == "ep-oauth"
    assert seen_headers[0]["authorization"] == f"Bearer {sample_credentials.access_token}"

    # 3. Cookie stored creds on index 1 -> SAPISIDHASH + Cookie headers
    cookie_creds = GoogleCredentials(
        auth_method=AuthMethod.COOKIE_IMPORT,
        auth_user_index=1,
        cookies=mock_cookie_credentials.cookies,
    )
    store_credentials(1, cookie_creds)
    cookie_headers: List[Dict[str, str]] = []

    def cookie_handler(req: httpx.Request) -> httpx.Response:
        cookie_headers.append(dict(req.headers))
        return httpx.Response(200, json={"assignments": []})

    async with ColabTFEClient(auth_user_index=1, transport=httpx.MockTransport(cookie_handler)) as c_client:
        await c_client.list_assignments()
    assert cookie_headers[0]["authorization"].startswith("SAPISIDHASH ")
    assert "SAPISID=" in cookie_headers[0]["cookie"]
    assert cookie_headers[0]["x-goog-authuser"] == "1"


@pytest.mark.asyncio
async def test_session_manager_kernel_params_refresh_and_disconnected_error(monkeypatch):
    mgr = SessionManager()
    with pytest.raises(RuntimeDisconnectedError):
        await mgr.get_kernel_connection_params()

    initial_assignment = Assignment(
        endpoint="ep-refresh",
        accelerator="T4",
        variant="GPU",
        runtime_proxy_url="https://colab.research.google.com/tun/m/ep-refresh",
        runtime_proxy_token="old-token",
        token_expires_in=3600,
    )
    rec = mgr.register_session(
        endpoint="ep-refresh",
        auth_user_index=0,
        assignment=initial_assignment,
        kernel_id="kernel-uuid-1",
    )
    # Simulate token acquired 3400s ago (within 300s refresh buffer)
    rec.token_acquired_at = time.time() - 3400.0

    refreshed_assignment = Assignment(
        endpoint="ep-refresh",
        accelerator="T4",
        variant="GPU",
        runtime_proxy_url="https://colab.research.google.com/tun/m/ep-refresh",
        runtime_proxy_token="fresh-token",
        token_expires_in=3600,
    )

    async def fake_list_assignments(self):
        return [refreshed_assignment]

    monkeypatch.setattr(ColabTFEClient, "list_assignments", fake_list_assignments)
    params = await mgr.get_kernel_connection_params("ep-refresh")
    assert params.proxy_token == "fresh-token"
    assert params.kernel_id == "kernel-uuid-1"
    assert params.token_expires_at > datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_runtime_tools_end_to_end_with_authenticated_mock_transport(
    mock_keystore_dir, mock_keyring, mock_machine_id, sample_credentials, monkeypatch
):
    store_credentials(0, sample_credentials)

    active_sessions: List[Dict[str, Any]] = []
    unassigned_endpoints: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/tun/m/ccu-info":
            return httpx.Response(
                200,
                json={
                    "user_tier": "PRO",
                    "compute_units_remaining": 42.5,
                    "consumption_rate_per_hour": 1.96,
                    "active_assignments_count": len(active_sessions),
                    "auto_refill_enabled": True,
                },
            )
        if path == "/tun/m/assignments":
            return httpx.Response(200, json={"assignments": active_sessions})
        if path == "/tun/m/assign" and request.method == "GET":
            return httpx.Response(200, json={"token": "xsrf-abc"})
        if path == "/tun/m/assign" and request.method == "POST":
            session = {
                "endpoint": "colab-ep-live-1",
                "accelerator": "T4",
                "variant": "GPU",
                "runtimeProxyInfo": {
                    "url": "https://colab.research.google.com/tun/m/colab-ep-live-1",
                    "token": "proxy-tok-1",
                    "tokenExpiresInSeconds": 3600,
                },
            }
            active_sessions.append(session)
            return httpx.Response(200, json=session)
        if path.startswith("/tun/m/unassign/") and request.method == "GET":
            return httpx.Response(200, json={"token": "unassign-tok"})
        if path.startswith("/tun/m/unassign/") and request.method == "POST":
            ep = path.split("/")[-1]
            unassigned_endpoints.append(ep)
            active_sessions.clear()
            return httpx.Response(200, json={"status": "unassigned"})
        if path.endswith("/api/kernels"):
            return httpx.Response(200, json=[{"id": "kern-123", "execution_state": "idle"}])
        if path.endswith("/keep-alive/"):
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    orig_async_client = httpx.AsyncClient

    def patched_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return orig_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_async_client)

    types_res = await colab_list_runtime_types(auth_user_index=0)
    assert types_res.account_tier == "PRO"

    conn_res = await colab_connect_runtime(accelerator="T4", high_ram=False, reuse_existing=True)
    assert conn_res.runtime_id == "colab-ep-live-1"
    assert conn_res.status == "connected"
    assert conn_res.kernel_id == "kern-123"
    assert conn_res.reused_existing is False

    status_res = await colab_get_runtime_status("colab-ep-live-1")
    assert status_res.status == "active"
    assert status_res.kernel_state == "idle"
    assert status_res.keep_alive_healthy is True

    disc_res = await colab_disconnect_runtime("colab-ep-live-1")
    assert disc_res.status == "terminated"
    assert disc_res.compute_resources_released is True
    assert unassigned_endpoints == ["colab-ep-live-1"]

    after_status = await colab_get_runtime_status("colab-ep-live-1")
    assert after_status.status == "disconnected"
    await get_session_manager().shutdown()


def test_assignment_runtime_proxy_info_instance_and_zero_ttl_preserved():
    inst = Assignment(
        endpoint="ep-inst",
        accelerator="T4",
        runtimeProxyInfo=RuntimeProxyInfo(
            url="https://proxy.example/inst",
            token="tok-inst",
            tokenExpiresInSeconds=1800,
        ),
    )
    assert inst.variant == "GPU"
    assert inst.runtime_proxy_url == "https://proxy.example/inst"
    assert inst.runtime_proxy_token == "tok-inst"
    assert inst.token_expires_in == 1800

    inst_zero = Assignment(
        endpoint="ep-inst-zero",
        accelerator="NONE",
        runtimeProxyInfo=RuntimeProxyInfo(
            url="https://proxy.example/zero",
            token="tok-zero",
            tokenExpiresInSeconds=0,
        ),
    )
    assert inst_zero.token_expires_in == 0
    assert inst_zero.runtimeProxyInfo is not None
    assert inst_zero.runtimeProxyInfo.token_expires_in_seconds == 0

    validated_zero = Assignment.model_validate(
        {
            "endpoint": "ep0",
            "accelerator": "T4",
            "runtimeProxyInfo": {
                "url": "https://p",
                "token": "t",
                "tokenExpiresInSeconds": 0,
            },
        }
    )
    assert validated_zero.token_expires_in == 0
    assert validated_zero.runtimeProxyInfo is not None
    assert validated_zero.runtimeProxyInfo.token_expires_in_seconds == 0


@pytest.mark.asyncio
async def test_session_manager_zero_ttl_triggers_immediate_refresh(monkeypatch):
    mgr = SessionManager()
    expired_assignment = Assignment(
        endpoint="ep-zero-ttl",
        accelerator="T4",
        variant="GPU",
        runtime_proxy_url="https://colab.research.google.com/tun/m/ep-zero-ttl",
        runtime_proxy_token="expired-tok",
        token_expires_in=0,
    )
    mgr.register_session(
        endpoint="ep-zero-ttl",
        auth_user_index=0,
        assignment=expired_assignment,
        kernel_id="kern-zero",
    )

    refreshed_assignment = Assignment(
        endpoint="ep-zero-ttl",
        accelerator="T4",
        variant="GPU",
        runtime_proxy_url="https://colab.research.google.com/tun/m/ep-zero-ttl",
        runtime_proxy_token="refreshed-tok",
        token_expires_in=3600,
    )

    async def fake_list_assignments(self):
        return [refreshed_assignment]

    monkeypatch.setattr(ColabTFEClient, "list_assignments", fake_list_assignments)
    params = await mgr.get_kernel_connection_params("ep-zero-ttl")
    assert params.proxy_token == "refreshed-tok"


@pytest.mark.asyncio
async def test_unassign_step2_post_404_idempotent_and_url_escaped():
    seen_paths: List[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen_paths.append(req.url.raw_path.decode("ascii"))
        if req.method == "GET":
            return httpx.Response(200, json={"token": "xsrf-step1"})
        if req.method == "POST":
            return httpx.Response(404, json={"error": "already gone"})
        return httpx.Response(500)

    async with ColabTFEClient(
        auth_token="tok",
        auth_user_index=0,
        transport=httpx.MockTransport(handler),
    ) as client:
        ok = await client.unassign_runtime("ep/with?special#chars")

    assert ok is True
    assert len(seen_paths) == 2
    assert "/tun/m/unassign/ep%2Fwith%3Fspecial%23chars" in seen_paths[0]


@pytest.mark.asyncio
async def test_keepalive_restart_after_stop_and_auth_failure_unhealthy(
    mock_keystore_dir, mock_keyring, mock_machine_id
):
    ping_paths: List[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        ping_paths.append(req.url.raw_path.decode("ascii"))
        return httpx.Response(200, json={"status": "ok"})

    daemon = KeepAliveDaemon(
        endpoint="ep/restart?1",
        auth_token="tok-restart",
        interval_seconds=0.02,
        transport=httpx.MockTransport(handler),
    )
    await daemon.start()
    await asyncio.sleep(0.06)
    await daemon.stop()
    first_count = len(ping_paths)
    assert first_count >= 1
    assert daemon.client.is_closed is True

    await daemon.start()
    assert daemon.client.is_closed is False
    await asyncio.sleep(0.06)
    await daemon.stop()
    assert len(ping_paths) > first_count
    assert "/tun/m/ep%2Frestart%3F1/keep-alive/" in ping_paths[0]

    # Auth failure (no stored credentials, auth_token=None) must mark unhealthy
    unauth_daemon = KeepAliveDaemon(
        endpoint="ep-unauth",
        auth_token=None,
        auth_user_index=0,
        interval_seconds=0.05,
        transport=httpx.MockTransport(handler),
    )
    assert unauth_daemon.is_healthy is True
    ok = await unauth_daemon.ping_once()
    assert ok is False
    assert unauth_daemon.is_healthy is False
    assert unauth_daemon.consecutive_failures == 1
    await unauth_daemon.stop()


@pytest.mark.asyncio
async def test_session_manager_record_activity_scoping_and_remote_disappearance(monkeypatch):
    mgr = SessionManager()
    old_ts = time.time() - 500.0
    mgr.register_session("ep-1", auth_user_index=0, last_activity=old_ts)
    mgr.register_session("ep-2", auth_user_index=0, last_activity=old_ts)
    assert mgr.active_endpoint == "ep-2"

    # Activity on non-active ep-1 must NOT reset idle_seconds for active ep-2
    mgr.record_activity(endpoint="ep-1")
    assert mgr.idle_seconds >= 490.0
    assert mgr.get_session("ep-2").last_activity_time == old_ts
    assert mgr.get_session("ep-1").last_activity_time > old_ts

    # Remote disappearance stops keepalive daemon and clears active references
    mock_daemon = MagicMock()
    mock_daemon.stop = AsyncMock()
    expired_assignment = Assignment(
        endpoint="ep-gone",
        accelerator="T4",
        variant="GPU",
        runtime_proxy_url="https://colab.research.google.com/tun/m/ep-gone",
        runtime_proxy_token="tok-gone",
        token_expires_in=0,
    )
    mgr.register_session(
        "ep-gone",
        auth_user_index=0,
        assignment=expired_assignment,
        keepalive_daemon=mock_daemon,
    )

    async def fake_list_assignments_empty(self):
        return []

    monkeypatch.setattr(ColabTFEClient, "list_assignments", fake_list_assignments_empty)
    with pytest.raises(RuntimeDisconnectedError):
        await mgr.get_kernel_connection_params("ep-gone")

    mock_daemon.stop.assert_awaited_once()
    assert not mgr.has_active_session("ep-gone")
    assert mgr.active_endpoint is None
    assert mgr.keepalive_daemon is None


@pytest.mark.asyncio
async def test_terminate_all_sessions_and_logout_resilient_when_tfe_offline(
    mock_keystore_dir, mock_keyring, mock_machine_id, sample_credentials, monkeypatch
):
    store_credentials(0, sample_credentials)

    def offline_handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("TFE offline")

    transport = httpx.MockTransport(offline_handler)
    orig_async_client = httpx.AsyncClient

    def patched_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return orig_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_async_client)

    mgr = get_session_manager()
    d1 = MagicMock()
    d1.stop = AsyncMock()
    d2 = MagicMock()
    d2.stop = AsyncMock()
    d_other = MagicMock()
    d_other.stop = AsyncMock()

    mgr.register_session("ep-u0-1", auth_user_index=0, keepalive_daemon=d1)
    mgr.register_session("ep-u0-2", auth_user_index=0, keepalive_daemon=d2)
    mgr.register_session("ep-u1-1", auth_user_index=1, keepalive_daemon=d_other)

    logout_resp = await colab_logout_impl(auth_user_index=0, terminate_active_runtimes=True)
    assert logout_resp.status == "logged_out"
    assert logout_resp.keystore_purged is True
    assert logout_resp.terminated_runtimes_count == 2
    assert get_credentials(0) is None

    d1.stop.assert_awaited_once()
    d2.stop.assert_awaited_once()
    d_other.stop.assert_not_awaited()
    assert not mgr.has_active_session("ep-u0-1")
    assert not mgr.has_active_session("ep-u0-2")
    assert mgr.has_active_session("ep-u1-1")


@pytest.mark.asyncio
async def test_watchdog_loop_concurrency_stress_leaves_zero_running_daemons():
    unassigned: List[str] = []

    async def slow_unassign(endpoint: str) -> bool:
        await asyncio.sleep(0.03)
        unassigned.append(endpoint)
        return True

    sm = SessionManager(inactivity_timeout_seconds=0.05, unassign_handler=slow_unassign)
    created_daemons: List[KeepAliveDaemon] = []
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"status": "ok"}))

    await sm.start_watchdog_loop(check_interval_seconds=0.02)

    async def worker(worker_id: int) -> None:
        for step in range(4):
            ep = f"ep-stress-{worker_id}-{step % 2}"
            daemon = KeepAliveDaemon(
                endpoint=ep,
                auth_token=f"tok-{worker_id}-{step}",
                interval_seconds=0.02,
                transport=transport,
            )
            created_daemons.append(daemon)
            sm.register_session(
                ep,
                auth_user_index=0,
                last_activity=time.time(),
                keepalive_daemon=daemon,
            )
            await daemon.start()
            await asyncio.sleep(0.02)
            sm.record_activity(endpoint=ep)
            await asyncio.sleep(0.04)
            if step % 2 == 0:
                await sm.terminate_session(ep)

    await asyncio.gather(*(worker(i) for i in range(4)))
    await sm.shutdown()

    still_running = [d.endpoint for d in created_daemons if d.is_running]
    unclosed_clients = [d.endpoint for d in created_daemons if not d.client.is_closed]
    assert still_running == []
    assert unclosed_clients == []


@pytest.mark.asyncio
async def test_check_inactivity_watchdog_toctou_reregister_preserves_new_session():
    unassign_started = asyncio.Event()
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"status": "ok"}))

    async def slow_unassign(endpoint: str) -> bool:
        unassign_started.set()
        await asyncio.sleep(0.03)
        return True

    sm = SessionManager(inactivity_timeout_seconds=0.05, unassign_handler=slow_unassign)
    old_daemon = KeepAliveDaemon(
        endpoint="ep-race",
        auth_token="tok-old",
        interval_seconds=0.05,
        transport=transport,
    )
    new_daemon = KeepAliveDaemon(
        endpoint="ep-race",
        auth_token="tok-new",
        interval_seconds=0.05,
        transport=transport,
    )
    await old_daemon.start()
    sm.register_session(
        "ep-race",
        auth_user_index=0,
        last_activity=time.time() - 10.0,
        keepalive_daemon=old_daemon,
    )

    async def reregister_mid_unassign() -> None:
        await unassign_started.wait()
        await new_daemon.start()
        sm.register_session(
            "ep-race",
            auth_user_index=0,
            last_activity=time.time(),
            keepalive_daemon=new_daemon,
        )

    await asyncio.gather(
        sm.check_inactivity_watchdog(0.05),
        reregister_mid_unassign(),
    )

    assert sm.has_active_session("ep-race") is True
    rec = sm.get_session("ep-race")
    assert rec is not None
    assert rec.keepalive_daemon is new_daemon
    assert sm.active_endpoint == "ep-race"
    assert sm.keepalive_daemon is new_daemon
    assert sm.session_status == "CONNECTED"
    assert old_daemon.is_running is False

    await sm.shutdown()
    assert new_daemon.is_running is False
    assert new_daemon.client.is_closed is True


@pytest.mark.asyncio
async def test_low_polish_unassign_validation_xssi_comma_and_none_ttl():
    async with ColabTFEClient(
        auth_token="tok",
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={})),
    ) as client:
        for invalid in ("", ".", ".."):
            with pytest.raises(ValueError, match="Invalid endpoint identifier"):
                await client.unassign_runtime(invalid)

    assert _strip_xssi_prefix(")]}',\n{}") == "{}"
    assert _strip_xssi_prefix(")]}',\r\n{}") == "{}"
    assert _strip_xssi_prefix(")]}',{}") == "{}"
    assert _strip_xssi_prefix(")]}'\n{}") == "{}"

    nested_none = Assignment.model_validate(
        {
            "endpoint": "ep-nested-none",
            "accelerator": "T4",
            "runtimeProxyInfo": {
                "url": "https://proxy.example/n",
                "token": "tok-n",
                "tokenExpiresInSeconds": None,
            },
        }
    )
    assert nested_none.token_expires_in == 3600
    assert nested_none.runtimeProxyInfo is not None
    assert nested_none.runtimeProxyInfo.token_expires_in_seconds == 3600

    nested_snake_none = RuntimeProxyInfo.model_validate(
        {
            "url": "https://proxy.example/sn",
            "token": "tok-sn",
            "token_expires_in": None,
        }
    )
    assert nested_snake_none.token_expires_in_seconds == 3600

    flat_none = Assignment.model_validate(
        {
            "endpoint": "ep-flat-none",
            "accelerator": "NONE",
            "runtime_proxy_url": "https://proxy.example/f",
            "runtime_proxy_token": "tok-f",
            "token_expires_in": None,
        }
    )
    assert flat_none.token_expires_in == 3600
    assert flat_none.runtimeProxyInfo is not None
    assert flat_none.runtimeProxyInfo.token_expires_in_seconds == 3600

    flat_zero = Assignment.model_validate(
        {
            "endpoint": "ep-flat-zero",
            "accelerator": "NONE",
            "runtime_proxy_url": "https://proxy.example/z",
            "runtime_proxy_token": "tok-z",
            "token_expires_in": 0,
        }
    )
    assert flat_zero.token_expires_in == 0
    assert flat_zero.runtimeProxyInfo is not None
    assert flat_zero.runtimeProxyInfo.token_expires_in_seconds == 0

    # Switching active session via register_session(..., keepalive_daemon=None)
    # must not leave prior endpoint's daemon dangling in sm.keepalive_daemon.
    sm = SessionManager()
    d1 = MagicMock()
    d1.stop = AsyncMock()
    sm.register_session("ep-1", keepalive_daemon=d1)
    sm.register_session("ep-2", keepalive_daemon=None)
    assert sm.active_endpoint == "ep-2"
    assert sm.keepalive_daemon is None
    assert sm.get_session("ep-1").keepalive_daemon is d1


@pytest.mark.asyncio
async def test_list_sessions_and_remote_disconnect_lifecycle(
    mock_keystore_dir, mock_keyring, mock_machine_id, sample_credentials, monkeypatch
):
    from colab_mcp.runtime.tools import colab_list_sessions

    store_credentials(0, sample_credentials)

    remote_sessions = [
        {
            "endpoint": "colab-local-1",
            "accelerator": "T4",
            "variant": "GPU",
            "runtimeProxyInfo": {
                "url": "https://colab.research.google.com/tun/m/colab-local-1",
                "token": "tok-local-1",
                "tokenExpiresInSeconds": 3600,
            },
        },
        {
            "endpoint": "colab-browser-orphan",
            "accelerator": "NONE",
            "variant": "DEFAULT",
            "runtimeProxyInfo": {
                "url": "https://colab.research.google.com/tun/m/colab-browser-orphan",
                "token": "tok-orphan",
                "tokenExpiresInSeconds": 3600,
            },
        },
    ]
    unassigned: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/tun/m/assignments":
            return httpx.Response(200, json={"assignments": remote_sessions})
        if path == "/tun/m/assign" and request.method == "GET":
            return httpx.Response(200, json={"token": "xsrf"})
        if path == "/tun/m/assign" and request.method == "POST":
            return httpx.Response(200, json=remote_sessions[0])
        if path.startswith("/tun/m/unassign/") and request.method == "GET":
            return httpx.Response(200, json={"token": "unassign-tok"})
        if path.startswith("/tun/m/unassign/") and request.method == "POST":
            ep = path.split("/")[-1]
            unassigned.append(ep)
            remote_sessions[:] = [s for s in remote_sessions if s["endpoint"] != ep]
            return httpx.Response(200, json={"status": "unassigned"})
        if path.endswith("/api/kernels"):
            return httpx.Response(200, json=[{"id": "kern-s1", "execution_state": "busy"}])
        if path.endswith("/keep-alive/"):
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    orig_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: orig_async_client(*args, **{**kwargs, "transport": transport}),
    )

    await colab_connect_runtime(accelerator="T4", reuse_existing=True)

    listed = await colab_list_sessions(auth_user_index=0, probe_kernels=True)
    assert listed.total_sessions == 2
    assert listed.active_runtime_id == "colab-local-1"
    by_id = {s.runtime_id: s for s in listed.sessions}
    assert by_id["colab-local-1"].tracked_locally is True
    assert by_id["colab-local-1"].is_active_local is True
    assert by_id["colab-local-1"].kernel_state == "busy"
    assert by_id["colab-browser-orphan"].tracked_locally is False
    assert by_id["colab-browser-orphan"].kernel_id == "kern-s1"

    # Kill only the remote browser session not tracked locally
    disc_orphan = await colab_disconnect_runtime("colab-browser-orphan")
    assert disc_orphan.status == "terminated"
    assert disc_orphan.compute_resources_released is True
    assert disc_orphan.terminated_runtime_ids == ["colab-browser-orphan"]
    assert disc_orphan.terminated_count == 1

    # Killing an already-gone remote endpoint is idempotent
    disc_missing = await colab_disconnect_runtime("colab-browser-orphan")
    assert disc_missing.status == "disconnected"
    assert disc_missing.already_terminated is True
    assert disc_missing.terminated_count == 0

    # Kill all remaining sessions (local + remote)
    disc_all = await colab_disconnect_runtime(disconnect_all=True)
    assert disc_all.status == "terminated"
    assert disc_all.terminated_runtime_ids == ["colab-local-1"]
    assert disc_all.terminated_count == 1
    assert set(unassigned) == {"colab-browser-orphan", "colab-local-1"}
    await get_session_manager().shutdown()


@pytest.mark.asyncio
async def test_list_sessions_requires_auth_when_no_local_sessions(
    mock_keystore_dir, mock_keyring, mock_machine_id
):
    from colab_mcp.auth.models import AuthenticationRequiredError
    from colab_mcp.runtime.tools import colab_list_sessions

    with pytest.raises(AuthenticationRequiredError):
        await colab_list_sessions(auth_user_index=0)


def test_live_tfe_ccu_camelcase_and_int_enum_assignment_normalization():
    from colab_mcp.runtime.models import CCUInfo, get_accelerator_catalog

    ccu = CCUInfo.model_validate(
        {
            "currentBalance": 452.8,
            "consumptionRateHourly": 1.94,
            "assignmentsCount": 1,
            "eligibleGpus": ["G4", "A100", "L4", "T4"],
            "ineligibleGpus": ["H100"],
            "eligibleTpus": ["V6E1", "V5E1"],
        }
    )
    assert ccu.user_tier == "PRO"
    assert ccu.compute_units_remaining == 452.8
    assert ccu.consumption_rate_per_hour == 1.94
    assert ccu.active_assignments_count == 1

    catalog = {
        item.accelerator: item.available
        for item in get_accelerator_catalog(
            ccu.user_tier,
            eligible_accelerators={*ccu.eligible_gpus, *ccu.eligible_tpus},
            ineligible_accelerators=set(ccu.ineligible_gpus),
        )
    }
    assert catalog["NONE"] is True
    assert catalog["L4"] is True
    assert catalog["V6E1"] is True
    assert catalog["H100"] is False

    assignment = Assignment.model_validate(
        {
            "endpoint": "gpu-l4-s-test",
            "variant": 1,
            "machineShape": 0,
            "accelerator": "L4",
            "runtimeProxyInfo": {
                "url": "https://8080-gpu-l4-s-test.prod.r.colab.dev",
                "token": "tok-l4",
                "tokenExpiresInSeconds": 3600,
            },
        }
    )
    assert assignment.variant == "GPU"
    assert assignment.machine_shape == "STANDARD"
    assert assignment.accelerator == "L4"


@pytest.mark.asyncio
async def test_unassign_runtime_accepts_204_no_content():
    from colab_mcp.runtime.client import ColabTFEClient

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"token": "xsrf-204"})
        if request.method == "POST":
            return httpx.Response(204)
        return httpx.Response(404)

    async with ColabTFEClient(
        auth_token="tok-204",
        transport=httpx.MockTransport(handler),
    ) as client:
        assert await client.unassign_runtime("gpu-l4-s-test") is True


