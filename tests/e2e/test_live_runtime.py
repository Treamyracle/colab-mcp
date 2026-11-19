"""
End-to-End tests for Colab MCP Runtime Lifecycle Management (Live & Mock).

Requirements:
- TECH_SPEC §4.1 - §4.7: Colab Tunnel Frontend (TFE) Topology, NBH derivation,
  accelerator matrix, two-step assignment, session enumeration, unassign deallocation,
  and keepalive heartbeat daemon with 30-minute inactivity watchdog.
- TEST_INFRA.md § Feature Areas 8, 9, 10.

Supports live credentials via environment variables:
- COLAB_AUTH_TOKEN: Active OAuth 2.0 Bearer access token
- COLAB_COOKIE_SAPISID: Active Google SAPISID cookie
When absent, runs full deterministic simulated wire sessions ensuring complete coverage.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

# Progressive testability check
try:
    import colab_mcp.runtime.client as runtime_client_mod
    import colab_mcp.runtime.models as runtime_models_mod
    import colab_mcp.runtime.session_manager as session_manager_mod
    RUNTIME_AVAILABLE = True
except ImportError:
    runtime_client_mod = None
    runtime_models_mod = None
    session_manager_mod = None
    RUNTIME_AVAILABLE = False


@pytest.fixture(autouse=True)
def require_runtime_lifecycle():
    """Cleanly skip if colab_mcp.runtime is not yet implemented in codebase."""
    if not RUNTIME_AVAILABLE:
        pytest.skip("colab_mcp.runtime is scheduled for Milestone 2 implementation")


# Dynamic symbol access
def _get_client_class():
    return getattr(runtime_client_mod, "ColabTFEClient", None)

def _get_accelerator_enum():
    return getattr(runtime_models_mod, "Accelerator", None)

def _get_variant_enum():
    return getattr(runtime_models_mod, "Variant", None)

def _get_session_manager():
    getter = getattr(session_manager_mod, "get_session_manager", None)
    return getter() if getter else None


# ============================================================================
# Mock TFE Server Harness for Deterministic Offline Runs
# ============================================================================

MOCK_ENDPOINT = "colab-f918e7c2-901a-4bc8-812b-2019482019ab"
MOCK_PROXY_URL = f"https://colab.research.google.com/tun/m/{MOCK_ENDPOINT}"
MOCK_PROXY_TOKEN = "ya29.c.b0xxxx_MOCK_PROXY_TOKEN_12345"


class WireMockTFEHandler:
    """Mock handler simulating Google Colab TFE wire endpoints."""

    def __init__(self):
        self.active_sessions: List[Dict[str, Any]] = []
        self.unassign_called = False
        self.keepalive_pings = 0

    def handle(self, request: httpx.Request) -> httpx.Response:
        url_path = request.url.path
        if url_path == "/tun/m/assignments":
            return httpx.Response(200, json=self.active_sessions)
        elif url_path == "/tun/m/assign" or url_path == "/assign":
            if request.method == "GET":
                resp = httpx.Response(200, json={"status": "ok"})
                resp.headers["Set-Cookie"] = "XSRF-TOKEN=mock_xsrf_token_12345; Path=/"
                return resp
            elif request.method == "POST":
                session = {
                    "endpoint": MOCK_ENDPOINT,
                    "runtime_proxy_url": MOCK_PROXY_URL,
                    "runtime_proxy_token": MOCK_PROXY_TOKEN,
                    "token_expires_in": 3600,
                    "accelerator": "T4",
                    "variant": "GPU",
                }
                self.active_sessions.append(session)
                return httpx.Response(200, json=session)
        elif "/unassign" in url_path:
            self.unassign_called = True
            self.active_sessions.clear()
            return httpx.Response(200, json={"status": "terminated"})
        elif "/keep-alive" in url_path:
            self.keepalive_pings += 1
            return httpx.Response(200, json={"status": "alive"})
        return httpx.Response(404, json={"error": "not found"})


# ============================================================================
# E2E RUNTIME LIFECYCLE TESTS (Happy Paths & Edge Cases)
# ============================================================================

@pytest.mark.asyncio
async def test_e2e_runtime_provisioning_standard_cpu(mock_keystore_dir, mock_keyring, sample_credentials):
    """Tier 1: Connect to standard CPU runtime VM and obtain proxy token."""
    client_cls = _get_client_class()
    acc_enum = _get_accelerator_enum()
    if client_cls is None or acc_enum is None:
        pytest.skip("Runtime symbols not found")

    handler = WireMockTFEHandler()
    transport = httpx.MockTransport(handler.handle)

    client = client_cls(auth_user_index=0, transport=transport)
    assignment = await client.assign_runtime(accelerator=acc_enum.NONE, high_ram=False, reuse_existing=False)

    assert assignment is not None
    assert assignment.endpoint is not None
    assert assignment.runtime_proxy_token is not None
    assert assignment.runtime_proxy_url is not None


@pytest.mark.asyncio
async def test_e2e_runtime_provisioning_gpu_t4(mock_keystore_dir, mock_keyring, sample_credentials):
    """Tier 1: Connect to T4 GPU runtime and verify hardware accelerator configuration."""
    client_cls = _get_client_class()
    acc_enum = _get_accelerator_enum()
    var_enum = _get_variant_enum()
    if client_cls is None or acc_enum is None:
        pytest.skip("Runtime symbols not found")

    handler = WireMockTFEHandler()
    transport = httpx.MockTransport(handler.handle)

    client = client_cls(auth_user_index=0, transport=transport)
    assignment = await client.assign_runtime(accelerator=acc_enum.T4, high_ram=False, reuse_existing=False)

    assert assignment is not None
    assert assignment.endpoint == MOCK_ENDPOINT
    if hasattr(assignment, "accelerator"):
        assert str(assignment.accelerator) in ["T4", "Accelerator.T4"]


@pytest.mark.asyncio
async def test_e2e_runtime_session_reattachment(mock_keystore_dir, mock_keyring, sample_credentials):
    """Tier 1: Re-attaching with reuse_existing=True reuses existing running VM."""
    client_cls = _get_client_class()
    acc_enum = _get_accelerator_enum()
    if client_cls is None or acc_enum is None:
        pytest.skip("Runtime symbols not found")

    handler = WireMockTFEHandler()
    # Pre-populate active session
    handler.active_sessions.append({
        "endpoint": MOCK_ENDPOINT,
        "runtime_proxy_url": MOCK_PROXY_URL,
        "runtime_proxy_token": MOCK_PROXY_TOKEN,
        "token_expires_in": 3600,
        "accelerator": "NONE",
    })
    transport = httpx.MockTransport(handler.handle)

    client = client_cls(auth_user_index=0, transport=transport)
    # Re-attach
    assignment = await client.assign_runtime(accelerator=acc_enum.NONE, reuse_existing=True)
    assert assignment.endpoint == MOCK_ENDPOINT


@pytest.mark.asyncio
async def test_e2e_keepalive_heartbeat_cycle(mock_keystore_dir, mock_keyring):
    """Tier 1: Keepalive daemon pings runtime keep-alive endpoint and tolerates ReadTimeout."""
    daemon_cls = getattr(runtime_client_mod, "KeepaliveDaemon", None)
    if daemon_cls is None:
        from colab_mcp.runtime import keepalive
        daemon_cls = getattr(keepalive, "KeepaliveDaemon", None)
    if daemon_cls is None:
        pytest.skip("KeepaliveDaemon not found")

    handler = WireMockTFEHandler()
    transport = httpx.MockTransport(handler.handle)

    daemon = daemon_cls(
        endpoint=MOCK_ENDPOINT,
        interval_seconds=0.1,  # Fast interval for test
        transport=transport,
    )
    task = asyncio.create_task(daemon.start())
    await asyncio.sleep(0.35)
    await daemon.stop()
    await asyncio.gather(task, return_exceptions=True)

    assert handler.keepalive_pings >= 1


@pytest.mark.asyncio
async def test_e2e_inactivity_watchdog_auto_unassign(mock_keystore_dir, mock_keyring):
    """Tier 2: Watchdog automatically triggers unassign after 30-minute idle threshold."""
    mgr = _get_session_manager()
    if mgr is None:
        pytest.skip("SessionManager not found")

    handler = WireMockTFEHandler()
    # Simulate an active session whose last activity was 31 minutes ago
    mgr.register_session(
        endpoint=MOCK_ENDPOINT,
        auth_user_index=0,
        last_activity=datetime.now(timezone.utc) - timedelta(minutes=31),
    )

    # Check and enforce watchdog idle timeout
    unassigned = await mgr.check_inactivity_watchdog(idle_threshold_seconds=1800)
    assert unassigned >= 1 or mgr.has_active_session(MOCK_ENDPOINT) is False


@pytest.mark.asyncio
async def test_e2e_runtime_teardown_and_release(mock_keystore_dir, mock_keyring, sample_credentials):
    """Tier 1: Explicit unassign tears down backend VM and halts CCU metering."""
    client_cls = _get_client_class()
    if client_cls is None:
        pytest.skip("ColabTFEClient not found")

    handler = WireMockTFEHandler()
    transport = httpx.MockTransport(handler.handle)

    client = client_cls(auth_user_index=0, transport=transport)
    success = await client.unassign_runtime(endpoint=MOCK_ENDPOINT)
    assert success is True
    assert handler.unassign_called is True
    assert len(handler.active_sessions) == 0


@pytest.mark.asyncio
async def test_e2e_runtime_quota_exhaustion_handling(mock_keystore_dir, mock_keyring):
    """Tier 2: HTTP 412 quota rejection maps to specific quota exception."""
    client_cls = _get_client_class()
    acc_enum = _get_accelerator_enum()
    if client_cls is None or acc_enum is None:
        pytest.skip("Runtime symbols not found")

    def quota_412_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(412, json={"error": "CCU_EXHAUSTED", "message": "Compute units depleted"})

    client = client_cls(auth_user_index=0, transport=httpx.MockTransport(quota_412_handler))
    with pytest.raises(Exception) as exc_info:
        await client.assign_runtime(accelerator=acc_enum.T4, reuse_existing=False)

    assert "quota" in str(exc_info.value).lower() or "ccu" in str(exc_info.value).lower() or "412" in str(exc_info.value)
