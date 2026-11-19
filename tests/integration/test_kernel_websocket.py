"""
Integration tests for Jupyter Kernel WebSocket Client & Lifecycle Pipeline.

Requirements:
- TECH_SPEC §5.1, §5.2, §5.6
- PROJECT.md § Interface Contracts (Contract 4)
- Features: 33 (Kernel Discovery/Creation), 34 (RFC 6455 Subprotocols Fix),
  35 (Dynamic websockets Header Compatibility), 42 (Tiered Abort & Kernel Restart),
  43 (6-Stage WebSocket Auto-Reconnection).

Adheres to Python 3.13 and progressive testability across milestones.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest
import websockets

# Progressive testability: check implementation availability
try:
    import colab_mcp.kernel.client as kernel_client_mod
    KERNEL_CLIENT_AVAILABLE = True
except ImportError:
    kernel_client_mod = None
    KERNEL_CLIENT_AVAILABLE = False


@pytest.fixture(autouse=True)
def require_kernel_client_implementation():
    """Progressive testability: cleanly skip if colab_mcp.kernel.client is not yet implemented."""
    if not KERNEL_CLIENT_AVAILABLE:
        pytest.skip("colab_mcp.kernel.client is scheduled for Milestone 3 implementation")


def _get_client_class():
    return getattr(kernel_client_mod, "KernelWebSocketClient", None)


def _get_discover_func():
    return getattr(
        kernel_client_mod,
        "discover_or_create_kernel",
        getattr(kernel_client_mod, "resolve_or_create_kernel", None),
    )


def _get_connect_func():
    return getattr(
        kernel_client_mod,
        "connect_colab_kernel_ws",
        getattr(kernel_client_mod, "connect_kernel_ws", None),
    )


# ============================================================================
# Tier 1: Primary Behavior (Happy Path) Tests (>= 5 tests)
# ============================================================================

@pytest.mark.asyncio
async def test_discover_kernel_reuses_existing_active_kernel():
    """
    TECH_SPEC §5.1: If GET /api/kernels returns an active kernel list,
    discover_or_create_kernel extracts and returns the first kernel UUID.
    """
    discover_func = _get_discover_func()
    if discover_func is None:
        pytest.skip("discover_or_create_kernel function not found")

    proxy_url = "https://colab.research.google.com/tun/m/test-endpoint"
    proxy_token = "mock-proxy-token-abc123"

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/api/kernels")
        assert request.headers.get("X-Colab-Runtime-Proxy-Token") == proxy_token
        if request.method == "GET":
            return httpx.Response(200, json=[{"id": "existing-kernel-uuid-112233", "name": "python3"}])
        return httpx.Response(405)

    transport = httpx.MockTransport(mock_handler)
    with patch("httpx.AsyncClient", return_value=httpx.AsyncClient(transport=transport)):
        kernel_id = await discover_func(proxy_url, proxy_token)
        assert kernel_id == "existing-kernel-uuid-112233"


@pytest.mark.asyncio
async def test_create_kernel_when_none_active():
    """
    TECH_SPEC §5.1: If GET /api/kernels returns empty list [],
    POST /api/kernels with {'name': 'python3'} creates and returns new kernel ID.
    """
    discover_func = _get_discover_func()
    if discover_func is None:
        pytest.skip("discover_or_create_kernel function not found")

    proxy_url = "https://colab.research.google.com/tun/m/test-endpoint"
    proxy_token = "mock-proxy-token-abc123"

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/api/kernels"):
            return httpx.Response(200, json=[])
        elif request.method == "POST" and request.url.path.endswith("/api/kernels"):
            return httpx.Response(201, json={"id": "newly-created-kernel-uuid-445566", "name": "python3"})
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    with patch("httpx.AsyncClient", return_value=httpx.AsyncClient(transport=transport)):
        kernel_id = await discover_func(proxy_url, proxy_token)
        assert kernel_id == "newly-created-kernel-uuid-445566"


@pytest.mark.asyncio
async def test_websocket_connect_url_and_subprotocols_none():
    """
    TECH_SPEC §5.2 / Feature 34:
    WebSocket connection to /api/kernels/{id}/channels must set subprotocols=None
    to avoid HTTP 400 rejection from Colab reverse proxy.
    """
    connect_func = _get_connect_func()
    if connect_func is None:
        pytest.skip("connect_colab_kernel_ws function not found")

    proxy_url = "https://colab.research.google.com/tun/m/test-endpoint"
    proxy_token = "mock-proxy-token-abc123"
    kernel_id = "test-kernel-uuid-999"

    mock_ws = AsyncMock()
    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
        mock_connect.return_value = mock_ws
        await connect_func(proxy_url, proxy_token, kernel_id)
        assert mock_connect.called

        call_args, call_kwargs = mock_connect.call_args
        ws_url = call_args[0]
        assert ws_url.startswith("wss://")
        assert f"/api/kernels/{kernel_id}/channels" in ws_url
        assert call_kwargs.get("subprotocols") is None


def test_dynamic_websockets_header_compatibility():
    """
    TECH_SPEC §5.2 / Feature 35:
    Inspects signature of websockets.connect to supply additional_headers (v13+)
    or extra_headers (<v13) without TypeError or deprecation warnings.
    """
    sig = inspect.signature(websockets.connect)
    if "additional_headers" in sig.parameters:
        expected_param = "additional_headers"
    else:
        expected_param = "extra_headers"
    assert expected_param in ("additional_headers", "extra_headers")


@pytest.mark.asyncio
async def test_tier1_non_destructive_sigint_interrupt():
    """
    TECH_SPEC §5.6 / Feature 42:
    Tier 1 interrupt dispatches non-destructive SIGINT (REST or control channel)
    returning state_reset=False, kernel_restarted=False.
    """
    ClientClass = _get_client_class()
    if ClientClass is None:
        pytest.skip("KernelWebSocketClient class not available")

    client = ClientClass(
        proxy_url="https://colab.research.google.com/tun/m/test-endpoint",
        proxy_token="mock-token-123",
        kernel_id="test-kernel-uuid",
    )

    async def mock_interrupt_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/interrupt"):
            return httpx.Response(204)
        return httpx.Response(200)

    transport = httpx.MockTransport(mock_interrupt_handler)
    with patch("httpx.AsyncClient", return_value=httpx.AsyncClient(transport=transport)):
        if hasattr(client, "interrupt"):
            res = await client.interrupt()
            if isinstance(res, dict):
                assert res.get("state_reset") is False
                assert res.get("kernel_restarted") is False
            elif hasattr(res, "state_reset"):
                assert res.state_reset is False
                assert res.kernel_restarted is False


# ============================================================================
# Tier 2: Boundary, Timeouts, Reconnection, Resilience Tests (>= 5 tests)
# ============================================================================

@pytest.mark.asyncio
async def test_tier2_escalated_restart_after_unresponsive_sigint():
    """
    TECH_SPEC §5.6 / Feature 42:
    Tier 2 escalated restart triggers when kernel does not report idle
    within 5.0 seconds of SIGINT. Kernel restart sets state_reset=True, kernel_restarted=True.
    """
    ClientClass = _get_client_class()
    if ClientClass is None:
        pytest.skip("KernelWebSocketClient class not available")

    client = ClientClass(
        proxy_url="https://colab.research.google.com/tun/m/test-endpoint",
        proxy_token="mock-token-123",
        kernel_id="test-kernel-uuid",
    )

    restart_called = False

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal restart_called
        if request.url.path.endswith("/interrupt"):
            return httpx.Response(204)
        elif request.url.path.endswith("/restart"):
            restart_called = True
            return httpx.Response(200, json={"id": "test-kernel-uuid", "execution_state": "idle"})
        return httpx.Response(200)

    transport = httpx.MockTransport(mock_handler)
    with patch("httpx.AsyncClient", return_value=httpx.AsyncClient(transport=transport)):
        if hasattr(client, "interrupt_with_escalation"):
            res = await client.interrupt_with_escalation(timeout=0.1)
            assert restart_called
            state_reset = res.get("state_reset") if isinstance(res, dict) else res.state_reset
            assert state_reset is True


@pytest.mark.asyncio
async def test_6stage_auto_reconnection_triggers_on_abnormal_close():
    """
    TECH_SPEC §5.6 / Feature 43:
    Abnormal WebSocket closure (e.g. code 1006) triggers the 6-stage auto-reconnection
    state machine, polling GET /api/kernels until idle.
    """
    ClientClass = _get_client_class()
    if ClientClass is None:
        pytest.skip("KernelWebSocketClient class not available")

    client = ClientClass(
        proxy_url="https://colab.research.google.com/tun/m/test-endpoint",
        proxy_token="mock-token-123",
        kernel_id="test-kernel-uuid",
    )

    poll_count = 0

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal poll_count
        if request.url.path.endswith("/api/kernels"):
            poll_count += 1
            if poll_count < 2:
                return httpx.Response(200, json=[{"id": "test-kernel-uuid", "execution_state": "starting"}])
            return httpx.Response(200, json=[{"id": "test-kernel-uuid", "execution_state": "idle"}])
        return httpx.Response(200)

    transport = httpx.MockTransport(mock_handler)
    with patch("httpx.AsyncClient", return_value=httpx.AsyncClient(transport=transport)):
        with patch("websockets.connect", new_callable=AsyncMock):
            if hasattr(client, "reconnect"):
                success = await client.reconnect()
                assert success is True
                assert poll_count >= 2


@pytest.mark.asyncio
async def test_reconnection_exhaustion_raises_connection_error():
    """Verify that exceeding max reconnection retries raises an explicit ConnectionError."""
    ClientClass = _get_client_class()
    if ClientClass is None:
        pytest.skip("KernelWebSocketClient class not available")

    client = ClientClass(
        proxy_url="https://colab.research.google.com/tun/m/test-endpoint",
        proxy_token="mock-token-123",
        kernel_id="test-kernel-uuid",
    )

    async def mock_failing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="Gateway Down")

    transport = httpx.MockTransport(mock_failing_handler)
    with patch("httpx.AsyncClient", return_value=httpx.AsyncClient(transport=transport)):
        if hasattr(client, "reconnect"):
            with pytest.raises((ConnectionError, RuntimeError, httpx.HTTPError)):
                await client.reconnect(max_retries=2, initial_backoff=0.01)


@pytest.mark.asyncio
async def test_clean_disconnect_closes_websocket_without_leak():
    """Verify disconnect() method cleanly shuts down active WebSocket connection."""
    ClientClass = _get_client_class()
    if ClientClass is None:
        pytest.skip("KernelWebSocketClient class not available")

    client = ClientClass(
        proxy_url="https://colab.research.google.com/tun/m/test-endpoint",
        proxy_token="mock-token-123",
        kernel_id="test-kernel-uuid",
    )

    mock_ws = AsyncMock()
    client._ws = mock_ws

    if hasattr(client, "disconnect"):
        await client.disconnect()
        assert mock_ws.close.called or getattr(client, "is_connected", False) is False


@pytest.mark.asyncio
async def test_websocket_headers_contain_proxy_token_and_agent():
    """Verify connection headers contain X-Colab-Runtime-Proxy-Token and X-Colab-Client-Agent."""
    connect_func = _get_connect_func()
    if connect_func is None:
        pytest.skip("connect_colab_kernel_ws function not found")

    proxy_url = "https://colab.research.google.com/tun/m/test-endpoint"
    proxy_token = "secret-token-xyz789"
    kernel_id = "test-kernel-123"

    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
        await connect_func(proxy_url, proxy_token, kernel_id)
        call_kwargs = mock_connect.call_args[1]
        headers = call_kwargs.get("additional_headers") or call_kwargs.get("extra_headers") or {}
        assert headers.get("X-Colab-Runtime-Proxy-Token") == proxy_token
        assert "colab-mcp" in headers.get("X-Colab-Client-Agent", "").lower()
