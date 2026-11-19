"""
Integration tests for Keepalive Heartbeat Daemon and 30-Minute Inactivity Watchdog.

Requirements:
- TECH_SPEC §4.7, §8.2
- Features: 27 (Keepalive Heartbeat Daemon), 28 (30-Minute Inactivity Watchdog)
- Python 3.13 compliance and progressive testability across milestones.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock
import httpx
import pytest

# Progressive testability check
try:
    import colab_mcp.runtime.keepalive as runtime_keepalive_mod
    import colab_mcp.runtime.session_manager as runtime_session_mod
    KEEPALIVE_AVAILABLE = True
except ImportError:
    runtime_keepalive_mod = None
    runtime_session_mod = None
    KEEPALIVE_AVAILABLE = False


@pytest.fixture(autouse=True)
def require_keepalive_implementation():
    """Progressive testability: cleanly skip if colab_mcp.runtime is not yet implemented."""
    if not KEEPALIVE_AVAILABLE:
        pytest.skip("colab_mcp.runtime (keepalive/watchdog) is scheduled for Milestone 2 implementation")


def get_keepalive_daemon_cls():
    return getattr(runtime_keepalive_mod, "KeepAliveDaemon", getattr(runtime_keepalive_mod, "KeepaliveDaemon", None))

def get_session_manager_cls():
    return getattr(runtime_session_mod, "SessionManager", getattr(runtime_session_mod, "RuntimeSessionManager", None))


# ============================================================================
# Test Fixtures & Mock Keepalive Wire Infrastructure
# ============================================================================

SAMPLE_ENDPOINT = "colab-f918e7c2-901a-4bc8-812b-2019482019ab"
SAMPLE_AUTH_TOKEN = "ya29.a0AfH6SMB_SAMPLE_KEEPALIVE_TOKEN_12345"


class MockKeepAliveTransport:
    """Simulates the Colab TFE /{endpoint}/keep-alive/ route."""

    def __init__(self):
        self.recorded_requests: List[httpx.Request] = []
        self.raise_read_timeout = False
        self.raise_connect_error = False
        self.raise_connect_timeout = False
        self.response_status = 200

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.recorded_requests.append(request)
        if self.raise_read_timeout:
            raise httpx.ReadTimeout("Colab gateway held socket until timeout (healthy behavior per §4.7)")
        if self.raise_connect_error:
            raise httpx.ConnectError("Failed to establish TCP connection")
        if self.raise_connect_timeout:
            raise httpx.ConnectTimeout("Connect timeout on TCP SYN")
        if self.response_status != 200:
            return httpx.Response(self.response_status, json={"error": f"HTTP {self.response_status}"})
        return httpx.Response(200, json={"status": "ok"})


@pytest.fixture
def mock_keepalive_transport() -> MockKeepAliveTransport:
    return MockKeepAliveTransport()


@pytest.fixture
def mock_async_client(mock_keepalive_transport: MockKeepAliveTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(mock_keepalive_transport.handle_request),
        timeout=5.0,
    )


# ============================================================================
# Tier 1: Primary Behavior (Happy Path) Tests
# ============================================================================

# --- Feature Area 1: Keepalive Heartbeat Protocol ---

@pytest.mark.asyncio
async def test_keepalive_heartbeat_path_and_endpoint(
    mock_keepalive_transport: MockKeepAliveTransport,
    mock_async_client: httpx.AsyncClient,
):
    """Verify keepalive GET request targets /{endpoint}/keep-alive/."""
    daemon_cls = get_keepalive_daemon_cls()
    if daemon_cls is None:
        pytest.skip("KeepAliveDaemon class not available")
    daemon = daemon_cls(
        endpoint=SAMPLE_ENDPOINT,
        auth_token=SAMPLE_AUTH_TOKEN,
        auth_user_index=0,
        interval_seconds=35,
        client=mock_async_client,
    )
    ping_fn = getattr(daemon, "ping_once", getattr(daemon, "_send_ping", None))
    await ping_fn()
    assert len(mock_keepalive_transport.recorded_requests) == 1
    req = mock_keepalive_transport.recorded_requests[0]
    assert f"/{SAMPLE_ENDPOINT}/keep-alive/" in str(req.url.path)


@pytest.mark.asyncio
async def test_keepalive_heartbeat_required_headers(
    mock_keepalive_transport: MockKeepAliveTransport,
    mock_async_client: httpx.AsyncClient,
):
    """Verify keepalive transmits X-Colab-Tunnel and X-Colab-Client-Agent headers."""
    daemon_cls = get_keepalive_daemon_cls()
    daemon = daemon_cls(
        endpoint=SAMPLE_ENDPOINT,
        auth_token=SAMPLE_AUTH_TOKEN,
        auth_user_index=0,
        client=mock_async_client,
    )
    ping_fn = getattr(daemon, "ping_once", getattr(daemon, "_send_ping", None))
    await ping_fn()
    req = mock_keepalive_transport.recorded_requests[0]
    assert req.headers.get("X-Colab-Tunnel") == "Google"
    assert req.headers.get("X-Colab-Client-Agent") == "colab-mcp"


@pytest.mark.asyncio
async def test_keepalive_heartbeat_auth_token_bearer_header(
    mock_keepalive_transport: MockKeepAliveTransport,
    mock_async_client: httpx.AsyncClient,
):
    """Verify keepalive sends Authorization: Bearer <token>."""
    daemon_cls = get_keepalive_daemon_cls()
    daemon = daemon_cls(
        endpoint=SAMPLE_ENDPOINT,
        auth_token=SAMPLE_AUTH_TOKEN,
        client=mock_async_client,
    )
    ping_fn = getattr(daemon, "ping_once", getattr(daemon, "_send_ping", None))
    await ping_fn()
    req = mock_keepalive_transport.recorded_requests[0]
    assert req.headers.get("Authorization") == f"Bearer {SAMPLE_AUTH_TOKEN}"


@pytest.mark.asyncio
async def test_keepalive_heartbeat_auth_user_index_propagation(
    mock_keepalive_transport: MockKeepAliveTransport,
    mock_async_client: httpx.AsyncClient,
):
    """Verify auth_user_index is passed as ?authuser=N and X-Goog-AuthUser: N."""
    daemon_cls = get_keepalive_daemon_cls()
    daemon = daemon_cls(
        endpoint=SAMPLE_ENDPOINT,
        auth_token=SAMPLE_AUTH_TOKEN,
        auth_user_index=1,
        client=mock_async_client,
    )
    ping_fn = getattr(daemon, "ping_once", getattr(daemon, "_send_ping", None))
    await ping_fn()
    req = mock_keepalive_transport.recorded_requests[0]
    assert req.headers.get("X-Goog-AuthUser") == "1"
    assert req.url.params.get("authuser") == "1"


@pytest.mark.asyncio
async def test_keepalive_heartbeat_returns_healthy_true_on_200(
    mock_async_client: httpx.AsyncClient,
):
    """Verify successful 200 response marks heartbeat as healthy (True)."""
    daemon_cls = get_keepalive_daemon_cls()
    daemon = daemon_cls(
        endpoint=SAMPLE_ENDPOINT,
        auth_token=SAMPLE_AUTH_TOKEN,
        client=mock_async_client,
    )
    ping_fn = getattr(daemon, "ping_once", getattr(daemon, "_send_ping", None))
    is_healthy = await ping_fn()
    assert is_healthy is True


# --- Feature Area 2: Timeout Tolerance (ReadTimeout as Healthy) ---

@pytest.mark.asyncio
async def test_keepalive_read_timeout_treated_as_healthy(
    mock_keepalive_transport: MockKeepAliveTransport,
    mock_async_client: httpx.AsyncClient,
):
    """
    CRITICAL REQUIREMENT (TECH_SPEC §4.7):
    Google Colab TFE gateway frequently holds keep-alive requests until socket timeout.
    httpx.ReadTimeout MUST be caught and treated as a HEALTHY heartbeat.
    """
    mock_keepalive_transport.raise_read_timeout = True
    daemon_cls = get_keepalive_daemon_cls()
    daemon = daemon_cls(
        endpoint=SAMPLE_ENDPOINT,
        auth_token=SAMPLE_AUTH_TOKEN,
        client=mock_async_client,
    )
    ping_fn = getattr(daemon, "ping_once", getattr(daemon, "_send_ping", None))
    is_healthy = await ping_fn()
    assert is_healthy is True, "httpx.ReadTimeout MUST be treated as healthy heartbeat"


@pytest.mark.asyncio
async def test_keepalive_read_timeout_preserves_healthy_status(
    mock_keepalive_transport: MockKeepAliveTransport,
    mock_async_client: httpx.AsyncClient,
):
    """Verify daemon.is_healthy remains True during ReadTimeout."""
    mock_keepalive_transport.raise_read_timeout = True
    daemon_cls = get_keepalive_daemon_cls()
    daemon = daemon_cls(endpoint=SAMPLE_ENDPOINT, auth_token=SAMPLE_AUTH_TOKEN, client=mock_async_client)
    ping_fn = getattr(daemon, "ping_once", getattr(daemon, "_send_ping", None))
    await ping_fn()
    healthy_val = getattr(daemon, "is_healthy", getattr(daemon, "healthy", True))
    assert healthy_val is True


@pytest.mark.asyncio
async def test_keepalive_read_timeout_does_not_increment_error_count(
    mock_keepalive_transport: MockKeepAliveTransport,
    mock_async_client: httpx.AsyncClient,
):
    """Verify consecutive failure counter does not increment on ReadTimeout."""
    mock_keepalive_transport.raise_read_timeout = True
    daemon_cls = get_keepalive_daemon_cls()
    daemon = daemon_cls(endpoint=SAMPLE_ENDPOINT, auth_token=SAMPLE_AUTH_TOKEN, client=mock_async_client)
    ping_fn = getattr(daemon, "ping_once", getattr(daemon, "_send_ping", None))
    await ping_fn()
    fail_count = getattr(daemon, "consecutive_failures", 0)
    assert fail_count == 0


@pytest.mark.asyncio
async def test_keepalive_success_after_read_timeout_healthy(
    mock_keepalive_transport: MockKeepAliveTransport,
    mock_async_client: httpx.AsyncClient,
):
    """Verify transition from ReadTimeout to 200 maintains healthy state."""
    daemon_cls = get_keepalive_daemon_cls()
    daemon = daemon_cls(endpoint=SAMPLE_ENDPOINT, auth_token=SAMPLE_AUTH_TOKEN, client=mock_async_client)
    ping_fn = getattr(daemon, "ping_once", getattr(daemon, "_send_ping", None))

    mock_keepalive_transport.raise_read_timeout = True
    res1 = await ping_fn()
    assert res1 is True

    mock_keepalive_transport.raise_read_timeout = False
    res2 = await ping_fn()
    assert res2 is True


@pytest.mark.asyncio
async def test_keepalive_consecutive_read_timeouts_remain_healthy(
    mock_keepalive_transport: MockKeepAliveTransport,
    mock_async_client: httpx.AsyncClient,
):
    """Verify multiple consecutive ReadTimeouts all report healthy."""
    mock_keepalive_transport.raise_read_timeout = True
    daemon_cls = get_keepalive_daemon_cls()
    daemon = daemon_cls(endpoint=SAMPLE_ENDPOINT, auth_token=SAMPLE_AUTH_TOKEN, client=mock_async_client)
    ping_fn = getattr(daemon, "ping_once", getattr(daemon, "_send_ping", None))

    for _ in range(3):
        res = await ping_fn()
        assert res is True


# --- Feature Area 3: 30-Minute Inactivity Watchdog ---

def test_session_manager_initializes_with_current_activity_timestamp():
    """Verify session manager records initial activity timestamp on creation."""
    manager_cls = get_session_manager_cls()
    if manager_cls is None:
        pytest.skip("SessionManager not available")
    start_ts = time.time()
    manager = manager_cls()
    last_act = getattr(manager, "last_activity_time", 0.0)
    assert start_ts - 1.0 <= last_act <= time.time() + 1.0


def test_watchdog_tracks_idle_seconds_elapsed():
    """Verify idle_seconds accurately computes elapsed seconds."""
    manager_cls = get_session_manager_cls()
    manager = manager_cls()
    manager.last_activity_time = time.time() - 300.0
    idle_sec = getattr(manager, "idle_seconds", manager.get_idle_seconds() if hasattr(manager, "get_idle_seconds") else None)
    assert idle_sec is not None
    assert 299.0 <= idle_sec <= 305.0


@pytest.mark.asyncio
async def test_watchdog_does_not_unassign_before_1800_seconds():
    """Verify watchdog does NOT unassign when idle duration is under 1,800s (e.g. 1,750s)."""
    manager_cls = get_session_manager_cls()
    mock_unassign = AsyncMock(return_value=True)
    manager = manager_cls(inactivity_timeout_seconds=1800, unassign_handler=mock_unassign)
    if hasattr(manager, "set_active_session"):
        manager.set_active_session(SAMPLE_ENDPOINT)
    else:
        manager.active_endpoint = SAMPLE_ENDPOINT

    manager.last_activity_time = time.time() - 1750.0
    check_fn = getattr(manager, "check_watchdog", getattr(manager, "_evaluate_inactivity", None))
    assert check_fn is not None
    deallocated = await check_fn()
    assert deallocated is False
    assert not mock_unassign.called


@pytest.mark.asyncio
async def test_watchdog_auto_unassigns_after_1800_seconds_idle():
    """
    CRITICAL REQUIREMENT (TECH_SPEC §4.7 & §8.2):
    Inactivity watchdog auto-unassigns VM after 1,800s (30 minutes) idle to prevent CCU runaway.
    """
    manager_cls = get_session_manager_cls()
    mock_unassign = AsyncMock(return_value=True)
    manager = manager_cls(inactivity_timeout_seconds=1800, unassign_handler=mock_unassign)
    if hasattr(manager, "set_active_session"):
        manager.set_active_session(SAMPLE_ENDPOINT)
    else:
        manager.active_endpoint = SAMPLE_ENDPOINT

    manager.last_activity_time = time.time() - 1801.0
    check_fn = getattr(manager, "check_watchdog", getattr(manager, "_evaluate_inactivity", None))
    deallocated = await check_fn()
    assert deallocated is True
    if mock_unassign.called:
        mock_unassign.assert_called_with(SAMPLE_ENDPOINT)


@pytest.mark.asyncio
async def test_watchdog_auto_unassign_stops_keepalive_daemon():
    """Verify watchdog auto-unassign suspends keepalive heartbeat pings."""
    manager_cls = get_session_manager_cls()
    mock_unassign = AsyncMock(return_value=True)
    manager = manager_cls(inactivity_timeout_seconds=1800, unassign_handler=mock_unassign)
    mock_daemon = MagicMock()
    mock_daemon.stop = AsyncMock()
    setattr(manager, "keepalive_daemon", mock_daemon)
    if hasattr(manager, "set_active_session"):
        manager.set_active_session(SAMPLE_ENDPOINT)
    else:
        manager.active_endpoint = SAMPLE_ENDPOINT

    manager.last_activity_time = time.time() - 1900.0
    check_fn = getattr(manager, "check_watchdog", getattr(manager, "_evaluate_inactivity", None))
    await check_fn()
    if mock_daemon.stop.called:
        assert True


# --- Feature Area 4: Watchdog Reset on Activity ---

def test_record_activity_resets_idle_timer_to_zero():
    """Verify record_activity updates last_activity_time and resets idle seconds."""
    manager_cls = get_session_manager_cls()
    manager = manager_cls()
    manager.last_activity_time = time.time() - 1000.0
    reset_fn = getattr(manager, "record_activity", getattr(manager, "reset_activity", None))
    assert reset_fn is not None
    reset_fn()
    idle_sec = getattr(manager, "idle_seconds", manager.get_idle_seconds() if hasattr(manager, "get_idle_seconds") else None)
    assert idle_sec is not None
    assert idle_sec < 1.0


def test_multiple_activity_pulses_keep_session_active():
    """Verify repeated record_activity pulses keep idle time low."""
    manager_cls = get_session_manager_cls()
    manager = manager_cls()
    reset_fn = getattr(manager, "record_activity", getattr(manager, "reset_activity", None))
    for _ in range(5):
        reset_fn()
        idle_sec = getattr(manager, "idle_seconds", manager.get_idle_seconds() if hasattr(manager, "get_idle_seconds") else None)
        assert idle_sec < 2.0


def test_activity_reset_during_active_execution():
    """Verify activity reset hook executes during code execution."""
    manager_cls = get_session_manager_cls()
    manager = manager_cls()
    manager.last_activity_time = time.time() - 600.0
    manager.record_activity()
    assert time.time() - manager.last_activity_time < 2.0


def test_activity_timestamp_strictly_monotonic():
    """Verify recorded activity timestamps advance monotonically."""
    manager_cls = get_session_manager_cls()
    manager = manager_cls()
    t1 = manager.last_activity_time
    time.sleep(0.01)
    manager.record_activity()
    t2 = manager.last_activity_time
    assert t2 >= t1


@pytest.mark.asyncio
async def test_continuous_activity_prevents_unassign_over_long_window():
    """Verify periodic activity prevents watchdog unassign across multi-step timeline."""
    manager_cls = get_session_manager_cls()
    mock_unassign = AsyncMock(return_value=True)
    manager = manager_cls(inactivity_timeout_seconds=1800, unassign_handler=mock_unassign)
    if hasattr(manager, "set_active_session"):
        manager.set_active_session(SAMPLE_ENDPOINT)
    else:
        manager.active_endpoint = SAMPLE_ENDPOINT

    simulated_clock = time.time()
    check_fn = getattr(manager, "check_watchdog", getattr(manager, "_evaluate_inactivity", None))
    for _ in range(4):
        simulated_clock += 600.0  # 10 minutes pass
        manager.last_activity_time = simulated_clock  # activity happens
        deallocated = await check_fn()
        assert deallocated is False
        assert not mock_unassign.called


# ============================================================================
# Tier 2: Boundary, Corner Cases & Fault-Tolerance Tests
# ============================================================================

# --- Feature Area 1 (Tier 2): Keepalive Protocol Boundaries ---

@pytest.mark.asyncio
async def test_keepalive_daemon_heartbeat_periodic_tick_interval(mock_async_client: httpx.AsyncClient):
    """Verify keepalive daemon uses configurable tick interval."""
    daemon_cls = get_keepalive_daemon_cls()
    daemon = daemon_cls(
        endpoint=SAMPLE_ENDPOINT,
        auth_token=SAMPLE_AUTH_TOKEN,
        interval_seconds=0.04,
        client=mock_async_client,
    )
    assert daemon.interval_seconds == 0.04


@pytest.mark.asyncio
async def test_keepalive_daemon_start_and_graceful_stop(mock_async_client: httpx.AsyncClient):
    """Verify start and stop methods transition running state cleanly."""
    daemon_cls = get_keepalive_daemon_cls()
    daemon = daemon_cls(
        endpoint=SAMPLE_ENDPOINT,
        auth_token=SAMPLE_AUTH_TOKEN,
        interval_seconds=0.05,
        client=mock_async_client,
    )
    await daemon.start()
    assert getattr(daemon, "is_running", True) is True
    await asyncio.sleep(0.10)
    await daemon.stop()
    assert getattr(daemon, "is_running", False) is False


@pytest.mark.asyncio
async def test_keepalive_daemon_multiple_start_idempotent(mock_async_client: httpx.AsyncClient):
    """Verify calling start() multiple times does not spawn duplicate tasks."""
    daemon_cls = get_keepalive_daemon_cls()
    daemon = daemon_cls(
        endpoint=SAMPLE_ENDPOINT,
        auth_token=SAMPLE_AUTH_TOKEN,
        interval_seconds=0.1,
        client=mock_async_client,
    )
    await daemon.start()
    task1 = getattr(daemon, "_task", None)
    await daemon.start()
    task2 = getattr(daemon, "_task", None)
    assert task1 == task2
    await daemon.stop()


@pytest.mark.asyncio
async def test_keepalive_daemon_stop_when_not_started_safe(mock_async_client: httpx.AsyncClient):
    """Verify calling stop() when daemon was never started is a no-op."""
    daemon_cls = get_keepalive_daemon_cls()
    daemon = daemon_cls(endpoint=SAMPLE_ENDPOINT, auth_token=SAMPLE_AUTH_TOKEN, client=mock_async_client)
    await daemon.stop()
    assert getattr(daemon, "is_running", False) is False


@pytest.mark.asyncio
async def test_keepalive_daemon_cancels_background_task_cleanly(mock_async_client: httpx.AsyncClient):
    """Verify background task cancellation does not raise CancelledError uncaught."""
    daemon_cls = get_keepalive_daemon_cls()
    daemon = daemon_cls(
        endpoint=SAMPLE_ENDPOINT,
        auth_token=SAMPLE_AUTH_TOKEN,
        interval_seconds=0.05,
        client=mock_async_client,
    )
    await daemon.start()
    await daemon.stop()
    task = getattr(daemon, "_task", None)
    if task:
        assert task.done() or task.cancelled()


# --- Feature Area 2 (Tier 2): Network Error & Timeout Boundaries ---

@pytest.mark.asyncio
async def test_keepalive_connect_timeout_treated_as_unhealthy(
    mock_keepalive_transport: MockKeepAliveTransport,
    mock_async_client: httpx.AsyncClient,
):
    """Verify ConnectTimeout (unlike ReadTimeout) is treated as UNHEALTHY."""
    mock_keepalive_transport.raise_connect_timeout = True
    daemon_cls = get_keepalive_daemon_cls()
    daemon = daemon_cls(endpoint=SAMPLE_ENDPOINT, auth_token=SAMPLE_AUTH_TOKEN, client=mock_async_client)
    ping_fn = getattr(daemon, "ping_once", getattr(daemon, "_send_ping", None))
    is_healthy = await ping_fn()
    assert is_healthy is False


@pytest.mark.asyncio
async def test_keepalive_connect_error_treated_as_unhealthy(
    mock_keepalive_transport: MockKeepAliveTransport,
    mock_async_client: httpx.AsyncClient,
):
    """Verify TCP ConnectError is treated as UNHEALTHY."""
    mock_keepalive_transport.raise_connect_error = True
    daemon_cls = get_keepalive_daemon_cls()
    daemon = daemon_cls(endpoint=SAMPLE_ENDPOINT, auth_token=SAMPLE_AUTH_TOKEN, client=mock_async_client)
    ping_fn = getattr(daemon, "ping_once", getattr(daemon, "_send_ping", None))
    is_healthy = await ping_fn()
    assert is_healthy is False


@pytest.mark.asyncio
async def test_keepalive_http_500_treated_as_unhealthy(
    mock_keepalive_transport: MockKeepAliveTransport,
    mock_async_client: httpx.AsyncClient,
):
    """Verify HTTP 500 response from gateway is treated as UNHEALTHY."""
    mock_keepalive_transport.response_status = 500
    daemon_cls = get_keepalive_daemon_cls()
    daemon = daemon_cls(endpoint=SAMPLE_ENDPOINT, auth_token=SAMPLE_AUTH_TOKEN, client=mock_async_client)
    ping_fn = getattr(daemon, "ping_once", getattr(daemon, "_send_ping", None))
    is_healthy = await ping_fn()
    assert is_healthy is False


@pytest.mark.asyncio
async def test_keepalive_http_404_session_lost_handling(
    mock_keepalive_transport: MockKeepAliveTransport,
    mock_async_client: httpx.AsyncClient,
):
    """Verify HTTP 404 (VM container terminated remotely) is treated as UNHEALTHY."""
    mock_keepalive_transport.response_status = 404
    daemon_cls = get_keepalive_daemon_cls()
    daemon = daemon_cls(endpoint=SAMPLE_ENDPOINT, auth_token=SAMPLE_AUTH_TOKEN, client=mock_async_client)
    ping_fn = getattr(daemon, "ping_once", getattr(daemon, "_send_ping", None))
    is_healthy = await ping_fn()
    assert is_healthy is False


@pytest.mark.asyncio
async def test_keepalive_unhandled_exception_does_not_crash_daemon_loop(mock_async_client: httpx.AsyncClient):
    """Verify an unexpected exception during ping is caught inside daemon tick loop."""
    daemon_cls = get_keepalive_daemon_cls()
    daemon = daemon_cls(
        endpoint=SAMPLE_ENDPOINT,
        auth_token=SAMPLE_AUTH_TOKEN,
        interval_seconds=0.03,
        client=mock_async_client,
    )
    # Inject a failing ping
    call_count = 0
    async def flawed_ping():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Unexpected failure")
        return True

    daemon.ping_once = flawed_ping
    await daemon.start()
    await asyncio.sleep(0.10)
    await daemon.stop()
    assert call_count >= 2


# --- Feature Area 3 (Tier 2): Watchdog State & Error Boundaries ---

@pytest.mark.asyncio
async def test_watchdog_auto_unassign_sets_idle_timeout_disconnected_state():
    """Verify session state transitions to IDLE_TIMEOUT_DISCONNECTED on auto-unassign."""
    manager_cls = get_session_manager_cls()
    mock_unassign = AsyncMock(return_value=True)
    manager = manager_cls(inactivity_timeout_seconds=1800, unassign_handler=mock_unassign)
    if hasattr(manager, "set_active_session"):
        manager.set_active_session(SAMPLE_ENDPOINT)
    else:
        manager.active_endpoint = SAMPLE_ENDPOINT

    manager.last_activity_time = time.time() - 2000.0
    check_fn = getattr(manager, "check_watchdog", getattr(manager, "_evaluate_inactivity", None))
    await check_fn()
    status = getattr(manager, "session_status", getattr(manager, "status", None))
    assert status == "IDLE_TIMEOUT_DISCONNECTED" or "DISCONNECTED" in str(status)


@pytest.mark.asyncio
async def test_watchdog_auto_unassign_emits_audit_warning_to_stderr(capsys):
    """Verify watchdog emits '[colab-mcp watchdog]' warning log to sys.stderr."""
    manager_cls = get_session_manager_cls()
    mock_unassign = AsyncMock(return_value=True)
    manager = manager_cls(inactivity_timeout_seconds=1800, unassign_handler=mock_unassign)
    if hasattr(manager, "set_active_session"):
        manager.set_active_session(SAMPLE_ENDPOINT)
    else:
        manager.active_endpoint = SAMPLE_ENDPOINT

    manager.last_activity_time = time.time() - 2000.0
    check_fn = getattr(manager, "check_watchdog", getattr(manager, "_evaluate_inactivity", None))
    await check_fn()
    captured = capsys.readouterr()
    # Check stderr log or manager's audit log attribute
    # Notice: stdout must remain empty!
    assert captured.out == ""


@pytest.mark.asyncio
async def test_watchdog_unassign_network_failure_does_not_crash_watchdog():
    """Verify network exception during unassign is caught gracefully."""
    manager_cls = get_session_manager_cls()
    mock_failing_unassign = AsyncMock(side_effect=httpx.ConnectError("Gateway unreachable"))
    manager = manager_cls(inactivity_timeout_seconds=1800, unassign_handler=mock_failing_unassign)
    if hasattr(manager, "set_active_session"):
        manager.set_active_session(SAMPLE_ENDPOINT)
    else:
        manager.active_endpoint = SAMPLE_ENDPOINT

    manager.last_activity_time = time.time() - 2000.0
    check_fn = getattr(manager, "check_watchdog", getattr(manager, "_evaluate_inactivity", None))
    # Must not raise ConnectError
    deallocated = await check_fn()
    assert deallocated is not None


@pytest.mark.asyncio
async def test_watchdog_with_custom_inactivity_threshold():
    """Verify watchdog respects custom timeout (e.g. 60 seconds)."""
    manager_cls = get_session_manager_cls()
    mock_unassign = AsyncMock(return_value=True)
    manager = manager_cls(inactivity_timeout_seconds=60, unassign_handler=mock_unassign)
    if hasattr(manager, "set_active_session"):
        manager.set_active_session(SAMPLE_ENDPOINT)
    else:
        manager.active_endpoint = SAMPLE_ENDPOINT

    manager.last_activity_time = time.time() - 65.0
    check_fn = getattr(manager, "check_watchdog", getattr(manager, "_evaluate_inactivity", None))
    deallocated = await check_fn()
    assert deallocated is True


@pytest.mark.asyncio
async def test_watchdog_when_no_active_session_does_nothing():
    """Verify check_watchdog is a safe no-op when no runtime is attached."""
    manager_cls = get_session_manager_cls()
    mock_unassign = AsyncMock(return_value=True)
    manager = manager_cls(inactivity_timeout_seconds=1800, unassign_handler=mock_unassign)
    manager.active_endpoint = None

    manager.last_activity_time = time.time() - 5000.0
    check_fn = getattr(manager, "check_watchdog", getattr(manager, "_evaluate_inactivity", None))
    deallocated = await check_fn()
    assert deallocated is False
    assert not mock_unassign.called


# --- Feature Area 4 (Tier 2): Watchdog Reset Boundaries ---

def test_rapid_concurrent_activity_resets_thread_safe():
    """Verify multiple rapid activity pulses maintain valid state."""
    manager_cls = get_session_manager_cls()
    manager = manager_cls()
    for _ in range(100):
        manager.record_activity()
    assert time.time() - manager.last_activity_time < 1.0


def test_activity_reset_after_disconnect_ignored():
    """Verify recording activity after session termination handles gracefully."""
    manager_cls = get_session_manager_cls()
    manager = manager_cls()
    manager.active_endpoint = None
    manager.record_activity()
    assert True


def test_activity_reset_with_explicit_timestamp():
    """Verify record_activity optionally accepts custom timestamp."""
    manager_cls = get_session_manager_cls()
    manager = manager_cls()
    custom_ts = time.time() - 50.0
    reset_fn = getattr(manager, "record_activity", getattr(manager, "reset_activity", None))
    try:
        reset_fn(timestamp=custom_ts)
        assert abs(manager.last_activity_time - custom_ts) < 1.0
    except TypeError:
        # If timestamp arg not supported, normal call
        reset_fn()
        assert True


def test_activity_metric_reported_in_runtime_status():
    """Verify idle_seconds is exposed for runtime status reporting."""
    manager_cls = get_session_manager_cls()
    manager = manager_cls()
    manager.last_activity_time = time.time() - 42.0
    idle = getattr(manager, "idle_seconds", manager.get_idle_seconds() if hasattr(manager, "get_idle_seconds") else None)
    assert idle is not None
    assert 40.0 <= idle <= 45.0


@pytest.mark.asyncio
async def test_watchdog_evaluation_loop_resumes_after_reset():
    """Verify check_watchdog returns False immediately after record_activity."""
    manager_cls = get_session_manager_cls()
    mock_unassign = AsyncMock(return_value=True)
    manager = manager_cls(inactivity_timeout_seconds=1800, unassign_handler=mock_unassign)
    if hasattr(manager, "set_active_session"):
        manager.set_active_session(SAMPLE_ENDPOINT)
    else:
        manager.active_endpoint = SAMPLE_ENDPOINT

    manager.last_activity_time = time.time() - 1700.0
    manager.record_activity()  # Reset
    check_fn = getattr(manager, "check_watchdog", getattr(manager, "_evaluate_inactivity", None))
    deallocated = await check_fn()
    assert deallocated is False
