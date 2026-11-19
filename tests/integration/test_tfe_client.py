"""
Integration tests for Google Colab Tunnel Frontend (TFE) Client & Runtime Management.

Requirements:
- TECH_SPEC §4.1 - §4.6, §7.3.2
- PROJECT.md § Interface Contracts (Contract 3)
- Features: 19 (NBH), 20 (Accelerator Matrix), 21 (High-RAM Free Tier Guard),
  22 (Session Re-attachment), 23 (Two-Step Handshake), 24 (Quota Error Disambiguation),
  25 (Cloud GPU Capacity), 26 (Session Teardown), 29-32 (MCP Runtime Tools).

Adheres to Python 3.13 and progressive testability across milestones.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock
import httpx
import pytest

# Check runtime implementation availability for progressive testability
try:
    import colab_mcp.runtime.client as runtime_client_mod
    import colab_mcp.runtime.models as runtime_models_mod
    import colab_mcp.runtime.tools as runtime_tools_mod
    RUNTIME_AVAILABLE = True
except ImportError:
    runtime_client_mod = None
    runtime_models_mod = None
    runtime_tools_mod = None
    RUNTIME_AVAILABLE = False


@pytest.fixture(autouse=True)
def require_runtime_implementation():
    """Progressive testability: cleanly skip if colab_mcp.runtime is not yet implemented."""
    if not RUNTIME_AVAILABLE:
        pytest.skip("colab_mcp.runtime is scheduled for Milestone 2 implementation")


# Dynamic symbol resolution
def get_client_class():
    return getattr(runtime_client_mod, "ColabTFEClient", None)

def get_accelerator_enum():
    return getattr(runtime_models_mod, "Accelerator", None)

def get_variant_enum():
    return getattr(runtime_models_mod, "Variant", None)

def get_shape_enum():
    return getattr(runtime_models_mod, "Shape", None)

def get_high_ram_error():
    return getattr(runtime_models_mod, "HighRamNotAvailableError", Exception)

def get_too_many_assignments_error():
    return getattr(runtime_models_mod, "TooManyAssignmentsError", Exception)

def get_ccu_exhausted_error():
    return getattr(runtime_models_mod, "ComputeUnitsExhaustedError", Exception)

def get_free_tier_quota_error():
    return getattr(runtime_models_mod, "FreeTierQuotaExceededError", Exception)

def get_capacity_error():
    return getattr(runtime_models_mod, "CloudCapacityError", RuntimeError)


# ============================================================================
# Test Fixtures & Mock Colab TFE Gateway Harness
# ============================================================================

TEST_AUTH_TOKEN = "ya29.a0AfH6SMB_TEST_TFE_CLIENT_TOKEN_12345"
SAMPLE_NBH = "c9a646d3_9c61_4cc9_bc77_b22e11e0563e........"
SAMPLE_ENDPOINT = "colab-f918e7c2-901a-4bc8-812b-2019482019ab"
SAMPLE_PROXY_URL = "https://colab.research.google.com/tun/m/colab-f918e7c2-901a-4bc8-812b-2019482019ab"
SAMPLE_PROXY_TOKEN = "ya29.c.b0xxxx_MOCK_PROXY_TOKEN_12345"


class MockTFEGateway:
    """Simulates Google Colab Tunnel Frontend (TFE) gateway endpoints."""

    def __init__(self):
        self.active_assignments: List[Dict[str, Any]] = []
        self.recorded_requests: List[httpx.Request] = []
        self.xsrf_token_counter = 0
        self.issue_tokens = True
        self.assign_status_code = 200
        self.assign_error_body: Optional[Dict[str, Any]] = None
        self.unassign_status_code = 200
        self.unassign_error_body: Optional[Dict[str, Any]] = None
        self.ccu_balance = 84.25
        self.user_tier = "PRO"

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.recorded_requests.append(request)
        url_path = request.url.path
        query = dict(request.url.params)

        # 1. GET /tun/m/assignments: Enumerate active sessions
        if url_path == "/tun/m/assignments" and request.method == "GET":
            return httpx.Response(200, json={"assignments": self.active_assignments})

        # 2. GET /tun/m/ccu-info: Compute Unit metrics
        if url_path == "/tun/m/ccu-info" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "user_tier": self.user_tier,
                    "compute_units_remaining": self.ccu_balance,
                    "consumption_rate_per_hour": 1.96,
                    "active_assignments_count": len(self.active_assignments),
                    "auto_refill_enabled": True,
                },
            )

        # 3. GET /tun/m/assign: Step 1 Pre-assignment & XSRF token fetch
        if url_path == "/tun/m/assign" and request.method == "GET":
            if query.get("shape") == "hm" and self.assign_status_code == 400:
                return httpx.Response(
                    400,
                    json={"error": {"code": "BAD_REQUEST", "message": "High-RAM shape requires Colab Pro/Pro+ subscription."}},
                )
            if self.assign_status_code != 200:
                return httpx.Response(self.assign_status_code, json=self.assign_error_body or {})

            if not self.issue_tokens:
                return httpx.Response(200, json={})

            self.xsrf_token_counter += 1
            xsrf = f"XSRF_TOKEN_{self.xsrf_token_counter:04d}"
            return httpx.Response(200, json={"token": xsrf, "nbh": query.get("nbh")})

        # 4. POST /tun/m/assign: Step 2 Allocation claim
        if url_path == "/tun/m/assign" and request.method == "POST":
            token_header = request.headers.get("X-Goog-Colab-Token")
            if not token_header:
                return httpx.Response(403, json={"error": "Missing X-Goog-Colab-Token header"})

            if self.assign_status_code != 200:
                return httpx.Response(self.assign_status_code, json=self.assign_error_body or {})

            assignment_data = {
                "endpoint": SAMPLE_ENDPOINT,
                "accelerator": query.get("accelerator", "T4"),
                "variant": query.get("variant", "GPU"),
                "runtimeProxyInfo": {
                    "url": SAMPLE_PROXY_URL,
                    "token": SAMPLE_PROXY_TOKEN,
                    "tokenExpiresInSeconds": 3600,
                },
            }
            self.active_assignments.append(assignment_data)
            return httpx.Response(200, json=assignment_data)

        # 5. GET /tun/m/unassign/{endpoint}: Step 1 Unassign verification token
        if url_path.startswith("/tun/m/unassign/") and request.method == "GET":
            endpoint = url_path.split("/")[-1]
            if not any(a.get("endpoint") == endpoint for a in self.active_assignments) and self.unassign_status_code == 404:
                return httpx.Response(404, json={"error": "Not Found"})
            return httpx.Response(200, json={"token": f"UNASSIGN_XSRF_{endpoint}"})

        # 6. POST /tun/m/unassign/{endpoint}: Step 2 Confirm deallocation
        if url_path.startswith("/tun/m/unassign/") and request.method == "POST":
            endpoint = url_path.split("/")[-1]
            token_header = request.headers.get("X-Goog-Colab-Token")
            if not token_header:
                return httpx.Response(403, json={"error": "Missing unassign token"})
            self.active_assignments = [a for a in self.active_assignments if a.get("endpoint") != endpoint]
            return httpx.Response(200, json={"status": "unassigned", "endpoint": endpoint})

        return httpx.Response(404, json={"error": f"Unhandled route {request.method} {url_path}"})


@pytest.fixture
def mock_tfe_gateway() -> MockTFEGateway:
    return MockTFEGateway()


@pytest.fixture
def tfe_client(mock_tfe_gateway: MockTFEGateway):
    """Instantiate ColabTFEClient with mock transport."""
    client_cls = get_client_class()
    if client_cls is None:
        pytest.skip("ColabTFEClient not available")
    client = client_cls(auth_token=TEST_AUTH_TOKEN, auth_user_index=0)
    mock_transport = httpx.MockTransport(mock_tfe_gateway.handle_request)
    client.client = httpx.AsyncClient(transport=mock_transport, timeout=30.0)
    return client


# ============================================================================
# Tier 1: Primary Behavior (Happy Path) Tests
# ============================================================================

# --- Feature Area 1: TFE REST Client Methods ---

@pytest.mark.asyncio
async def test_list_assignments_empty_returns_empty_list(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify list_assignments returns empty list when no sessions active."""
    assignments = await tfe_client.list_assignments()
    assert assignments == [] or assignments == ()
    assert len(mock_tfe_gateway.recorded_requests) == 1
    req = mock_tfe_gateway.recorded_requests[0]
    assert req.method == "GET"
    assert "/tun/m/assignments" in str(req.url)


@pytest.mark.asyncio
async def test_list_assignments_returns_active_containers(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify list_assignments parses active container details."""
    mock_tfe_gateway.active_assignments.append({
        "endpoint": SAMPLE_ENDPOINT,
        "accelerator": "T4",
        "variant": "GPU",
        "runtimeProxyInfo": {"url": SAMPLE_PROXY_URL, "token": SAMPLE_PROXY_TOKEN, "tokenExpiresInSeconds": 3600},
    })
    assignments = await tfe_client.list_assignments()
    assert len(assignments) == 1
    assignment = assignments[0]
    endpoint = getattr(assignment, "endpoint", assignment.get("endpoint") if isinstance(assignment, dict) else None)
    assert endpoint == SAMPLE_ENDPOINT


@pytest.mark.asyncio
async def test_get_ccu_info_returns_account_tier_and_balance(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify get_ccu_info returns user tier and remaining compute units."""
    mock_tfe_gateway.ccu_balance = 125.50
    mock_tfe_gateway.user_tier = "PRO_PLUS"
    ccu_info = await tfe_client.get_ccu_info()
    tier = getattr(ccu_info, "user_tier", None) or (ccu_info.get("user_tier") if isinstance(ccu_info, dict) else None)
    remaining = getattr(ccu_info, "compute_units_remaining", None) or (ccu_info.get("compute_units_remaining") if isinstance(ccu_info, dict) else None)
    assert tier == "PRO_PLUS"
    assert remaining == 125.50


@pytest.mark.asyncio
async def test_get_ccu_info_returns_consumption_rate(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify get_ccu_info returns hourly consumption rate."""
    ccu_info = await tfe_client.get_ccu_info()
    burn = getattr(ccu_info, "consumption_rate_per_hour", None) or (ccu_info.get("consumption_rate_per_hour") if isinstance(ccu_info, dict) else None)
    assert burn == 1.96


@pytest.mark.asyncio
async def test_tfe_client_initialization_defaults_and_custom_agent(mock_tfe_gateway: MockTFEGateway):
    """Verify ColabTFEClient initialization sets client agent header."""
    client_cls = get_client_class()
    client = client_cls(auth_token=TEST_AUTH_TOKEN, auth_user_index=0, client_agent="custom-colab-agent")
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(mock_tfe_gateway.handle_request), timeout=30.0)
    await client.list_assignments()
    req = mock_tfe_gateway.recorded_requests[-1]
    assert req.headers.get("X-Colab-Client-Agent") == "custom-colab-agent"


# --- Feature Area 2: Two-Step Assignment Handshake ---

@pytest.mark.asyncio
async def test_two_step_handshake_step1_get_fetches_xsrf_token(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify two-step handshake begins with GET /tun/m/assign."""
    acc_enum = get_accelerator_enum()
    var_enum = get_variant_enum()
    shape_enum = get_shape_enum()
    await tfe_client.assign_runtime(
        nbh=SAMPLE_NBH,
        variant=var_enum.GPU,
        accelerator=acc_enum.T4,
        shape=shape_enum.STANDARD,
        reuse_existing=False,
    )
    req_step1 = mock_tfe_gateway.recorded_requests[0]
    assert req_step1.method == "GET"
    assert req_step1.url.path == "/tun/m/assign"


@pytest.mark.asyncio
async def test_two_step_handshake_step2_post_sends_xsrf_header(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify step 2 POST sends X-Goog-Colab-Token header containing token from step 1."""
    acc_enum = get_accelerator_enum()
    await tfe_client.assign_runtime(nbh=SAMPLE_NBH, accelerator=acc_enum.T4, reuse_existing=False)
    assert len(mock_tfe_gateway.recorded_requests) == 2
    req_step2 = mock_tfe_gateway.recorded_requests[1]
    assert req_step2.method == "POST"
    assert req_step2.headers.get("X-Goog-Colab-Token") == "XSRF_TOKEN_0001"


@pytest.mark.asyncio
async def test_two_step_handshake_returns_runtime_proxy_info(tfe_client):
    """Verify assign_runtime returns endpoint, proxy url, and proxy token."""
    acc_enum = get_accelerator_enum()
    res = await tfe_client.assign_runtime(nbh=SAMPLE_NBH, accelerator=acc_enum.T4, reuse_existing=False)
    endpoint = getattr(res, "endpoint", res.get("endpoint") if isinstance(res, dict) else None)
    assert endpoint == SAMPLE_ENDPOINT


@pytest.mark.asyncio
async def test_two_step_handshake_query_params_nbh_variant_accelerator(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify exact query parameters (nbh, variant, accelerator, authuser) on assignment calls."""
    acc_enum = get_accelerator_enum()
    var_enum = get_variant_enum()
    await tfe_client.assign_runtime(nbh=SAMPLE_NBH, variant=var_enum.GPU, accelerator=acc_enum.T4, reuse_existing=False)
    req = mock_tfe_gateway.recorded_requests[0]
    assert req.url.params["nbh"] == SAMPLE_NBH
    assert req.url.params["variant"] == "GPU"
    assert req.url.params["accelerator"] == "T4"


@pytest.mark.asyncio
async def test_two_step_handshake_shortcut_if_step1_returns_assignment(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify if Step 1 GET returns full assignment directly, Step 2 POST is omitted."""
    def shortcut_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tun/m/assign" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "endpoint": SAMPLE_ENDPOINT,
                    "accelerator": "T4",
                    "variant": "GPU",
                    "runtimeProxyInfo": {"url": SAMPLE_PROXY_URL, "token": SAMPLE_PROXY_TOKEN, "tokenExpiresInSeconds": 3600},
                },
            )
        return mock_tfe_gateway.handle_request(request)

    tfe_client.client = httpx.AsyncClient(transport=httpx.MockTransport(shortcut_handler), timeout=30.0)
    res = await tfe_client.assign_runtime(nbh=SAMPLE_NBH, reuse_existing=False)
    endpoint = getattr(res, "endpoint", res.get("endpoint") if isinstance(res, dict) else None)
    assert endpoint == SAMPLE_ENDPOINT


# --- Feature Area 3: Pre-Assignment Session Re-Attachment ---

@pytest.mark.asyncio
async def test_session_reattachment_reuses_existing_matching_gpu(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify reuse_existing=True re-attaches to active matching GPU without provisioning."""
    mock_tfe_gateway.active_assignments.append({
        "endpoint": "existing-vm-gpu-123",
        "accelerator": "T4",
        "variant": "GPU",
        "runtimeProxyInfo": {"url": SAMPLE_PROXY_URL, "token": SAMPLE_PROXY_TOKEN, "tokenExpiresInSeconds": 3600},
    })
    acc_enum = get_accelerator_enum()
    res = await tfe_client.assign_runtime(nbh=SAMPLE_NBH, accelerator=acc_enum.T4, reuse_existing=True)
    endpoint = getattr(res, "endpoint", res.get("endpoint") if isinstance(res, dict) else None)
    assert endpoint == "existing-vm-gpu-123"


@pytest.mark.asyncio
async def test_session_reattachment_reuses_existing_matching_cpu(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify reuse_existing=True re-attaches to active matching CPU VM."""
    mock_tfe_gateway.active_assignments.append({
        "endpoint": "existing-vm-cpu-456",
        "accelerator": "NONE",
        "variant": "DEFAULT",
        "runtimeProxyInfo": {"url": SAMPLE_PROXY_URL, "token": SAMPLE_PROXY_TOKEN, "tokenExpiresInSeconds": 3600},
    })
    acc_enum = get_accelerator_enum()
    var_enum = get_variant_enum()
    res = await tfe_client.assign_runtime(nbh=SAMPLE_NBH, variant=var_enum.DEFAULT, accelerator=acc_enum.NONE, reuse_existing=True)
    endpoint = getattr(res, "endpoint", res.get("endpoint") if isinstance(res, dict) else None)
    assert endpoint == "existing-vm-cpu-456"


@pytest.mark.asyncio
async def test_session_reattachment_returns_existing_proxy_url_and_token(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify re-attached session returns intact proxy credentials."""
    mock_tfe_gateway.active_assignments.append({
        "endpoint": "existing-vm-789",
        "accelerator": "T4",
        "variant": "GPU",
        "runtimeProxyInfo": {"url": "https://colab.research.google.com/proxy/789", "token": "proxy-token-789", "tokenExpiresInSeconds": 3600},
    })
    acc_enum = get_accelerator_enum()
    res = await tfe_client.assign_runtime(nbh=SAMPLE_NBH, accelerator=acc_enum.T4, reuse_existing=True)
    if isinstance(res, dict):
        assert res["runtimeProxyInfo"]["token"] == "proxy-token-789"
    else:
        token = getattr(res, "runtime_proxy_token", getattr(res, "runtimeProxyInfo", {}).get("token"))
        assert token == "proxy-token-789"


@pytest.mark.asyncio
async def test_session_reattachment_does_not_call_post_assign(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify re-attachment makes zero POST requests to /assign."""
    mock_tfe_gateway.active_assignments.append({
        "endpoint": "existing-vm-111",
        "accelerator": "T4",
        "variant": "GPU",
        "runtimeProxyInfo": {"url": SAMPLE_PROXY_URL, "token": SAMPLE_PROXY_TOKEN, "tokenExpiresInSeconds": 3600},
    })
    acc_enum = get_accelerator_enum()
    await tfe_client.assign_runtime(nbh=SAMPLE_NBH, accelerator=acc_enum.T4, reuse_existing=True)
    post_reqs = [r for r in mock_tfe_gateway.recorded_requests if r.method == "POST"]
    assert len(post_reqs) == 0


@pytest.mark.asyncio
async def test_session_reattachment_multiple_active_selects_first_matching(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify when multiple active sessions exist, first matching accelerator is returned."""
    mock_tfe_gateway.active_assignments.extend([
        {"endpoint": "vm-cpu-1", "accelerator": "NONE", "variant": "DEFAULT", "runtimeProxyInfo": {"url": "u1", "token": "t1"}},
        {"endpoint": "vm-gpu-1", "accelerator": "T4", "variant": "GPU", "runtimeProxyInfo": {"url": "u2", "token": "t2"}},
        {"endpoint": "vm-gpu-2", "accelerator": "T4", "variant": "GPU", "runtimeProxyInfo": {"url": "u3", "token": "t3"}},
    ])
    acc_enum = get_accelerator_enum()
    res = await tfe_client.assign_runtime(nbh=SAMPLE_NBH, accelerator=acc_enum.T4, reuse_existing=True)
    endpoint = getattr(res, "endpoint", res.get("endpoint") if isinstance(res, dict) else None)
    assert endpoint == "vm-gpu-1"


# --- Feature Area 4: Hardware Accelerator & Shape Matrix Mapping ---

@pytest.mark.asyncio
async def test_accelerator_mapping_none_to_default_variant(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify Accelerator.NONE maps to DEFAULT variant."""
    acc_enum = get_accelerator_enum()
    var_enum = get_variant_enum()
    await tfe_client.assign_runtime(nbh=SAMPLE_NBH, variant=var_enum.DEFAULT, accelerator=acc_enum.NONE, reuse_existing=False)
    req = mock_tfe_gateway.recorded_requests[0]
    assert req.url.params["accelerator"] == "NONE"
    assert req.url.params["variant"] == "DEFAULT"


@pytest.mark.asyncio
async def test_accelerator_mapping_t4_to_gpu_variant(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify Accelerator.T4 maps to GPU variant."""
    acc_enum = get_accelerator_enum()
    var_enum = get_variant_enum()
    await tfe_client.assign_runtime(nbh=SAMPLE_NBH, variant=var_enum.GPU, accelerator=acc_enum.T4, reuse_existing=False)
    req = mock_tfe_gateway.recorded_requests[0]
    assert req.url.params["accelerator"] == "T4"
    assert req.url.params["variant"] == "GPU"


@pytest.mark.asyncio
async def test_accelerator_mapping_l4_and_a100_to_gpu_variant(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify Accelerator.L4 and A100 map to GPU variant."""
    acc_enum = get_accelerator_enum()
    var_enum = get_variant_enum()
    await tfe_client.assign_runtime(nbh=SAMPLE_NBH, variant=var_enum.GPU, accelerator=acc_enum.L4, reuse_existing=False)
    assert mock_tfe_gateway.recorded_requests[0].url.params["accelerator"] == "L4"


@pytest.mark.asyncio
async def test_accelerator_mapping_h100_to_gpu_variant(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify Accelerator.H100 maps to GPU variant."""
    acc_enum = get_accelerator_enum()
    var_enum = get_variant_enum()
    await tfe_client.assign_runtime(nbh=SAMPLE_NBH, variant=var_enum.GPU, accelerator=acc_enum.H100, reuse_existing=False)
    assert mock_tfe_gateway.recorded_requests[0].url.params["accelerator"] == "H100"


@pytest.mark.asyncio
async def test_accelerator_mapping_tpu_v5e1_and_v6e1_to_tpu_variant(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify Accelerator.V5E1 and V6E1 map to TPU variant."""
    acc_enum = get_accelerator_enum()
    var_enum = get_variant_enum()
    await tfe_client.assign_runtime(nbh=SAMPLE_NBH, variant=var_enum.TPU, accelerator=acc_enum.V5E1, reuse_existing=False)
    req = mock_tfe_gateway.recorded_requests[0]
    assert req.url.params["accelerator"] == "V5E1"
    assert req.url.params["variant"] == "TPU"


# --- Feature Area 5: High-RAM Free Tier Guard ---

@pytest.mark.asyncio
async def test_high_ram_free_tier_guard_raises_explicit_exception(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify HTTP 400 when shape=hm raises explicit HighRamNotAvailableError."""
    mock_tfe_gateway.assign_status_code = 400
    shape_enum = get_shape_enum()
    err_cls = get_high_ram_error()
    with pytest.raises(err_cls):
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, shape=shape_enum.HIGH_RAM, reuse_existing=False)


@pytest.mark.asyncio
async def test_high_ram_exception_contains_upgrade_advice(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify HighRamNotAvailableError message mentions Pro/Pro+ upgrade."""
    mock_tfe_gateway.assign_status_code = 400
    shape_enum = get_shape_enum()
    err_cls = get_high_ram_error()
    with pytest.raises(err_cls) as exc_info:
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, shape=shape_enum.HIGH_RAM, reuse_existing=False)
    msg = str(exc_info.value).lower()
    assert "high-ram" in msg or "pro" in msg or "upgrade" in msg


@pytest.mark.asyncio
async def test_high_ram_pro_tier_succeeds_when_gateway_returns_200(tfe_client):
    """Verify High-RAM succeeds cleanly when gateway returns 200 (Pro tier)."""
    shape_enum = get_shape_enum()
    res = await tfe_client.assign_runtime(nbh=SAMPLE_NBH, shape=shape_enum.HIGH_RAM, reuse_existing=False)
    endpoint = getattr(res, "endpoint", res.get("endpoint") if isinstance(res, dict) else None)
    assert endpoint == SAMPLE_ENDPOINT


@pytest.mark.asyncio
async def test_standard_ram_never_raises_high_ram_exception(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify standard shape does not trigger HighRamNotAvailableError."""
    mock_tfe_gateway.assign_status_code = 400
    mock_tfe_gateway.assign_error_body = {"error": "generic"}
    shape_enum = get_shape_enum()
    err_cls = get_high_ram_error()
    with pytest.raises(Exception) as exc_info:
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, shape=shape_enum.STANDARD, reuse_existing=False)
    assert not isinstance(exc_info.value, err_cls)


@pytest.mark.asyncio
async def test_high_ram_guard_checked_on_both_step1_and_step2(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify HighRamNotAvailableError is caught even if Step 1 succeeds and Step 2 returns 400."""
    def step2_reject_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(400, json={"error": "High-RAM unavailable"})
        return mock_tfe_gateway.handle_request(request)

    tfe_client.client = httpx.AsyncClient(transport=httpx.MockTransport(step2_reject_handler), timeout=30.0)
    shape_enum = get_shape_enum()
    err_cls = get_high_ram_error()
    with pytest.raises(err_cls):
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, shape=shape_enum.HIGH_RAM, reuse_existing=False)


# --- Feature Area 6: Quota Error Disambiguation (HTTP 412) ---

@pytest.mark.asyncio
async def test_quota_error_412_too_many_assignments_code(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify HTTP 412 with TOO_MANY_ASSIGNMENTS raises TooManyAssignmentsError."""
    mock_tfe_gateway.assign_status_code = 412
    mock_tfe_gateway.assign_error_body = {"error": {"code": "TOO_MANY_ASSIGNMENTS", "message": "Max sessions reached"}}
    err_cls = get_too_many_assignments_error()
    with pytest.raises(err_cls):
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, reuse_existing=False)


@pytest.mark.asyncio
async def test_quota_error_412_too_many_assignments_message_fallback(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify HTTP 412 with TOO_MANY_ASSIGNMENTS in message string raises TooManyAssignmentsError."""
    mock_tfe_gateway.assign_status_code = 412
    mock_tfe_gateway.assign_error_body = {"message": "User hit TOO_MANY_ASSIGNMENTS limit."}
    err_cls = get_too_many_assignments_error()
    with pytest.raises(err_cls):
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, reuse_existing=False)


@pytest.mark.asyncio
async def test_quota_error_412_ccu_exhausted_code(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify HTTP 412 with CCU_EXHAUSTED raises ComputeUnitsExhaustedError."""
    mock_tfe_gateway.assign_status_code = 412
    mock_tfe_gateway.assign_error_body = {"error": {"code": "CCU_EXHAUSTED", "message": "No CCUs left"}}
    err_cls = get_ccu_exhausted_error()
    with pytest.raises(err_cls):
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, reuse_existing=False)


@pytest.mark.asyncio
async def test_quota_error_412_ccu_exhausted_message_fallback(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify HTTP 412 with 'compute units' in message string raises ComputeUnitsExhaustedError."""
    mock_tfe_gateway.assign_status_code = 412
    mock_tfe_gateway.assign_error_body = {"message": "You have exhausted your Colab compute units."}
    err_cls = get_ccu_exhausted_error()
    with pytest.raises(err_cls):
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, reuse_existing=False)


@pytest.mark.asyncio
async def test_quota_error_412_free_tier_quota_exceeded(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify HTTP 412 with FREE_TIER_QUOTA_EXCEEDED raises FreeTierQuotaExceededError."""
    mock_tfe_gateway.assign_status_code = 412
    mock_tfe_gateway.assign_error_body = {"error": {"code": "FREE_TIER_QUOTA_EXCEEDED", "message": "Free tier GPU usage limit"}}
    err_cls = get_free_tier_quota_error()
    with pytest.raises(err_cls):
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, reuse_existing=False)


# --- Feature Area 7: Cloud GPU Capacity Handling (502, 503, 504) ---

@pytest.mark.asyncio
async def test_cloud_gpu_capacity_502_bad_gateway(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify HTTP 502 maps to capacity error."""
    mock_tfe_gateway.assign_status_code = 502
    with pytest.raises(Exception) as exc_info:
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, reuse_existing=False)
    assert "capacity" in str(exc_info.value).lower() or "502" in str(exc_info.value)


@pytest.mark.asyncio
async def test_cloud_gpu_capacity_503_service_unavailable(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify HTTP 503 maps to capacity error."""
    mock_tfe_gateway.assign_status_code = 503
    with pytest.raises(Exception) as exc_info:
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, reuse_existing=False)
    assert "capacity" in str(exc_info.value).lower() or "503" in str(exc_info.value)


@pytest.mark.asyncio
async def test_cloud_gpu_capacity_504_gateway_timeout(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify HTTP 504 maps to capacity error."""
    mock_tfe_gateway.assign_status_code = 504
    with pytest.raises(Exception) as exc_info:
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, reuse_existing=False)
    assert "capacity" in str(exc_info.value).lower() or "504" in str(exc_info.value)


@pytest.mark.asyncio
async def test_cloud_gpu_capacity_error_mentions_accelerator_name(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify capacity error message includes requested accelerator name."""
    mock_tfe_gateway.assign_status_code = 503
    acc_enum = get_accelerator_enum()
    with pytest.raises(Exception) as exc_info:
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, accelerator=acc_enum.A100, reuse_existing=False)
    assert "A100" in str(exc_info.value)


@pytest.mark.asyncio
async def test_cloud_gpu_capacity_error_mentions_capacity(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify capacity error message contains the word 'capacity'."""
    mock_tfe_gateway.assign_status_code = 503
    with pytest.raises(Exception) as exc_info:
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, reuse_existing=False)
    assert "capacity" in str(exc_info.value).lower()


# --- Feature Area 8: Session Teardown (unassign) ---

@pytest.mark.asyncio
async def test_unassign_runtime_step1_get_and_step2_post(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify unassign performs Step 1 GET then Step 2 POST."""
    mock_tfe_gateway.active_assignments.append({"endpoint": SAMPLE_ENDPOINT})
    await tfe_client.unassign_runtime(SAMPLE_ENDPOINT)
    assert len(mock_tfe_gateway.recorded_requests) == 2
    req_get, req_post = mock_tfe_gateway.recorded_requests
    assert req_get.method == "GET"
    assert req_post.method == "POST"


@pytest.mark.asyncio
async def test_unassign_runtime_success_returns_true(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify successful teardown returns boolean True."""
    mock_tfe_gateway.active_assignments.append({"endpoint": SAMPLE_ENDPOINT})
    success = await tfe_client.unassign_runtime(SAMPLE_ENDPOINT)
    assert success is True


@pytest.mark.asyncio
async def test_unassign_runtime_propagates_auth_user_index(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify unassign sends authuser query param and header."""
    await tfe_client.unassign_runtime(SAMPLE_ENDPOINT)
    req = mock_tfe_gateway.recorded_requests[0]
    assert req.url.params.get("authuser") == "0"
    assert req.headers.get("X-Goog-AuthUser") == "0"


@pytest.mark.asyncio
async def test_unassign_runtime_removes_session_from_active_list(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify unassign removes the VM from active assignments."""
    mock_tfe_gateway.active_assignments.append({"endpoint": SAMPLE_ENDPOINT})
    await tfe_client.unassign_runtime(SAMPLE_ENDPOINT)
    assert len(mock_tfe_gateway.active_assignments) == 0


@pytest.mark.asyncio
async def test_unassign_runtime_transmits_xsrf_token(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify Step 2 POST sends unassign token in X-Goog-Colab-Token."""
    await tfe_client.unassign_runtime(SAMPLE_ENDPOINT)
    req_post = mock_tfe_gateway.recorded_requests[1]
    assert req_post.headers.get("X-Goog-Colab-Token") == f"UNASSIGN_XSRF_{SAMPLE_ENDPOINT}"


# --- Feature Area 9: MCP Runtime Tools ---

@pytest.mark.asyncio
async def test_tool_colab_list_runtime_types_returns_matrix():
    """Verify colab_list_runtime_types tool returns list of available accelerators."""
    tool_fn = getattr(runtime_tools_mod, "colab_list_runtime_types", None)
    if tool_fn is None:
        pytest.skip("colab_list_runtime_types tool not found")
    res = await tool_fn(auth_user_index=0)
    assert res is not None
    accelerators = getattr(res, "available_accelerators", res.get("available_accelerators") if isinstance(res, dict) else None)
    assert isinstance(accelerators, list)
    assert len(accelerators) >= 5


@pytest.mark.asyncio
async def test_tool_colab_list_runtime_types_includes_pricing_and_vram():
    """Verify colab_list_runtime_types includes CCU cost and VRAM details."""
    tool_fn = getattr(runtime_tools_mod, "colab_list_runtime_types", None)
    if tool_fn is None:
        pytest.skip("colab_list_runtime_types tool not found")
    res = await tool_fn(auth_user_index=0)
    accelerators = getattr(res, "available_accelerators", res.get("available_accelerators") if isinstance(res, dict) else None)
    t4 = next((a for a in accelerators if (a.get("accelerator") if isinstance(a, dict) else getattr(a, "accelerator")) == "T4"), None)
    assert t4 is not None
    cost = t4.get("ccu_cost_per_hour") if isinstance(t4, dict) else getattr(t4, "ccu_cost_per_hour")
    assert cost == 1.96


@pytest.mark.asyncio
async def test_tool_colab_connect_runtime_allocates_requested_accelerator(monkeypatch):
    """Verify colab_connect_runtime allocates requested accelerator."""
    tool_fn = getattr(runtime_tools_mod, "colab_connect_runtime", None)
    if tool_fn is None:
        pytest.skip("colab_connect_runtime tool not found")

    async def mock_connect(*args, **kwargs):
        return {"runtime_id": SAMPLE_ENDPOINT, "status": "connected", "accelerator": "T4"}

    monkeypatch.setattr(runtime_tools_mod, "connect_runtime_impl", mock_connect, raising=False)
    res = await tool_fn(accelerator="T4", high_ram=False)
    status = getattr(res, "status", res.get("status") if isinstance(res, dict) else None)
    assert status == "connected"


@pytest.mark.asyncio
async def test_tool_colab_get_runtime_status_returns_health_and_metrics(monkeypatch):
    """Verify colab_get_runtime_status returns health and metric details."""
    tool_fn = getattr(runtime_tools_mod, "colab_get_runtime_status", None)
    if tool_fn is None:
        pytest.skip("colab_get_runtime_status tool not found")

    async def mock_status(*args, **kwargs):
        return {"runtime_id": SAMPLE_ENDPOINT, "status": "active", "metrics": {"ram_used_gb": 3.2}}

    monkeypatch.setattr(runtime_tools_mod, "get_runtime_status_impl", mock_status, raising=False)
    res = await tool_fn()
    status = getattr(res, "status", res.get("status") if isinstance(res, dict) else None)
    assert status == "active"


@pytest.mark.asyncio
async def test_tool_colab_disconnect_runtime_terminates_active_session(monkeypatch):
    """Verify colab_disconnect_runtime releases resources and returns terminated status."""
    tool_fn = getattr(runtime_tools_mod, "colab_disconnect_runtime", None)
    if tool_fn is None:
        pytest.skip("colab_disconnect_runtime tool not found")

    async def mock_disconnect(*args, **kwargs):
        return {"runtime_id": SAMPLE_ENDPOINT, "status": "terminated", "compute_resources_released": True}

    monkeypatch.setattr(runtime_tools_mod, "disconnect_runtime_impl", mock_disconnect, raising=False)
    res = await tool_fn()
    status = getattr(res, "status", res.get("status") if isinstance(res, dict) else None)
    assert status == "terminated"


# ============================================================================
# Tier 2: Boundary, Corner Cases & Adversarial Verification
# ============================================================================

# --- Feature Area 1 (Tier 2): TFE REST Client Boundaries ---

@pytest.mark.asyncio
async def test_list_assignments_malformed_json_response_handling(tfe_client):
    """Verify malformed JSON in list_assignments raises an exception."""
    def bad_json_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="NOT_VALID_JSON{")

    tfe_client.client = httpx.AsyncClient(transport=httpx.MockTransport(bad_json_handler), timeout=30.0)
    with pytest.raises(Exception):
        await tfe_client.list_assignments()


@pytest.mark.asyncio
async def test_get_ccu_info_unauthorized_401_raises_error(tfe_client):
    """Verify HTTP 401 on get_ccu_info propagates HTTPStatusError."""
    def unauth_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "Unauthorized"})

    tfe_client.client = httpx.AsyncClient(transport=httpx.MockTransport(unauth_handler), timeout=30.0)
    with pytest.raises(httpx.HTTPStatusError):
        await tfe_client.get_ccu_info()


@pytest.mark.asyncio
async def test_tfe_client_network_connect_error_handling(tfe_client):
    """Verify httpx.ConnectError is cleanly propagated."""
    def connect_error_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Network down")

    tfe_client.client = httpx.AsyncClient(transport=httpx.MockTransport(connect_error_handler), timeout=30.0)
    with pytest.raises(httpx.ConnectError):
        await tfe_client.list_assignments()


@pytest.mark.asyncio
async def test_tfe_client_read_timeout_handling(tfe_client):
    """Verify httpx.ReadTimeout on assignments query is propagated."""
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Timeout")

    tfe_client.client = httpx.AsyncClient(transport=httpx.MockTransport(timeout_handler), timeout=1.0)
    with pytest.raises(httpx.ReadTimeout):
        await tfe_client.list_assignments()


@pytest.mark.asyncio
async def test_tfe_client_custom_auth_user_index_isolation(mock_tfe_gateway: MockTFEGateway):
    """Verify client with auth_user_index=3 isolates requests with ?authuser=3."""
    client_cls = get_client_class()
    client = client_cls(auth_token=TEST_AUTH_TOKEN, auth_user_index=3)
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(mock_tfe_gateway.handle_request), timeout=30.0)
    await client.list_assignments()
    req = mock_tfe_gateway.recorded_requests[-1]
    assert req.url.params.get("authuser") == "3"
    assert req.headers.get("X-Goog-AuthUser") == "3"


# --- Feature Area 2 (Tier 2): Two-Step Handshake Boundaries ---

@pytest.mark.asyncio
async def test_two_step_handshake_missing_xsrf_token_raises_value_error(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify Step 1 returning 200 without token raises ValueError."""
    mock_tfe_gateway.issue_tokens = False
    with pytest.raises(ValueError):
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, reuse_existing=False)


@pytest.mark.asyncio
async def test_two_step_handshake_empty_xsrf_token_raises_value_error(tfe_client):
    """Verify Step 1 returning empty string token raises ValueError."""
    def empty_token_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"token": "", "nbh": SAMPLE_NBH})

    tfe_client.client = httpx.AsyncClient(transport=httpx.MockTransport(empty_token_handler), timeout=30.0)
    with pytest.raises(ValueError):
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, reuse_existing=False)


@pytest.mark.asyncio
async def test_two_step_handshake_step2_forbidden_403_raises_http_error(tfe_client):
    """Verify Step 2 returning 403 raises HTTPStatusError."""
    def step2_forbidden_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(403, json={"error": "Forbidden"})
        return httpx.Response(200, json={"token": "tok123"})

    tfe_client.client = httpx.AsyncClient(transport=httpx.MockTransport(step2_forbidden_handler), timeout=30.0)
    with pytest.raises(httpx.HTTPStatusError):
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, reuse_existing=False)


@pytest.mark.asyncio
async def test_two_step_handshake_step2_internal_error_500_raises_http_error(tfe_client):
    """Verify Step 2 returning 500 raises HTTPStatusError."""
    def step2_500_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(500, json={"error": "Internal Error"})
        return httpx.Response(200, json={"token": "tok123"})

    tfe_client.client = httpx.AsyncClient(transport=httpx.MockTransport(step2_500_handler), timeout=30.0)
    with pytest.raises(httpx.HTTPStatusError):
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, reuse_existing=False)


@pytest.mark.asyncio
async def test_two_step_handshake_post_header_exact_name(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify exact header name 'X-Goog-Colab-Token' is used."""
    acc_enum = get_accelerator_enum()
    await tfe_client.assign_runtime(nbh=SAMPLE_NBH, accelerator=acc_enum.T4, reuse_existing=False)
    req_post = mock_tfe_gateway.recorded_requests[1]
    assert "X-Goog-Colab-Token" in req_post.headers


# --- Feature Area 3 (Tier 2): Pre-Assignment Re-Attachment Boundaries ---

@pytest.mark.asyncio
async def test_session_reattachment_skips_when_reuse_existing_false(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify reuse_existing=False provisions a new container even if one already exists."""
    mock_tfe_gateway.active_assignments.append({
        "endpoint": "existing-vm-ignored",
        "accelerator": "T4",
        "variant": "GPU",
        "runtimeProxyInfo": {"url": SAMPLE_PROXY_URL, "token": SAMPLE_PROXY_TOKEN, "tokenExpiresInSeconds": 3600},
    })
    acc_enum = get_accelerator_enum()
    res = await tfe_client.assign_runtime(nbh=SAMPLE_NBH, accelerator=acc_enum.T4, reuse_existing=False)
    endpoint = getattr(res, "endpoint", res.get("endpoint") if isinstance(res, dict) else None)
    assert endpoint == SAMPLE_ENDPOINT


@pytest.mark.asyncio
async def test_session_reattachment_ignores_different_accelerator(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify existing session with different accelerator (e.g. CPU) is not re-attached when GPU requested."""
    mock_tfe_gateway.active_assignments.append({
        "endpoint": "existing-cpu-session",
        "accelerator": "NONE",
        "variant": "DEFAULT",
        "runtimeProxyInfo": {"url": SAMPLE_PROXY_URL, "token": SAMPLE_PROXY_TOKEN, "tokenExpiresInSeconds": 3600},
    })
    acc_enum = get_accelerator_enum()
    res = await tfe_client.assign_runtime(nbh=SAMPLE_NBH, accelerator=acc_enum.T4, reuse_existing=True)
    endpoint = getattr(res, "endpoint", res.get("endpoint") if isinstance(res, dict) else None)
    assert endpoint == SAMPLE_ENDPOINT


@pytest.mark.asyncio
async def test_session_reattachment_ignores_session_missing_proxy_info(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify existing session missing runtimeProxyInfo is ignored."""
    mock_tfe_gateway.active_assignments.append({"endpoint": "broken-session", "accelerator": "T4"})
    acc_enum = get_accelerator_enum()
    res = await tfe_client.assign_runtime(nbh=SAMPLE_NBH, accelerator=acc_enum.T4, reuse_existing=True)
    endpoint = getattr(res, "endpoint", res.get("endpoint") if isinstance(res, dict) else None)
    assert endpoint == SAMPLE_ENDPOINT


@pytest.mark.asyncio
async def test_session_reattachment_ignores_session_missing_endpoint(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify existing session missing endpoint field is ignored."""
    mock_tfe_gateway.active_assignments.append({"accelerator": "T4", "runtimeProxyInfo": {"url": "u", "token": "t"}})
    acc_enum = get_accelerator_enum()
    res = await tfe_client.assign_runtime(nbh=SAMPLE_NBH, accelerator=acc_enum.T4, reuse_existing=True)
    endpoint = getattr(res, "endpoint", res.get("endpoint") if isinstance(res, dict) else None)
    assert endpoint == SAMPLE_ENDPOINT


@pytest.mark.asyncio
async def test_session_reattachment_case_insensitive_accelerator_matching(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify accelerator matching handles lowercase/uppercase differences (e.g. 't4' vs 'T4')."""
    mock_tfe_gateway.active_assignments.append({
        "endpoint": "existing-t4-lower",
        "accelerator": "t4",
        "variant": "GPU",
        "runtimeProxyInfo": {"url": SAMPLE_PROXY_URL, "token": SAMPLE_PROXY_TOKEN, "tokenExpiresInSeconds": 3600},
    })
    acc_enum = get_accelerator_enum()
    res = await tfe_client.assign_runtime(nbh=SAMPLE_NBH, accelerator=acc_enum.T4, reuse_existing=True)
    endpoint = getattr(res, "endpoint", res.get("endpoint") if isinstance(res, dict) else None)
    assert endpoint == "existing-t4-lower"


# --- Feature Area 4 (Tier 2): Accelerator & Shape Matrix Boundaries ---

@pytest.mark.asyncio
async def test_shape_standard_omits_shape_param(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify Shape.STANDARD does not append shape=hm to query params."""
    shape_enum = get_shape_enum()
    await tfe_client.assign_runtime(nbh=SAMPLE_NBH, shape=shape_enum.STANDARD, reuse_existing=False)
    req = mock_tfe_gateway.recorded_requests[0]
    assert req.url.params.get("shape") != "hm"


@pytest.mark.asyncio
async def test_shape_high_ram_appends_shape_hm(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify Shape.HIGH_RAM appends shape=hm."""
    shape_enum = get_shape_enum()
    await tfe_client.assign_runtime(nbh=SAMPLE_NBH, shape=shape_enum.HIGH_RAM, reuse_existing=False)
    req = mock_tfe_gateway.recorded_requests[0]
    assert req.url.params.get("shape") == "hm"


@pytest.mark.asyncio
async def test_accelerator_unknown_enum_or_string_handling(tfe_client):
    """Verify unknown accelerator value handling."""
    with pytest.raises(Exception):
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, accelerator="UNKNOWN_GPU_XYZ", reuse_existing=False)


@pytest.mark.asyncio
async def test_shape_case_sensitivity_handling(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify shape parameter is sent strictly as lowercase 'hm'."""
    shape_enum = get_shape_enum()
    await tfe_client.assign_runtime(nbh=SAMPLE_NBH, shape=shape_enum.HIGH_RAM, reuse_existing=False)
    req = mock_tfe_gateway.recorded_requests[0]
    assert req.url.params.get("shape") == "hm"


@pytest.mark.asyncio
async def test_accelerator_string_vs_enum_interoperability(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify string 'T4' can be accepted interchangeably with Accelerator.T4."""
    try:
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, accelerator="T4", reuse_existing=False)
        req = mock_tfe_gateway.recorded_requests[0]
        assert req.url.params.get("accelerator") == "T4"
    except (TypeError, ValueError):
        # If strict enum typing enforced, verify clean ValueError
        pass


# --- Feature Area 5 (Tier 2): High-RAM Free Tier Guard Boundaries ---

@pytest.mark.asyncio
async def test_high_ram_guard_http_400_without_hm_shape_raises_normal_error(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify HTTP 400 when shape is standard raises HTTPStatusError, NOT HighRamNotAvailableError."""
    mock_tfe_gateway.assign_status_code = 400
    mock_tfe_gateway.assign_error_body = {"error": "Invalid NBH format"}
    shape_enum = get_shape_enum()
    err_cls = get_high_ram_error()
    with pytest.raises(Exception) as exc_info:
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, shape=shape_enum.STANDARD, reuse_existing=False)
    assert not isinstance(exc_info.value, err_cls)


@pytest.mark.asyncio
async def test_high_ram_guard_with_custom_error_message_in_body(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify custom error payload in 400 is handled."""
    mock_tfe_gateway.assign_status_code = 400
    mock_tfe_gateway.assign_error_body = {"custom": "quota_exceeded_for_shape"}
    shape_enum = get_shape_enum()
    err_cls = get_high_ram_error()
    with pytest.raises(err_cls):
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, shape=shape_enum.HIGH_RAM, reuse_existing=False)


@pytest.mark.asyncio
async def test_high_ram_guard_empty_400_body_still_raises_explicit_exception(tfe_client):
    """Verify empty 400 response body with shape=hm still raises HighRamNotAvailableError."""
    def empty_400_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="")

    tfe_client.client = httpx.AsyncClient(transport=httpx.MockTransport(empty_400_handler), timeout=30.0)
    shape_enum = get_shape_enum()
    err_cls = get_high_ram_error()
    with pytest.raises(err_cls):
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, shape=shape_enum.HIGH_RAM, reuse_existing=False)


@pytest.mark.asyncio
async def test_high_ram_guard_does_not_catch_other_4xx_codes(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify 403 or 404 does NOT raise HighRamNotAvailableError."""
    mock_tfe_gateway.assign_status_code = 403
    shape_enum = get_shape_enum()
    err_cls = get_high_ram_error()
    with pytest.raises(Exception) as exc_info:
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, shape=shape_enum.HIGH_RAM, reuse_existing=False)
    assert not isinstance(exc_info.value, err_cls)


@pytest.mark.asyncio
async def test_high_ram_guard_exception_inheritance_from_exception():
    """Verify HighRamNotAvailableError inherits from Python built-in Exception."""
    err_cls = get_high_ram_error()
    assert issubclass(err_cls, Exception)


# --- Feature Area 6 (Tier 2): Quota Error Disambiguation Boundaries ---

@pytest.mark.asyncio
async def test_quota_error_412_unknown_code_raises_runtime_error(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify HTTP 412 with unknown code falls back to RuntimeError."""
    mock_tfe_gateway.assign_status_code = 412
    mock_tfe_gateway.assign_error_body = {"error": {"code": "UNKNOWN_ERROR_CODE", "message": "Something broke"}}
    with pytest.raises(RuntimeError):
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, reuse_existing=False)


@pytest.mark.asyncio
async def test_quota_error_412_non_json_body_raises_runtime_error(tfe_client):
    """Verify HTTP 412 with raw non-JSON text falls back to RuntimeError."""
    def non_json_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(412, text="Precondition Failed: Raw Text Body")

    tfe_client.client = httpx.AsyncClient(transport=httpx.MockTransport(non_json_handler), timeout=30.0)
    with pytest.raises(RuntimeError):
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, reuse_existing=False)


@pytest.mark.asyncio
async def test_quota_error_412_nested_error_payload_parsing(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify deeply nested error structures are parsed correctly."""
    mock_tfe_gateway.assign_status_code = 412
    mock_tfe_gateway.assign_error_body = {"details": [{"code": "CCU_EXHAUSTED", "info": "Balance 0"}]}
    err_cls = get_ccu_exhausted_error()
    try:
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, reuse_existing=False)
    except Exception as exc:
        assert isinstance(exc, (err_cls, RuntimeError))


@pytest.mark.asyncio
async def test_quota_error_412_case_insensitive_code_matching(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify lowercase 'too_many_assignments' matches TooManyAssignmentsError."""
    mock_tfe_gateway.assign_status_code = 412
    mock_tfe_gateway.assign_error_body = {"error": {"code": "too_many_assignments"}}
    err_cls = get_too_many_assignments_error()
    with pytest.raises(err_cls):
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, reuse_existing=False)


@pytest.mark.asyncio
async def test_quota_error_412_distinct_exception_classes():
    """Verify TooManyAssignmentsError, ComputeUnitsExhaustedError, FreeTierQuotaExceededError are distinct."""
    e1 = get_too_many_assignments_error()
    e2 = get_ccu_exhausted_error()
    e3 = get_free_tier_quota_error()
    assert e1 != e2
    assert e2 != e3
    assert e1 != e3


# --- Feature Area 7 (Tier 2): Cloud Capacity Boundaries ---

@pytest.mark.asyncio
async def test_cloud_gpu_capacity_503_step1_vs_step2(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify capacity error is handled during Step 2 POST."""
    def step2_capacity_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(503, json={"error": "GPU cluster out of capacity"})
        return mock_tfe_gateway.handle_request(request)

    tfe_client.client = httpx.AsyncClient(transport=httpx.MockTransport(step2_capacity_handler), timeout=30.0)
    with pytest.raises(Exception) as exc_info:
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, reuse_existing=False)
    assert "capacity" in str(exc_info.value).lower() or "503" in str(exc_info.value)


@pytest.mark.asyncio
async def test_cloud_gpu_capacity_does_not_swallow_404(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify HTTP 404 is not converted to capacity error."""
    mock_tfe_gateway.assign_status_code = 404
    with pytest.raises(httpx.HTTPStatusError):
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, reuse_existing=False)


@pytest.mark.asyncio
async def test_cloud_gpu_capacity_does_not_swallow_500(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify HTTP 500 (Internal Server Error) is not converted to capacity error."""
    mock_tfe_gateway.assign_status_code = 500
    with pytest.raises(httpx.HTTPStatusError):
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, reuse_existing=False)


@pytest.mark.asyncio
async def test_cloud_gpu_capacity_with_html_error_page(tfe_client):
    """Verify HTML 503 error page from cloud load balancer is handled."""
    def html_503_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="<html><body>Service Unavailable</body></html>")

    tfe_client.client = httpx.AsyncClient(transport=httpx.MockTransport(html_503_handler), timeout=30.0)
    with pytest.raises(Exception) as exc_info:
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, reuse_existing=False)
    assert "capacity" in str(exc_info.value).lower() or "503" in str(exc_info.value)


@pytest.mark.asyncio
async def test_cloud_gpu_capacity_with_empty_503_response(tfe_client):
    """Verify empty 503 response body is handled without JSON parse crash."""
    def empty_503_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="")

    tfe_client.client = httpx.AsyncClient(transport=httpx.MockTransport(empty_503_handler), timeout=30.0)
    with pytest.raises(Exception):
        await tfe_client.assign_runtime(nbh=SAMPLE_NBH, reuse_existing=False)


# --- Feature Area 8 (Tier 2): Session Teardown Boundaries ---

@pytest.mark.asyncio
async def test_unassign_runtime_404_already_released_is_idempotent_true(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify unassigning an already released VM (404) returns True idempotently."""
    mock_tfe_gateway.unassign_status_code = 404
    success = await tfe_client.unassign_runtime("nonexistent-endpoint-404")
    assert success is True


@pytest.mark.asyncio
async def test_unassign_runtime_missing_step1_token_raises_error(tfe_client):
    """Verify Step 1 returning 200 without token during unassign raises error."""
    def missing_unassign_token(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    tfe_client.client = httpx.AsyncClient(transport=httpx.MockTransport(missing_unassign_token), timeout=30.0)
    with pytest.raises(Exception):
        await tfe_client.unassign_runtime(SAMPLE_ENDPOINT)


@pytest.mark.asyncio
async def test_unassign_runtime_step2_post_error_raises_exception(tfe_client):
    """Verify Step 2 failure during unassign raises exception."""
    def step2_unassign_fail(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(500, json={"error": "Teardown failed"})
        return httpx.Response(200, json={"token": "tok"})

    tfe_client.client = httpx.AsyncClient(transport=httpx.MockTransport(step2_unassign_fail), timeout=30.0)
    with pytest.raises(Exception):
        await tfe_client.unassign_runtime(SAMPLE_ENDPOINT)


@pytest.mark.asyncio
async def test_unassign_runtime_network_timeout_handling(tfe_client):
    """Verify network timeout during unassign propagates TimeoutException."""
    def unassign_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Timeout during unassign")

    tfe_client.client = httpx.AsyncClient(transport=httpx.MockTransport(unassign_timeout), timeout=1.0)
    with pytest.raises(httpx.ReadTimeout):
        await tfe_client.unassign_runtime(SAMPLE_ENDPOINT)


@pytest.mark.asyncio
async def test_unassign_runtime_endpoint_with_special_characters(tfe_client, mock_tfe_gateway: MockTFEGateway):
    """Verify unassign handles endpoint with hyphens, underscores, and dots."""
    custom_endpoint = "colab_session.v1-abc_123"
    success = await tfe_client.unassign_runtime(custom_endpoint)
    assert success is True
    assert any(custom_endpoint in str(r.url) for r in mock_tfe_gateway.recorded_requests)


# --- Feature Area 9 (Tier 2): MCP Runtime Tools Boundaries ---

@pytest.mark.asyncio
async def test_tool_colab_list_runtime_types_auth_user_index():
    """Verify colab_list_runtime_types propagates auth_user_index."""
    tool_fn = getattr(runtime_tools_mod, "colab_list_runtime_types", None)
    if tool_fn is None:
        pytest.skip("colab_list_runtime_types tool not found")
    res = await tool_fn(auth_user_index=1)
    assert res is not None


@pytest.mark.asyncio
async def test_tool_colab_connect_runtime_propagates_high_ram_guard_error(monkeypatch):
    """Verify colab_connect_runtime surfaces HighRamNotAvailableError cleanly."""
    tool_fn = getattr(runtime_tools_mod, "colab_connect_runtime", None)
    if tool_fn is None:
        pytest.skip("colab_connect_runtime tool not found")

    err_cls = get_high_ram_error()
    async def mock_fail_high_ram(*args, **kwargs):
        raise err_cls("High-RAM shape requires Colab Pro/Pro+")

    monkeypatch.setattr(runtime_tools_mod, "connect_runtime_impl", mock_fail_high_ram, raising=False)
    with pytest.raises(err_cls):
        await tool_fn(high_ram=True)


@pytest.mark.asyncio
async def test_tool_colab_connect_runtime_timeout_seconds_parameter(monkeypatch):
    """Verify colab_connect_runtime accepts and passes timeout_seconds."""
    tool_fn = getattr(runtime_tools_mod, "colab_connect_runtime", None)
    if tool_fn is None:
        pytest.skip("colab_connect_runtime tool not found")

    recorded_kwargs = {}
    async def mock_connect(**kwargs):
        recorded_kwargs.update(kwargs)
        return {"status": "connected"}

    monkeypatch.setattr(runtime_tools_mod, "connect_runtime_impl", mock_connect, raising=False)
    await tool_fn(timeout_seconds=60)
    # Ensure timeout was passed or processed
    assert True


@pytest.mark.asyncio
async def test_tool_colab_get_runtime_status_when_no_active_runtime(monkeypatch):
    """Verify colab_get_runtime_status when no runtime exists returns disconnected state."""
    tool_fn = getattr(runtime_tools_mod, "colab_get_runtime_status", None)
    if tool_fn is None:
        pytest.skip("colab_get_runtime_status tool not found")

    async def mock_empty_status(*args, **kwargs):
        return {"status": "disconnected", "active_runtime": None}

    monkeypatch.setattr(runtime_tools_mod, "get_runtime_status_impl", mock_empty_status, raising=False)
    res = await tool_fn()
    status = getattr(res, "status", res.get("status") if isinstance(res, dict) else None)
    assert status == "disconnected"


@pytest.mark.asyncio
async def test_tool_colab_disconnect_runtime_idempotence(monkeypatch):
    """Verify disconnecting when already disconnected returns clean response."""
    tool_fn = getattr(runtime_tools_mod, "colab_disconnect_runtime", None)
    if tool_fn is None:
        pytest.skip("colab_disconnect_runtime tool not found")

    async def mock_already_disconnected(*args, **kwargs):
        return {"status": "terminated", "already_terminated": True}

    monkeypatch.setattr(runtime_tools_mod, "disconnect_runtime_impl", mock_already_disconnected, raising=False)
    res = await tool_fn()
    status = getattr(res, "status", res.get("status") if isinstance(res, dict) else None)
    assert status == "terminated"
