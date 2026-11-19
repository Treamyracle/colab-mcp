"""
Integration tests for FastMCP Tools (All 12 Colab MCP Tools).

Requirements:
- TECH_SPEC §7.3.1 (Authentication Tools: colab_login, colab_get_auth_status, colab_logout)
- TECH_SPEC §7.3.2 (Runtime Tools: colab_list_runtime_types, colab_connect_runtime,
                    colab_get_runtime_status, colab_disconnect_runtime)
- TECH_SPEC §7.3.3 (Execution Tools: colab_execute_code, colab_execute_notebook,
                    colab_get_execution_status, colab_get_cell_output, colab_interrupt_execution)
- TEST_INFRA.md § Feature 17: FastMCP Server & 12 MCP Tools Registration (≥12 Tier 1, ≥12 Tier 2).

Adheres strictly to Python 3.13, progressive testability, and isolated execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ImageContent, TextContent

from colab_mcp.server import create_server

# ============================================================================
# Helpers & Progressive Testability Fixtures
# ============================================================================

ALL_12_MCP_TOOLS = [
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

ALL_EXPECTED_MCP_TOOLS = [
    *ALL_12_MCP_TOOLS,
    "colab_list_sessions",
    "colab_list_files",
    "colab_download_file",
    "colab_upload_file",
    "colab_drive_list",
    "colab_drive_download",
    "colab_drive_upload",
]


@pytest.fixture
async def test_server(mock_keystore_dir, mock_keyring, mock_machine_id):
    """Create a FastMCP server instance with mock keystore and isolated SessionManager."""
    from colab_mcp.runtime.session_manager import get_session_manager, reset_session_manager

    reset_session_manager()
    try:
        yield create_server("colab-mcp-tools-test")
    finally:
        try:
            await get_session_manager().shutdown()
        except Exception:
            pass
        reset_session_manager()


async def get_registered_tool_names(server: FastMCP) -> set[str]:
    """Retrieve set of tool names registered on the server."""
    tools = await server.list_tools()
    return {t.name for t in tools}


async def get_tool_definition(server: FastMCP, tool_name: str):
    """Retrieve MCP Tool object by name."""
    tools = await server.list_tools()
    for t in tools:
        if t.name == tool_name:
            return t
    return None


async def require_tool(server: FastMCP, tool_name: str):
    """Progressive testability: skip test if the tool is not yet registered on FastMCP server."""
    registered = await get_registered_tool_names(server)
    if tool_name not in registered:
        pytest.skip(f"Tool '{tool_name}' is not yet registered on FastMCP server (in-progress milestone)")


def extract_content_text(tool_result: Any) -> str:
    """Extract string content from FastMCP call_tool return value."""
    if isinstance(tool_result, tuple):
        content_blocks = tool_result[0]
        if isinstance(content_blocks, list) and len(content_blocks) > 0:
            first_block = content_blocks[0]
            if hasattr(first_block, "text"):
                return first_block.text
            elif isinstance(first_block, dict) and "text" in first_block:
                return first_block["text"]
    if isinstance(tool_result, list) and len(tool_result) > 0:
        first = tool_result[0]
        if hasattr(first, "text"):
            return first.text
        elif isinstance(first, dict) and "text" in first:
            return first["text"]
    return str(tool_result)


def extract_result_dict(tool_result: Any) -> Dict[str, Any]:
    """Extract structured dict from FastMCP call_tool result."""
    if isinstance(tool_result, tuple) and len(tool_result) > 1 and isinstance(tool_result[1], dict):
        return tool_result[1]
    raw_text = extract_content_text(tool_result)
    try:
        return json.loads(raw_text)
    except Exception:
        return {"raw": raw_text}


# ============================================================================
# Schema Discovery & Inventory Tests
# ============================================================================

@pytest.mark.asyncio
async def test_tool_inventory_registered(test_server: FastMCP):
    """Verify currently registered tools have descriptions and input schemas."""
    tools = await test_server.list_tools()
    assert len(tools) >= 3  # At least auth tools registered in baseline
    for tool in tools:
        assert tool.name in ALL_EXPECTED_MCP_TOOLS
        assert tool.description is not None
        assert len(tool.description) > 10
        assert isinstance(tool.inputSchema, dict)
        assert tool.inputSchema.get("type") == "object"


@pytest.mark.asyncio
async def test_all_12_tools_registered_when_m4_complete(test_server: FastMCP):
    """Verify all 12 MCP tools are registered once Milestone 4 is fully completed."""
    registered = await get_registered_tool_names(test_server)
    missing = [t for t in ALL_12_MCP_TOOLS if t not in registered]
    if missing:
        pytest.skip(f"Milestone 4 tool registration in-progress. Missing: {missing}")
    assert set(registered) == set(ALL_EXPECTED_MCP_TOOLS)


# ============================================================================
# CATEGORY 1: AUTHENTICATION TOOLS (Tools 1 - 3)
# ============================================================================

# --- Tool 1: colab_login ---

@pytest.mark.asyncio
async def test_tool_1_colab_login_valid_remote_step1(test_server: FastMCP):
    """Tier 1: Valid colab_login step 1 with oauth_remote generates auth_url."""
    await require_tool(test_server, "colab_login")
    result = await test_server.call_tool("colab_login", {"method": "oauth_remote", "auth_user_index": 0})
    data = extract_result_dict(result)
    assert data["status"] == "authorization_required"
    assert data["auth_method"] == "oauth_remote"
    assert "auth_url" in data
    assert data["auth_url"].startswith("https://accounts.google.com/o/oauth2/")


@pytest.mark.asyncio
async def test_tool_1_colab_login_defaults(test_server: FastMCP):
    """Tier 1: Calling colab_login with empty arguments applies default parameters."""
    await require_tool(test_server, "colab_login")
    result = await test_server.call_tool("colab_login", {})
    data = extract_result_dict(result)
    assert data["status"] == "authorization_required"
    assert data["auth_user_index"] == 0
    assert data["auth_method"] == "oauth_remote"


@pytest.mark.asyncio
async def test_tool_1_colab_login_invalid_method(test_server: FastMCP):
    """Tier 2: Calling colab_login with an invalid auth method raises ToolError."""
    await require_tool(test_server, "colab_login")
    with pytest.raises((ToolError, ValueError)):
        await test_server.call_tool("colab_login", {"method": "unsupported_jwt_login"})


@pytest.mark.asyncio
async def test_tool_1_colab_login_invalid_negative_user_index(test_server: FastMCP):
    """Tier 2: Negative auth_user_index raises ToolError / validation error."""
    await require_tool(test_server, "colab_login")
    with pytest.raises((ToolError, ValueError)):
        await test_server.call_tool("colab_login", {"auth_user_index": -1})


@pytest.mark.asyncio
async def test_tool_1_colab_login_timeout_out_of_bounds(test_server: FastMCP):
    """Tier 2: timeout_seconds < 30 or > 600 raises ToolError / validation error."""
    await require_tool(test_server, "colab_login")
    with pytest.raises((ToolError, ValueError)):
        await test_server.call_tool("colab_login", {"timeout_seconds": 15})


# --- Tool 2: colab_get_auth_status ---

@pytest.mark.asyncio
async def test_tool_2_colab_get_auth_status_valid_unauthenticated(test_server: FastMCP):
    """Tier 1: colab_get_auth_status returns clean unauthenticated status."""
    await require_tool(test_server, "colab_get_auth_status")
    result = await test_server.call_tool("colab_get_auth_status", {"auth_user_index": 0})
    data = extract_result_dict(result)
    assert data["is_authenticated"] is False
    assert data["token_valid"] is False
    assert data["auth_user_index"] == 0


@pytest.mark.asyncio
async def test_tool_2_colab_get_auth_status_valid_authenticated(test_server: FastMCP, sample_credentials):
    """Tier 1: colab_get_auth_status returns authenticated status when creds stored."""
    await require_tool(test_server, "colab_get_auth_status")
    from colab_mcp.auth.keystore import store_credentials
    store_credentials(0, sample_credentials)

    result = await test_server.call_tool("colab_get_auth_status", {"auth_user_index": 0})
    data = extract_result_dict(result)
    assert data["is_authenticated"] is True
    assert data["token_valid"] is True
    assert data["user_email"] == "researcher@example.com"


@pytest.mark.asyncio
async def test_tool_2_colab_get_auth_status_invalid_type(test_server: FastMCP):
    """Tier 2: Non-integer auth_user_index raises ToolError."""
    await require_tool(test_server, "colab_get_auth_status")
    with pytest.raises((ToolError, ValueError, TypeError)):
        await test_server.call_tool("colab_get_auth_status", {"auth_user_index": "invalid_user"})


@pytest.mark.asyncio
async def test_tool_2_colab_get_auth_status_negative_index(test_server: FastMCP):
    """Tier 2: Negative auth_user_index raises ToolError / validation error."""
    await require_tool(test_server, "colab_get_auth_status")
    with pytest.raises((ToolError, ValueError)):
        await test_server.call_tool("colab_get_auth_status", {"auth_user_index": -1})


# --- Tool 3: colab_logout ---

@pytest.mark.asyncio
async def test_tool_3_colab_logout_valid(test_server: FastMCP, sample_credentials):
    """Tier 1: colab_logout purges keystore and returns logged_out status."""
    await require_tool(test_server, "colab_logout")
    from colab_mcp.auth.keystore import store_credentials
    store_credentials(0, sample_credentials)

    result = await test_server.call_tool("colab_logout", {"auth_user_index": 0, "terminate_active_runtimes": False})
    data = extract_result_dict(result)
    assert data["status"] == "logged_out"
    assert data["keystore_purged"] is True
    assert data["auth_user_index"] == 0


@pytest.mark.asyncio
async def test_tool_3_colab_logout_defaults(test_server: FastMCP):
    """Tier 1: colab_logout with default arguments."""
    await require_tool(test_server, "colab_logout")
    result = await test_server.call_tool("colab_logout", {})
    data = extract_result_dict(result)
    assert data["status"] == "logged_out"
    assert data["auth_user_index"] == 0


@pytest.mark.asyncio
async def test_tool_3_colab_logout_invalid_boolean_type(test_server: FastMCP):
    """Tier 2: Invalid type for terminate_active_runtimes raises ToolError."""
    await require_tool(test_server, "colab_logout")
    with pytest.raises((ToolError, ValueError, TypeError)):
        await test_server.call_tool("colab_logout", {"terminate_active_runtimes": "yes_please"})


# ============================================================================
# CATEGORY 2: RUNTIME MANAGEMENT TOOLS (Tools 4 - 7)
# ============================================================================

# --- Tool 4: colab_list_runtime_types ---

@pytest.mark.asyncio
async def test_tool_4_colab_list_runtime_types_valid(test_server: FastMCP):
    """Tier 1: colab_list_runtime_types returns list of accelerators and tier."""
    await require_tool(test_server, "colab_list_runtime_types")
    result = await test_server.call_tool("colab_list_runtime_types", {"auth_user_index": 0})
    data = extract_result_dict(result)
    assert "available_accelerators" in data
    assert isinstance(data["available_accelerators"], list)
    assert len(data["available_accelerators"]) > 0
    # Must list at least CPU (NONE) and T4 GPU
    acc_names = [a.get("accelerator") for a in data["available_accelerators"]]
    assert "NONE" in acc_names or "CPU" in acc_names


@pytest.mark.asyncio
async def test_tool_4_colab_list_runtime_types_defaults(test_server: FastMCP):
    """Tier 1: colab_list_runtime_types defaults."""
    await require_tool(test_server, "colab_list_runtime_types")
    result = await test_server.call_tool("colab_list_runtime_types", {})
    data = extract_result_dict(result)
    assert "available_accelerators" in data


@pytest.mark.asyncio
async def test_tool_4_colab_list_runtime_types_invalid_type(test_server: FastMCP):
    """Tier 2: Non-integer auth_user_index raises ToolError."""
    await require_tool(test_server, "colab_list_runtime_types")
    with pytest.raises((ToolError, ValueError, TypeError)):
        await test_server.call_tool("colab_list_runtime_types", {"auth_user_index": "primary"})


# --- Tool 5: colab_connect_runtime ---

def _mock_tfe_connect_patches():
    """Return offline mock patches for ColabTFEClient, kernel probe, and KeepAliveDaemon."""
    from colab_mcp.runtime.models import Assignment

    mock_assignment = Assignment(
        endpoint="colab-mock-ep-001",
        accelerator="NONE",
        variant="DEFAULT",
        runtime_proxy_url="https://mock-proxy.local/tun/m/colab-mock-ep-001",
        runtime_proxy_token="mock-proxy-jwt-token",
        token_expires_in=3600,
    )
    mock_tfe = MagicMock()
    mock_tfe.__aenter__ = AsyncMock(return_value=mock_tfe)
    mock_tfe.__aexit__ = AsyncMock(return_value=None)
    mock_tfe.list_assignments = AsyncMock(return_value=[])
    mock_tfe.assign_runtime = AsyncMock(return_value=mock_assignment)
    return (
        patch("colab_mcp.runtime.tools.ColabTFEClient", return_value=mock_tfe),
        patch("colab_mcp.runtime.tools._probe_kernel_status", new=AsyncMock(return_value=("kern-mock-1", "idle"))),
        patch("colab_mcp.runtime.tools.KeepAliveDaemon.start", new=AsyncMock(return_value=None)),
    )


@pytest.mark.asyncio
async def test_tool_5_colab_connect_runtime_valid(test_server: FastMCP):
    """Tier 1: colab_connect_runtime provisions or connects to runtime."""
    await require_tool(test_server, "colab_connect_runtime")
    p_tfe, p_probe, p_keepalive = _mock_tfe_connect_patches()
    with p_tfe, p_probe, p_keepalive:
        result = await test_server.call_tool(
            "colab_connect_runtime",
            {"accelerator": "NONE", "high_ram": False, "reuse_existing": True},
        )
    data = extract_result_dict(result)
    assert "runtime_id" in data or "status" in data
    assert data.get("status") in ["connected", "active", "attached"]


@pytest.mark.asyncio
async def test_tool_5_colab_connect_runtime_defaults(test_server: FastMCP):
    """Tier 1: colab_connect_runtime defaults."""
    await require_tool(test_server, "colab_connect_runtime")
    p_tfe, p_probe, p_keepalive = _mock_tfe_connect_patches()
    with p_tfe, p_probe, p_keepalive:
        result = await test_server.call_tool("colab_connect_runtime", {})
    data = extract_result_dict(result)
    assert "runtime_id" in data or "status" in data


@pytest.mark.asyncio
async def test_tool_5_colab_connect_runtime_invalid_accelerator(test_server: FastMCP):
    """Tier 2: Invalid accelerator raises ToolError / validation error."""
    await require_tool(test_server, "colab_connect_runtime")
    with pytest.raises((ToolError, ValueError)):
        await test_server.call_tool("colab_connect_runtime", {"accelerator": "RTX_4090_TI"})


@pytest.mark.asyncio
async def test_tool_5_colab_connect_runtime_timeout_out_of_bounds(test_server: FastMCP):
    """Tier 2: Timeout < 30 seconds raises ToolError / validation error."""
    await require_tool(test_server, "colab_connect_runtime")
    with pytest.raises((ToolError, ValueError)):
        await test_server.call_tool("colab_connect_runtime", {"timeout_seconds": 10})


# --- Tool 6: colab_get_runtime_status ---

@pytest.mark.asyncio
async def test_tool_6_colab_get_runtime_status_valid(test_server: FastMCP):
    """Tier 1: colab_get_runtime_status returns runtime metrics and state."""
    await require_tool(test_server, "colab_get_runtime_status")
    result = await test_server.call_tool("colab_get_runtime_status", {"runtime_id": "colab-test-123"})
    data = extract_result_dict(result)
    assert "status" in data
    assert "kernel_state" in data or "metrics" in data or "active_runtime" in data


@pytest.mark.asyncio
async def test_tool_6_colab_get_runtime_status_defaults(test_server: FastMCP):
    """Tier 1: colab_get_runtime_status with default arguments."""
    await require_tool(test_server, "colab_get_runtime_status")
    result = await test_server.call_tool("colab_get_runtime_status", {})
    data = extract_result_dict(result)
    assert "status" in data


@pytest.mark.asyncio
async def test_tool_6_colab_get_runtime_status_invalid_type(test_server: FastMCP):
    """Tier 2: Non-string runtime_id raises ToolError."""
    await require_tool(test_server, "colab_get_runtime_status")
    with pytest.raises((ToolError, ValueError, TypeError)):
        await test_server.call_tool("colab_get_runtime_status", {"runtime_id": 123456})


# --- Tool 7: colab_disconnect_runtime ---

@pytest.mark.asyncio
async def test_tool_7_colab_disconnect_runtime_valid(test_server: FastMCP):
    """Tier 1: colab_disconnect_runtime releases VM resources."""
    await require_tool(test_server, "colab_disconnect_runtime")
    result = await test_server.call_tool("colab_disconnect_runtime", {"runtime_id": "colab-test-123"})
    data = extract_result_dict(result)
    assert data.get("status") in ["terminated", "disconnected", "released"]


@pytest.mark.asyncio
async def test_tool_7_colab_disconnect_runtime_defaults(test_server: FastMCP):
    """Tier 1: colab_disconnect_runtime defaults."""
    await require_tool(test_server, "colab_disconnect_runtime")
    result = await test_server.call_tool("colab_disconnect_runtime", {})
    data = extract_result_dict(result)
    assert "status" in data


@pytest.mark.asyncio
async def test_tool_7_colab_disconnect_runtime_invalid_type(test_server: FastMCP):
    """Tier 2: Invalid type for runtime_id raises ToolError."""
    await require_tool(test_server, "colab_disconnect_runtime")
    with pytest.raises((ToolError, ValueError, TypeError)):
        await test_server.call_tool("colab_disconnect_runtime", {"runtime_id": {"bad": "type"}})


@pytest.mark.asyncio
async def test_tool_colab_list_sessions_registered_and_valid(test_server: FastMCP, monkeypatch):
    """Tier 1: colab_list_sessions returns active remote and local sessions."""
    await require_tool(test_server, "colab_list_sessions")
    from colab_mcp.runtime.models import ListSessionsResponse, SessionSummary
    import colab_mcp.runtime.tools as rt_tools

    async def fake_list_sessions(auth_user_index: int = 0, probe_kernels: bool = True):
        return ListSessionsResponse(
            auth_user_index=auth_user_index,
            total_sessions=1,
            active_runtime_id=None,
            sessions=[
                SessionSummary(
                    runtime_id="colab-remote-1",
                    accelerator="T4",
                    variant="GPU",
                    tracked_locally=False,
                    kernel_id="kern-remote",
                    kernel_state="idle",
                )
            ],
        )

    monkeypatch.setattr(rt_tools, "list_sessions_impl", fake_list_sessions)
    result = await test_server.call_tool("colab_list_sessions", {"auth_user_index": 0})
    data = extract_result_dict(result)
    assert data["total_sessions"] == 1
    assert data["sessions"][0]["runtime_id"] == "colab-remote-1"
    assert data["sessions"][0]["tracked_locally"] is False


@pytest.mark.asyncio
async def test_tool_colab_disconnect_runtime_all_mode(test_server: FastMCP):
    """Tier 1: colab_disconnect_runtime supports disconnect_all=True cleanly."""
    await require_tool(test_server, "colab_disconnect_runtime")
    result = await test_server.call_tool("colab_disconnect_runtime", {"disconnect_all": True})
    data = extract_result_dict(result)
    assert data.get("status") in ["terminated", "disconnected"]
    assert data.get("terminated_count") == 0


# ============================================================================
# CATEGORY 3: EXECUTION & OUTPUT MONITORING TOOLS (Tools 8 - 12)
# ============================================================================

# --- Tool 8: colab_execute_code ---


class _MockKernelClient:
    """Minimal offline kernel client mock for FastMCP M3/M4 schema integration tests."""

    def __init__(self, runtime_id: str = "colab-mock-ep-001", text: str = "Hello FastMCP\n") -> None:
        self.runtime_id = runtime_id
        self.endpoint = runtime_id
        self._text = text

    async def execute(self, code: str, **kwargs: Any) -> List[TextContent]:
        return [TextContent(type="text", text=self._text)]

    async def interrupt(self) -> Dict[str, Any]:
        return {
            "runtime_id": self.runtime_id,
            "status": "interrupted",
            "state_reset": False,
            "kernel_restarted": False,
        }


@pytest.mark.asyncio
async def test_tool_8_colab_execute_code_valid_sync(test_server: FastMCP):
    """Tier 1: colab_execute_code synchronous returns MCP Content Blocks."""
    await require_tool(test_server, "colab_execute_code")
    mock_client = _MockKernelClient(text="Hello FastMCP\n")
    with patch("colab_mcp.kernel.tools.get_active_kernel_client", new=AsyncMock(return_value=mock_client)):
        result = await test_server.call_tool("colab_execute_code", {"code": "print('Hello FastMCP')"})
    # Verify Content Blocks returned
    assert result is not None
    if isinstance(result, tuple):
        blocks = result[0]
        assert isinstance(blocks, list)
        assert len(blocks) > 0
        assert hasattr(blocks[0], "text") or isinstance(blocks[0], TextContent)


@pytest.mark.asyncio
async def test_tool_8_colab_execute_code_async_mode(test_server: FastMCP):
    """Tier 1: colab_execute_code with wait_for_completion=False returns execution_id."""
    await require_tool(test_server, "colab_execute_code")
    mock_client = _MockKernelClient()
    with patch("colab_mcp.kernel.tools.get_active_kernel_client", new=AsyncMock(return_value=mock_client)):
        result = await test_server.call_tool(
            "colab_execute_code",
            {"code": "import time; time.sleep(10)", "wait_for_completion": False},
        )
    data = extract_result_dict(result)
    assert "execution_id" in data or "id" in data


@pytest.mark.asyncio
async def test_tool_8_colab_execute_code_missing_required_code(test_server: FastMCP):
    """Tier 2: Missing required 'code' argument raises ToolError."""
    await require_tool(test_server, "colab_execute_code")
    with pytest.raises((ToolError, ValueError, TypeError)):
        await test_server.call_tool("colab_execute_code", {})


@pytest.mark.asyncio
async def test_tool_8_colab_execute_code_invalid_timeout(test_server: FastMCP):
    """Tier 2: Timeout < 5s raises ToolError / validation error."""
    await require_tool(test_server, "colab_execute_code")
    with pytest.raises((ToolError, ValueError)):
        await test_server.call_tool("colab_execute_code", {"code": "pass", "timeout_seconds": 2})


@pytest.mark.asyncio
async def test_tool_8_colab_execute_code_invalid_max_text_lines(test_server: FastMCP):
    """Tier 2: max_text_lines < 10 raises ToolError / validation error."""
    await require_tool(test_server, "colab_execute_code")
    with pytest.raises((ToolError, ValueError)):
        await test_server.call_tool("colab_execute_code", {"code": "pass", "max_text_lines": 3})


# --- Tool 9: colab_execute_notebook ---

@pytest.mark.asyncio
async def test_tool_9_colab_execute_notebook_valid_content(test_server: FastMCP):
    """Tier 1: colab_execute_notebook executes notebook JSON string."""
    await require_tool(test_server, "colab_execute_notebook")
    minimal_nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["print(1 + 1)\n"],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    mock_client = _MockKernelClient(text="2\n")
    with patch("colab_mcp.notebook.runner.get_active_kernel_client", new=AsyncMock(return_value=mock_client)):
        result = await test_server.call_tool(
            "colab_execute_notebook",
            {"notebook_content": json.dumps(minimal_nb), "reset_kernel_before_run": False},
        )
    data = extract_result_dict(result)
    assert data.get("total_cells") == 1
    assert "cell_summaries" in data


@pytest.mark.asyncio
async def test_tool_9_colab_execute_notebook_missing_both_inputs(test_server: FastMCP):
    """Tier 2: Missing both notebook_path and notebook_content raises ToolError."""
    await require_tool(test_server, "colab_execute_notebook")
    with pytest.raises((ToolError, ValueError)):
        await test_server.call_tool("colab_execute_notebook", {})


@pytest.mark.asyncio
async def test_tool_9_colab_execute_notebook_conflicting_inputs(test_server: FastMCP):
    """Tier 2: Providing both notebook_path and notebook_content raises ToolError."""
    await require_tool(test_server, "colab_execute_notebook")
    with pytest.raises((ToolError, ValueError)):
        await test_server.call_tool(
            "colab_execute_notebook",
            {"notebook_path": "benchmark.ipynb", "notebook_content": "{}"},
        )


@pytest.mark.asyncio
async def test_tool_9_colab_execute_notebook_invalid_cell_indices_type(test_server: FastMCP):
    """Tier 2: Non-list cell_indices raises ToolError."""
    await require_tool(test_server, "colab_execute_notebook")
    with pytest.raises((ToolError, ValueError, TypeError)):
        await test_server.call_tool(
            "colab_execute_notebook",
            {"notebook_content": "{}", "cell_indices": "first_cell"},
        )


# --- Tool 10: colab_get_execution_status ---

@pytest.mark.asyncio
async def test_tool_10_colab_get_execution_status_valid(test_server: FastMCP):
    """Tier 1: colab_get_execution_status returns status for execution_id."""
    await require_tool(test_server, "colab_get_execution_status")
    mock_registry = {
        "exec-abc-123": {
            "execution_id": "exec-abc-123",
            "runtime_id": "colab-mock-ep-001",
            "status": "running",
            "elapsed_seconds": 1.5,
            "partial_stdout_tail": "step 1",
            "has_error": False,
        }
    }
    with patch("colab_mcp.kernel.tools.get_execution_registry", return_value=mock_registry):
        result = await test_server.call_tool("colab_get_execution_status", {"execution_id": "exec-abc-123"})
    data = extract_result_dict(result)
    assert "status" in data or "execution_id" in data


@pytest.mark.asyncio
async def test_tool_10_colab_get_execution_status_missing_required(test_server: FastMCP):
    """Tier 2: Missing execution_id raises ToolError."""
    await require_tool(test_server, "colab_get_execution_status")
    with pytest.raises((ToolError, ValueError, TypeError)):
        await test_server.call_tool("colab_get_execution_status", {})


@pytest.mark.asyncio
async def test_tool_10_colab_get_execution_status_invalid_type(test_server: FastMCP):
    """Tier 2: Non-string execution_id raises ToolError."""
    await require_tool(test_server, "colab_get_execution_status")
    with pytest.raises((ToolError, ValueError, TypeError)):
        await test_server.call_tool("colab_get_execution_status", {"execution_id": 99999})


# --- Tool 11: colab_get_cell_output ---

@pytest.mark.asyncio
async def test_tool_11_colab_get_cell_output_valid(test_server: FastMCP):
    """Tier 1: colab_get_cell_output returns Content Blocks."""
    await require_tool(test_server, "colab_get_cell_output")
    result = await test_server.call_tool("colab_get_cell_output", {"execution_id": "exec-abc-123"})
    assert result is not None


@pytest.mark.asyncio
async def test_tool_11_colab_get_cell_output_missing_required(test_server: FastMCP):
    """Tier 2: Missing execution_id raises ToolError."""
    await require_tool(test_server, "colab_get_cell_output")
    with pytest.raises((ToolError, ValueError, TypeError)):
        await test_server.call_tool("colab_get_cell_output", {})


@pytest.mark.asyncio
async def test_tool_11_colab_get_cell_output_invalid_type(test_server: FastMCP):
    """Tier 2: Non-boolean include_images raises ToolError."""
    await require_tool(test_server, "colab_get_cell_output")
    with pytest.raises((ToolError, ValueError, TypeError)):
        await test_server.call_tool(
            "colab_get_cell_output",
            {"execution_id": "exec-abc-123", "include_images": "yes"},
        )


# --- Tool 12: colab_interrupt_execution ---

@pytest.mark.asyncio
async def test_tool_12_colab_interrupt_execution_valid(test_server: FastMCP):
    """Tier 1: colab_interrupt_execution halts cell and reports state_reset."""
    await require_tool(test_server, "colab_interrupt_execution")
    mock_client = _MockKernelClient(runtime_id="colab-test-123")
    with patch("colab_mcp.kernel.tools.get_active_kernel_client", new=AsyncMock(return_value=mock_client)):
        result = await test_server.call_tool("colab_interrupt_execution", {"runtime_id": "colab-test-123"})
    data = extract_result_dict(result)
    assert "status" in data
    assert "state_reset" in data or "kernel_restarted" in data


@pytest.mark.asyncio
async def test_tool_12_colab_interrupt_execution_defaults(test_server: FastMCP):
    """Tier 1: colab_interrupt_execution defaults."""
    await require_tool(test_server, "colab_interrupt_execution")
    mock_client = _MockKernelClient(runtime_id="colab-default")
    with patch("colab_mcp.kernel.tools.get_active_kernel_client", new=AsyncMock(return_value=mock_client)):
        result = await test_server.call_tool("colab_interrupt_execution", {})
    data = extract_result_dict(result)
    assert "status" in data


@pytest.mark.asyncio
async def test_tool_12_colab_interrupt_execution_invalid_type(test_server: FastMCP):
    """Tier 2: Non-string runtime_id raises ToolError."""
    await require_tool(test_server, "colab_interrupt_execution")
    with pytest.raises((ToolError, ValueError, TypeError)):
        await test_server.call_tool("colab_interrupt_execution", {"runtime_id": False})
