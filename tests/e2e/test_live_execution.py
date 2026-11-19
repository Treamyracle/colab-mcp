"""
End-to-End Real-World Application Scenarios (Tier 4 Workloads).

Requirements:
- TEST_INFRA.md § Real-World Application Scenarios (Tier 4):
  - Scenario 1: Complete Agent ML Training Lifecycle
  - Scenario 2: Long-Running Asynchronous Execution & Polling
  - Scenario 3: Automated Notebook Benchmark & Serialization
  - Scenario 4: Interactive SIGINT Interrupt & Recovery
  - Scenario 5: Multi-Account Credential Routing & Safe Logout
  - Scenario 6: Free Tier Quota Guard & Graceful Fallback
- TECH_SPEC §5, §6, §7.3.1, §7.3.2, §7.3.3.

Adheres strictly to Python 3.13, progressive testability across milestones,
and exercises genuine FastMCP server tools and client interfaces.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest

from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ImageContent, TextContent

from colab_mcp.auth.keystore import (
    get_credentials,
    purge_credentials,
    store_credentials,
)
from colab_mcp.auth.models import AuthMethod, GoogleCredentials
from colab_mcp.server import create_server

# Progressive testability check across M2/M3/M4 execution components
try:
    import colab_mcp.runtime.client as runtime_client_mod
    import colab_mcp.runtime.models as runtime_models_mod
    import colab_mcp.kernel.client as kernel_client_mod
    import colab_mcp.notebook.runner as notebook_runner_mod
    EXECUTION_INFRA_AVAILABLE = True
except ImportError:
    runtime_client_mod = None
    runtime_models_mod = None
    kernel_client_mod = None
    notebook_runner_mod = None
    EXECUTION_INFRA_AVAILABLE = False


@pytest.fixture(autouse=True)
def require_execution_infrastructure():
    """Cleanly skip if execution subsystem (M2/M3/M4) is not yet available in codebase."""
    if not EXECUTION_INFRA_AVAILABLE:
        pytest.skip(
            "colab_mcp execution tools and runtime infrastructure are scheduled for Milestone 2/3/4 implementation"
        )


# ============================================================================
# Helpers for FastMCP Server Invocations & Output Parsing
# ============================================================================

def extract_content_text(tool_result: Any) -> str:
    """Extract string content from FastMCP call_tool return value."""
    blocks = tool_result[0] if isinstance(tool_result, tuple) else tool_result
    if isinstance(blocks, list):
        text_parts = []
        for b in blocks:
            if hasattr(b, "text"):
                text_parts.append(b.text)
            elif isinstance(b, dict) and "text" in b:
                text_parts.append(b["text"])
        if text_parts:
            return "\n".join(text_parts)
    if hasattr(tool_result, "text"):
        return tool_result.text
    return str(tool_result)


def extract_result_dict(tool_result: Any) -> Dict[str, Any]:
    """Extract structured dict from FastMCP call_tool result."""
    if isinstance(tool_result, tuple) and len(tool_result) > 1 and isinstance(tool_result[1], dict):
        return tool_result[1]
    if hasattr(tool_result, "model_dump"):
        return tool_result.model_dump()
    if isinstance(tool_result, dict):
        return tool_result
    raw_text = extract_content_text(tool_result)
    try:
        return json.loads(raw_text)
    except Exception:
        return {"raw": raw_text}


# ============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS
# ============================================================================

@pytest.mark.asyncio
async def test_scenario_1_complete_agent_ml_training_lifecycle(
    mock_keystore_dir, mock_keyring, sample_credentials
):
    """
    Scenario 1: Complete Agent ML Training Lifecycle.
    Agent Workflow:
      1. Login & Store Credentials (OAuth / Keystore)
      2. Discover Available Accelerators (colab_list_runtime_types)
      3. Provision T4 GPU VM (colab_connect_runtime)
      4. Execute PyTorch Training Pipeline (colab_execute_code)
      5. Stream Multimodal Output (TextContent logs + ImageContent loss curves)
      6. Cleanly Deallocate VM (colab_disconnect_runtime)
    """
    server = create_server("scenario-1-server")
    store_credentials(0, sample_credentials)

    # Step 1: Verify auth status
    auth_res = await server.call_tool("colab_get_auth_status", {"auth_user_index": 0})
    auth_data = extract_result_dict(auth_res)
    assert auth_data.get("is_authenticated") is True

    # Step 2: Discover Available Accelerators
    list_res = await server.call_tool("colab_list_runtime_types", {"auth_user_index": 0})
    list_data = extract_result_dict(list_res)
    assert "available_accelerators" in list_data
    acc_list = list_data["available_accelerators"]
    assert isinstance(acc_list, list) and len(acc_list) > 0
    acc_names = [a.get("accelerator") for a in acc_list]
    assert "T4" in acc_names

    # Step 3: Provision T4 GPU VM
    conn_res = await server.call_tool(
        "colab_connect_runtime",
        {"accelerator": "T4", "high_ram": False, "auth_user_index": 0},
    )
    conn_data = extract_result_dict(conn_res)
    assert conn_data.get("status") in ["connected", "active"]
    assert conn_data.get("accelerator") == "T4"
    runtime_id = conn_data.get("runtime_id", "colab-runtime-default")

    # Step 4: Execute PyTorch Training Pipeline
    training_code = (
        "import torch\n"
        "import matplotlib.pyplot as plt\n"
        "model = torch.nn.Linear(10, 2)\n"
        "# Train for 5 epochs and plot loss curve\n"
        "for epoch in range(5):\n"
        "    print(f'Epoch {epoch+1}/5 - loss: {0.8/(epoch+1):.4f}')\n"
        "plt.plot([0.8, 0.4, 0.27, 0.2, 0.16])\n"
        "plt.title('Training Loss')\n"
        "plt.show()\n"
    )
    exec_res = await server.call_tool(
        "colab_execute_code",
        {"code": training_code, "runtime_id": runtime_id},
    )

    # Step 5: Stream Multimodal Output (TextContent logs + ImageContent loss curves)
    blocks = exec_res[0] if isinstance(exec_res, tuple) else exec_res
    assert isinstance(blocks, list)
    assert len(blocks) >= 2

    # Verify chronological ordering: text output logs followed by plot image
    text_blocks = [b for b in blocks if isinstance(b, TextContent) or getattr(b, "type", None) == "text"]
    image_blocks = [b for b in blocks if isinstance(b, ImageContent) or getattr(b, "type", None) == "image"]
    assert len(text_blocks) > 0
    assert len(image_blocks) > 0

    first_text = extract_content_text(text_blocks)
    assert "loss" in first_text.lower() or "epoch" in first_text.lower()

    loss_chart = image_blocks[0]
    chart_mime = (
        getattr(loss_chart, "mime_type", None)
        or getattr(loss_chart, "mimeType", None)
        or (loss_chart.get("mimeType") if isinstance(loss_chart, dict) else None)
    )
    chart_data = getattr(loss_chart, "data", None) or (loss_chart.get("data") if isinstance(loss_chart, dict) else None)
    assert chart_mime == "image/png"
    assert chart_data is not None and len(chart_data) > 20

    # Step 6: Cleanly Deallocate VM
    disc_res = await server.call_tool(
        "colab_disconnect_runtime",
        {"runtime_id": runtime_id},
    )
    disc_data = extract_result_dict(disc_res)
    assert disc_data.get("status") in ["terminated", "disconnected", "released"]
    assert disc_data.get("compute_resources_released") is True or "message" in disc_data


@pytest.mark.asyncio
async def test_scenario_2_long_running_async_execution_and_polling(
    mock_keystore_dir, mock_keyring, sample_credentials
):
    """
    Scenario 2: Long-Running Asynchronous Execution & Polling.
    Agent Workflow:
      1. Dispatch asynchronous job (wait_for_completion=False)
      2. Receive execution_id
      3. Poll status (colab_get_execution_status)
      4. Retrieve buffered outputs with two-stage head-tail truncation (colab_get_cell_output)
    """
    server = create_server("scenario-2-server")
    store_credentials(0, sample_credentials)

    # Step 1 & 2: Dispatch async job and receive execution_id
    large_training_job = (
        "import time\n"
        "for i in range(1500):\n"
        "    print(f'Processing batch {i} of 10000')\n"
    )
    dispatch_res = await server.call_tool(
        "colab_execute_code",
        {"code": large_training_job, "wait_for_completion": False},
    )
    dispatch_data = extract_result_dict(dispatch_res)
    assert "execution_id" in dispatch_data or "id" in dispatch_data
    execution_id = dispatch_data.get("execution_id", dispatch_data.get("id"))
    assert execution_id is not None

    # Step 3: Poll status via colab_get_execution_status
    status_res = await server.call_tool(
        "colab_get_execution_status",
        {"execution_id": execution_id},
    )
    status_data = extract_result_dict(status_res)
    assert status_data.get("execution_id") == execution_id
    assert status_data.get("status") in ["running", "completed", "ok"]
    assert "has_error" in status_data
    assert status_data["has_error"] is False

    # Step 4: Retrieve buffered outputs via colab_get_cell_output
    output_res = await server.call_tool(
        "colab_get_cell_output",
        {
            "execution_id": execution_id,
            "include_images": True,
            "max_text_lines": 1000,
            "max_output_bytes": 102400,
        },
    )
    blocks = output_res[0] if isinstance(output_res, tuple) else output_res
    assert isinstance(blocks, list)
    assert len(blocks) > 0
    full_output_text = extract_content_text(output_res)
    assert "Processing batch" in full_output_text

    # Step 5: Verify two-stage head-tail truncation bounds
    bounded_res = await server.call_tool(
        "colab_get_cell_output",
        {
            "execution_id": execution_id,
            "include_images": False,
            "max_text_lines": 100,
            "max_output_bytes": 4096,
        },
    )
    bounded_text = extract_content_text(bounded_res)
    lines = bounded_text.splitlines()
    assert len(lines) <= 105
    assert len(bounded_text.encode("utf-8")) <= 8192
    if len(lines) < 1500:
        assert "[OUTPUT TRUNCATED" in bounded_text or len(lines) <= 101


@pytest.mark.asyncio
async def test_scenario_3_automated_notebook_benchmark_and_serialization(
    tmp_path: Path, mock_keystore_dir, mock_keyring, sample_credentials
):
    """
    Scenario 3: Automated Notebook Benchmark & Serialization.
    Agent Workflow:
      1. Load .ipynb notebook file
      2. Restart kernel prior to run (reset_kernel_before_run=True)
      3. Execute specific cell indices [0, 1]
      4. Handle failure in cell 1 with stop_on_error=True
      5. Trap error diagnostics in cell_summaries
      6. Atomically serialize updated notebook with outputs to disk
    """
    server = create_server("scenario-3-server")
    store_credentials(0, sample_credentials)

    notebook_file = tmp_path / "benchmark.ipynb"
    output_file = tmp_path / "benchmark_result.ipynb"

    sample_notebook_data = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["x = 100\nprint(f'Cell 0 executed: {x}')"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["y = x / 0\nprint('Will not reach here')"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["z = 200\nprint('Downstream cell')"],
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_file.write_text(json.dumps(sample_notebook_data), encoding="utf-8")

    # Execute notebook via FastMCP colab_execute_notebook tool
    exec_res = await server.call_tool(
        "colab_execute_notebook",
        {
            "notebook_path": str(notebook_file),
            "reset_kernel_before_run": True,
            "stop_on_error": True,
            "cell_indices": [0, 1, 2],
            "output_save_path": str(output_file),
        },
    )
    result_data = extract_result_dict(exec_res)

    # Step 4 & 5: Trap error diagnostics in cell_summaries
    assert result_data.get("failed_cells") == 1
    assert result_data.get("skipped_cells", 0) >= 1
    assert result_data.get("executed_cells") == 2
    assert result_data.get("kernel_reset_performed") is True

    summaries = result_data.get("cell_summaries", [])
    assert len(summaries) >= 2
    assert summaries[0].get("status") == "success"
    assert summaries[1].get("status") == "failed"
    assert summaries[1].get("error_name") == "ZeroDivisionError"

    # Step 6: Verify atomically serialized notebook with outputs on disk
    assert output_file.exists()
    serialized_nb = json.loads(output_file.read_text(encoding="utf-8"))
    assert serialized_nb["nbformat"] == 4
    assert len(serialized_nb["cells"]) == 3

    # Cell 0 has stream output
    cell_0_outs = serialized_nb["cells"][0]["outputs"]
    assert len(cell_0_outs) > 0
    assert any("Cell 0 executed" in str(out) for out in cell_0_outs)

    # Cell 1 has error output
    cell_1_outs = serialized_nb["cells"][1]["outputs"]
    assert len(cell_1_outs) > 0
    assert any(out.get("output_type") == "error" or out.get("ename") == "ZeroDivisionError" for out in cell_1_outs)

    # Cell 2 was stopped by stop_on_error policy
    cell_2_outs = serialized_nb["cells"][2]["outputs"]
    assert len(cell_2_outs) == 0 or serialized_nb["cells"][2]["execution_count"] is None


@pytest.mark.asyncio
async def test_scenario_4_interactive_sigint_interrupt_and_recovery(
    mock_keystore_dir, mock_keyring, sample_credentials
):
    """
    Scenario 4: Interactive SIGINT Interrupt & Recovery.
    Agent Workflow:
      1. Execute initialization code setting in-memory state
      2. Start infinite loop code snippet (while True: pass)
      3. Issue colab_interrupt_execution
      4. Verify SIGINT was non-destructive (state_reset: false, kernel_restarted: false)
      5. Verify in-memory variables preserved
      6. Resume execution confirming runtime continuity
    """
    server = create_server("scenario-4-server")
    store_credentials(0, sample_credentials)

    # Step 1: Execute initialization code setting in-memory state
    init_res = await server.call_tool(
        "colab_execute_code",
        {"code": "state_val = 42\nprint(f'state_val initialized to {state_val}')"},
    )
    init_text = extract_content_text(init_res)
    assert "42" in init_text

    # Step 2 & 3: Start infinite loop and issue colab_interrupt_execution
    loop_task = asyncio.create_task(
        server.call_tool(
            "colab_execute_code",
            {"code": "import time\nwhile True:\n    time.sleep(0.05)", "timeout_seconds": 60},
        )
    )
    await asyncio.sleep(0.1)

    interrupt_res = await server.call_tool(
        "colab_interrupt_execution",
        {},
    )
    interrupt_data = extract_result_dict(interrupt_res)

    # Step 4: Verify non-destructive SIGINT (state_reset: false, kernel_restarted: false)
    assert interrupt_data.get("status") in ["interrupted", "restarted"]
    assert interrupt_data.get("state_reset") is False
    assert interrupt_data.get("kernel_restarted") is False

    await asyncio.gather(loop_task, return_exceptions=True)

    # Step 5 & 6: Resume execution confirming runtime continuity and preserved state
    verify_res = await server.call_tool(
        "colab_execute_code",
        {"code": "print(f'state_val: {state_val}')"},
    )
    verify_text = extract_content_text(verify_res)
    assert "state_val: 42" in verify_text


@pytest.mark.asyncio
async def test_scenario_5_multi_account_credential_routing_and_safe_logout(
    mock_keystore_dir, mock_keyring, sample_credentials
):
    """
    Scenario 5: Multi-Account Credential Routing & Safe Logout.
    Agent Workflow:
      1. Store credentials for auth_user_index=0 and auth_user_index=1
      2. Route requests with authuser=1 / X-Goog-AuthUser: 1
      3. Call colab_logout with auth_user_index=1 and terminate_active_runtimes=True
      4. Verify account 1 is purged and runtimes released while account 0 remains active
    """
    server = create_server("scenario-5-server")

    creds_0 = sample_credentials.model_copy(
        update={"auth_user_index": 0, "user_email": "account0@gmail.com"}
    )
    creds_1 = sample_credentials.model_copy(
        update={
            "auth_user_index": 1,
            "user_email": "account1@gmail.com",
            "access_token": "ya29.user1_unique_token_routed",
        }
    )

    store_credentials(0, creds_0)
    store_credentials(1, creds_1)

    # Step 1: Verify both accounts exist in isolated keystore
    assert get_credentials(0).user_email == "account0@gmail.com"
    assert get_credentials(1).user_email == "account1@gmail.com"

    # Step 2: Route status checks per account index
    status_0 = await server.call_tool("colab_get_auth_status", {"auth_user_index": 0})
    data_0 = extract_result_dict(status_0)
    assert data_0["is_authenticated"] is True
    assert data_0["auth_user_index"] == 0
    assert data_0["user_email"] == "account0@gmail.com"

    status_1 = await server.call_tool("colab_get_auth_status", {"auth_user_index": 1})
    data_1 = extract_result_dict(status_1)
    assert data_1["is_authenticated"] is True
    assert data_1["auth_user_index"] == 1
    assert data_1["user_email"] == "account1@gmail.com"

    # Step 3: Logout account 1 only with terminate_active_runtimes=True
    logout_res = await server.call_tool(
        "colab_logout",
        {"auth_user_index": 1, "terminate_active_runtimes": True},
    )
    logout_data = extract_result_dict(logout_res)
    assert logout_data["status"] == "logged_out"
    assert logout_data["auth_user_index"] == 1
    assert logout_data["keystore_purged"] is True

    # Step 4: Account 1 must be purged; Account 0 must remain completely intact
    assert get_credentials(1) is None
    assert get_credentials(0) is not None
    assert get_credentials(0).user_email == "account0@gmail.com"

    # Confirm auth status tool reflects the purged state for user 1 but intact for user 0
    status_1_after = await server.call_tool("colab_get_auth_status", {"auth_user_index": 1})
    data_1_after = extract_result_dict(status_1_after)
    assert data_1_after["is_authenticated"] is False

    status_0_after = await server.call_tool("colab_get_auth_status", {"auth_user_index": 0})
    data_0_after = extract_result_dict(status_0_after)
    assert data_0_after["is_authenticated"] is True


@pytest.mark.asyncio
async def test_scenario_6_free_tier_quota_guard_and_graceful_fallback(
    mock_keystore_dir, mock_keyring, sample_credentials
):
    """
    Scenario 6: Free Tier Quota Guard & Graceful Fallback.
    Agent Workflow:
      1. Agent requests High-RAM on Free Tier via colab_connect_runtime(high_ram=True)
      2. Intercepts HighRamNotAvailableError / ToolError
      3. Gracefully falls back to standard RAM shape (high_ram=False)
      4. Successfully establishes standard runtime session
    """
    server = create_server("scenario-6-server")
    store_credentials(0, sample_credentials)

    # Resolve HighRamNotAvailableError class dynamically if imported
    HighRamNotAvailableError = getattr(
        runtime_client_mod,
        "HighRamNotAvailableError",
        getattr(runtime_models_mod, "HighRamNotAvailableError", Exception),
    )

    attempted_high_ram = True
    session_established = False

    # Step 1 & 2: Request High-RAM on Free Tier; must fail with HighRamNotAvailableError or ToolError
    try:
        res = await server.call_tool(
            "colab_connect_runtime",
            {"accelerator": "NONE", "high_ram": True, "auth_user_index": 0},
        )
        data = extract_result_dict(res)
        if "error" in data or data.get("is_error") or data.get("isError") or "HighRamNotAvailableError" in str(data):
            raise ValueError(f"HighRamNotAvailableError: {data}")
        pytest.fail("colab_connect_runtime(high_ram=True) must fail on Free Tier accounts")
    except (ToolError, HighRamNotAvailableError, ValueError, Exception) as exc:
        err_str = str(exc).lower()
        assert "highram" in err_str or "high-ram" in err_str or "tier" in err_str or "not available" in err_str or "pro" in err_str
        # Step 3: Graceful fallback policy - request standard RAM
        attempted_high_ram = False

    # Step 4: Graceful fallback to high_ram=False succeeds
    fallback_res = await server.call_tool(
        "colab_connect_runtime",
        {"accelerator": "NONE", "high_ram": False, "auth_user_index": 0},
    )
    conn_data = extract_result_dict(fallback_res)
    assert conn_data.get("status") in ["connected", "active"]
    assert conn_data.get("high_ram") is False
    session_established = True

    assert attempted_high_ram is False
    assert session_established is True
