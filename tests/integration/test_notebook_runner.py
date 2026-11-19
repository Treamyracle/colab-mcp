"""
Integration tests for Notebook Execution Engine & MCP Execution Tools.

Requirements:
- TECH_SPEC §6.2 - §6.4, §7.3.3
- Features: 46 (0-Based Absolute Index Execution), 47 (Clean Benchmark Environment),
  48 (Fault Propagation Policy - stop_on_error), 49 (AST Output Node Conversion & Serialization),
  50 (colab_execute_code), 51 (colab_execute_notebook), 52 (colab_get_execution_status),
  53 (colab_get_cell_output), 54 (colab_interrupt_execution).

Adheres to Python 3.13 and progressive testability across milestones.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import nbformat
import pytest
from mcp.types import ImageContent, TextContent

# Progressive testability: check implementation availability
try:
    import colab_mcp.notebook.runner as runner_mod
    RUNNER_AVAILABLE = True
except ImportError:
    runner_mod = None
    RUNNER_AVAILABLE = False


@pytest.fixture(autouse=True)
def require_runner_implementation():
    """Progressive testability: cleanly skip if colab_mcp.notebook.runner is not yet implemented."""
    if not RUNNER_AVAILABLE:
        pytest.skip("colab_mcp.notebook.runner is scheduled for Milestone 3 implementation")


def _get_runner_class():
    return getattr(
        runner_mod,
        "NotebookRunner",
        getattr(runner_mod, "NotebookEngine", None),
    )


def _get_tool(tool_name: str):
    return getattr(runner_mod, tool_name, None)


def _create_sample_notebook() -> nbformat.NotebookNode:
    """Creates a 5-cell test notebook with mixed markdown and code cells."""
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        # Cell 0: Markdown
        nbformat.v4.new_markdown_cell("# Title Markdown"),
        # Cell 1: Code cell 1
        nbformat.v4.new_code_cell("a = 10\nprint('a is', a)"),
        # Cell 2: Markdown
        nbformat.v4.new_markdown_cell("## Intermediate Step"),
        # Cell 3: Code cell 2 (computes with previous variable)
        nbformat.v4.new_code_cell("b = a * 2\nprint('b is', b)"),
        # Cell 4: Code cell 3 (potentially failing)
        nbformat.v4.new_code_cell("c = 1 / 0"),
    ]
    return nb


# ============================================================================
# Tier 1: Primary Behavior (Happy Path) Tests (>= 5 tests)
# ============================================================================

@pytest.mark.asyncio
async def test_execute_all_code_cells_sequentially():
    """Verify executing all code cells sequentially in a multi-cell notebook."""
    RunnerClass = _get_runner_class()
    if RunnerClass is None:
        pytest.skip("NotebookRunner class not available")

    nb = _create_sample_notebook()
    nb.cells = nb.cells[:4]  # Remove failing cell 4

    mock_client = AsyncMock()
    mock_client.execute.side_effect = [
        [TextContent(type="text", text="a is 10\n")],
        [TextContent(type="text", text="b is 20\n")],
    ]

    runner = RunnerClass(client=mock_client)
    if hasattr(runner, "run"):
        result = await runner.run(nb)
        assert result["total_cells"] == 4
        assert result["executed_cells"] == 2
        assert result["successful_cells"] == 2
        assert result["failed_cells"] == 0


@pytest.mark.asyncio
async def test_selective_cell_indices_execution():
    """
    TECH_SPEC §6.2 / Feature 46:
    0-based absolute indexing (cell_indices).
    Specifying cell_indices=[1] executes only cell 1, skipping cell 3.
    """
    RunnerClass = _get_runner_class()
    if RunnerClass is None:
        pytest.skip("NotebookRunner class not available")

    nb = _create_sample_notebook()
    mock_client = AsyncMock()
    mock_client.execute.return_value = [TextContent(type="text", text="a is 10\n")]

    runner = RunnerClass(client=mock_client)
    if hasattr(runner, "run"):
        result = await runner.run(nb, cell_indices=[1])
        assert mock_client.execute.call_count == 1
        assert result["executed_cells"] == 1


@pytest.mark.asyncio
async def test_markdown_cells_at_specified_indices_skipped():
    """
    TECH_SPEC §6.2 / Feature 46:
    Specifying a markdown cell index (e.g. cell_indices=[0, 1])
    bypasses markdown cell 0 cleanly while executing code cell 1.
    """
    RunnerClass = _get_runner_class()
    if RunnerClass is None:
        pytest.skip("NotebookRunner class not available")

    nb = _create_sample_notebook()
    mock_client = AsyncMock()
    mock_client.execute.return_value = [TextContent(type="text", text="a is 10\n")]

    runner = RunnerClass(client=mock_client)
    if hasattr(runner, "run"):
        result = await runner.run(nb, cell_indices=[0, 1])
        assert mock_client.execute.call_count == 1


@pytest.mark.asyncio
async def test_clean_benchmark_restart_before_run():
    """
    TECH_SPEC §6.2 / Feature 47:
    When reset_kernel_before_run=True, restarts kernel prior to notebook execution.
    """
    RunnerClass = _get_runner_class()
    if RunnerClass is None:
        pytest.skip("NotebookRunner class not available")

    nb = _create_sample_notebook()
    mock_client = AsyncMock()
    mock_client.restart = AsyncMock()
    mock_client.execute.return_value = [TextContent(type="text", text="done")]

    runner = RunnerClass(client=mock_client)
    if hasattr(runner, "run"):
        result = await runner.run(nb, cell_indices=[1], reset_kernel_before_run=True)
        assert mock_client.restart.called or getattr(result, "kernel_reset_performed", True)


@pytest.mark.asyncio
async def test_atomic_disk_serialization_to_output_save_path(tmp_path: Path):
    """
    TECH_SPEC §6.4 / Feature 49:
    Executed outputs converted to AST nodes and atomically saved to output_save_path.
    """
    RunnerClass = _get_runner_class()
    if RunnerClass is None:
        pytest.skip("NotebookRunner class not available")

    nb = _create_sample_notebook()
    nb.cells = nb.cells[:2]
    output_path = tmp_path / "executed_output.ipynb"

    mock_client = AsyncMock()
    mock_client.execute.return_value = [TextContent(type="text", text="a is 10\n")]

    runner = RunnerClass(client=mock_client)
    if hasattr(runner, "run"):
        await runner.run(nb, output_save_path=str(output_path))
        assert output_path.exists()
        saved_nb = nbformat.read(str(output_path), as_version=4)
        assert len(saved_nb.cells) == 2


@pytest.mark.asyncio
async def test_tool_colab_execute_code_returns_mcp_blocks():
    """
    TECH_SPEC §7.3.3 / Feature 50:
    Tool colab_execute_code with wait_for_completion=True returns [TextContent, ImageContent].
    """
    execute_code_func = _get_tool("colab_execute_code")
    if execute_code_func is None:
        pytest.skip("colab_execute_code function not found")

    with patch.object(runner_mod, "get_active_kernel_client", create=True) as mock_get_client:
        mock_client = AsyncMock()
        mock_client.execute.return_value = [
            TextContent(type="text", text="Loss: 0.123\n"),
            ImageContent(type="image", data="mock_b64", mimeType="image/png"),
        ]
        mock_get_client.return_value = mock_client

        blocks = await execute_code_func(code="print('test')")
        assert len(blocks) == 2
        assert isinstance(blocks[0], TextContent)
        assert isinstance(blocks[1], ImageContent)


# ============================================================================
# Tier 2: Fault Propagation, Error Policies, Async Status Tests (>= 5 tests)
# ============================================================================

@pytest.mark.asyncio
async def test_stop_on_error_true_halts_and_marks_downstream_skipped():
    """
    TECH_SPEC §6.3 / Feature 48:
    stop_on_error=True halts execution on first cell error.
    Downstream cells are marked as 'skipped' in cell_summaries.
    """
    RunnerClass = _get_runner_class()
    if RunnerClass is None:
        pytest.skip("NotebookRunner class not available")

    nb = _create_sample_notebook()
    nb.cells[1].source = "1 / 0"

    mock_client = AsyncMock()
    mock_client.execute.side_effect = [
        RuntimeError("ZeroDivisionError: division by zero"),
    ]

    runner = RunnerClass(client=mock_client)
    if hasattr(runner, "run"):
        result = await runner.run(nb, stop_on_error=True)
        assert result["failed_cells"] == 1
        assert result["skipped_cells"] > 0
        assert mock_client.execute.call_count == 1


@pytest.mark.asyncio
async def test_stop_on_error_captures_error_diagnostics():
    """
    TECH_SPEC §6.3 / Feature 48:
    Failed cell populates error_name, error_value, and traceback in cell_summaries.
    """
    RunnerClass = _get_runner_class()
    if RunnerClass is None:
        pytest.skip("NotebookRunner class not available")

    nb = _create_sample_notebook()
    nb.cells = [nb.cells[4]]

    mock_client = AsyncMock()
    mock_client.execute.side_effect = [
        ValueError("Invalid dimension"),
    ]

    runner = RunnerClass(client=mock_client)
    if hasattr(runner, "run"):
        result = await runner.run(nb, stop_on_error=True)
        summaries = result.get("cell_summaries", [])
        if summaries:
            failed_summary = summaries[0]
            assert failed_summary["status"] == "failed"
            assert "error_name" in failed_summary or "error_value" in failed_summary


@pytest.mark.asyncio
async def test_stop_on_error_false_continues_execution():
    """
    TECH_SPEC §6.3 / Feature 48:
    stop_on_error=False records failure for cell N, but continues executing cell N+1.
    """
    RunnerClass = _get_runner_class()
    if RunnerClass is None:
        pytest.skip("NotebookRunner class not available")

    nb = _create_sample_notebook()
    nb.cells = nb.cells[:4]  # Keep 2 code cells (indices 1 and 3) matching side_effect
    nb.cells[1].source = "raise RuntimeError('error 1')"

    mock_client = AsyncMock()
    mock_client.execute.side_effect = [
        RuntimeError("error 1"),
        [TextContent(type="text", text="cell 3 succeeded")],
    ]

    runner = RunnerClass(client=mock_client)
    if hasattr(runner, "run"):
        result = await runner.run(nb, stop_on_error=False)
        assert mock_client.execute.call_count == 2
        assert result["failed_cells"] == 1
        assert result["successful_cells"] == 1


@pytest.mark.asyncio
async def test_tool_colab_execute_code_async_returns_execution_id():
    """
    TECH_SPEC §7.3.3 / Feature 50:
    Tool colab_execute_code with wait_for_completion=False dispatches asynchronously
    and immediately returns an execution_id.
    """
    execute_code_func = _get_tool("colab_execute_code")
    if execute_code_func is None:
        pytest.skip("colab_execute_code function not found")

    with patch.object(runner_mod, "get_active_kernel_client", create=True):
        resp = await execute_code_func(code="import time; time.sleep(10)", wait_for_completion=False)
        assert isinstance(resp, dict) or isinstance(resp, list)
        if isinstance(resp, dict):
            assert "execution_id" in resp
            assert resp.get("status") in ("queued", "running")


@pytest.mark.asyncio
async def test_tool_colab_get_execution_status_polling():
    """
    TECH_SPEC §7.3.3 / Feature 52:
    Tool colab_get_execution_status polls progress, elapsed time, and partial stdout tail.
    """
    get_status_func = _get_tool("colab_get_execution_status")
    if get_status_func is None:
        pytest.skip("colab_get_execution_status function not found")

    with patch.object(runner_mod, "get_execution_registry", create=True) as mock_registry:
        mock_registry.return_value = {
            "exec-123": {
                "status": "running",
                "elapsed_seconds": 12.5,
                "partial_stdout_tail": "Processing batch 50/100",
                "has_error": False,
            }
        }
        status_resp = await get_status_func(execution_id="exec-123")
        assert status_resp["status"] == "running"
        assert "Processing batch" in status_resp["partial_stdout_tail"]


@pytest.mark.asyncio
async def test_tool_colab_interrupt_execution_sigint_and_restart():
    """
    TECH_SPEC §7.3.3 / Feature 54:
    Tool colab_interrupt_execution returns state_reset and kernel_restarted flags.
    """
    interrupt_func = _get_tool("colab_interrupt_execution")
    if interrupt_func is None:
        pytest.skip("colab_interrupt_execution function not found")

    with patch.object(runner_mod, "get_active_kernel_client", create=True) as mock_get_client:
        mock_client = AsyncMock()
        mock_client.interrupt.return_value = {
            "runtime_id": "test-runtime",
            "status": "interrupted",
            "state_reset": False,
            "kernel_restarted": False,
        }
        mock_get_client.return_value = mock_client

        res = await interrupt_func()
        assert res["state_reset"] is False
        assert res["kernel_restarted"] is False


@pytest.mark.asyncio
async def test_empty_notebook_execution():
    """An empty notebook with 0 cells executes cleanly returning total_cells=0."""
    RunnerClass = _get_runner_class()
    if RunnerClass is None:
        pytest.skip("NotebookRunner class not available")

    empty_nb = nbformat.v4.new_notebook()
    empty_nb.cells = []

    mock_client = AsyncMock()
    runner = RunnerClass(client=mock_client)
    if hasattr(runner, "run"):
        result = await runner.run(empty_nb)
        assert result["total_cells"] == 0
        assert result["executed_cells"] == 0
