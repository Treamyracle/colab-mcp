"""
Comprehensive unit tests for full-parity colab-mcp CLI:
- Token-free CLI session persistence & TFE auto-reattach (`cli_session.py`)
- Semantic exit codes, dual human/JSON output, & plot image saving (`cli_output.py`)
- MCP host auto-configurator (`cli_setup.py`)
- All 24 flat Typer CLI commands (`cli.py`)
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import ImageContent, TextContent
from typer.testing import CliRunner

from colab_mcp.auth.models import AuthenticationRequiredError
from colab_mcp.cli import app as cli_app
from colab_mcp.cli_output import CLIExitCode, classify_exception, save_image_blocks
from colab_mcp.cli_session import (
    CLISessionState,
    clear_cli_session,
    ensure_cli_runtime_attached,
    get_cli_session_file,
    load_cli_session,
    save_cli_session,
)
from colab_mcp.cli_setup import configure_mcp_clients, resolve_client_config_path
from colab_mcp.files.models import (
    DeleteFileResponse,
    DownloadFileResponse,
    DriveFileEntry,
    DriveListResponse,
    DriveTransferResponse,
    ListFilesResponse,
    RemoteFileEntry,
    UploadFileResponse,
)
from colab_mcp.kernel.models import KernelExecutionTimeoutError, StdinInputRequiredError
from colab_mcp.notebook.runner import CellExecutionSummary, NotebookExecutionResponse
from colab_mcp.runtime.models import (
    Assignment,
    CloudCapacityError,
    ComputeUnitsExhaustedError,
    ConnectRuntimeResponse,
    DisconnectRuntimeResponse,
    GpuMetric,
    HighRamNotAvailableError,
    RuntimeDisconnectedError,
    RuntimeStatusResponse,
    SystemMetricsResponse,
)
from colab_mcp.runtime.session_manager import get_session_manager, reset_session_manager

SAMPLE_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.fixture(autouse=True)
def _reset_mgr():
    reset_session_manager()
    yield
    reset_session_manager()


# ============================================================================
# 1. CLI Session Persistence & TFE Auto-Reattach Tests
# ============================================================================


def test_cli_session_save_load_clear_without_tokens(mock_keystore_dir: Path):
    state = CLISessionState(
        runtime_id="ep-gpu-t4-01",
        kernel_id="kern-123",
        accelerator="T4",
        variant="GPU",
        high_ram=False,
        auth_user_index=1,
    )
    saved_path = save_cli_session(state)
    assert saved_path == get_cli_session_file()
    assert saved_path.is_file()

    raw_text = saved_path.read_text(encoding="utf-8")
    assert "proxy_token" not in raw_text
    assert "access_token" not in raw_text
    assert "SAPISID" not in raw_text

    loaded = load_cli_session()
    assert loaded is not None
    assert loaded.runtime_id == "ep-gpu-t4-01"
    assert loaded.kernel_id == "kern-123"
    assert loaded.accelerator == "T4"
    assert loaded.auth_user_index == 1

    # Mismatched runtime_id does not clear
    assert clear_cli_session(runtime_id="other-ep") is False
    assert get_cli_session_file().is_file()

    assert clear_cli_session(runtime_id="ep-gpu-t4-01") is True
    assert not get_cli_session_file().is_file()
    assert load_cli_session() is None


@pytest.mark.asyncio
async def test_ensure_cli_runtime_attached_reattaches_from_saved_pointer(mock_keystore_dir: Path):
    save_cli_session(
        CLISessionState(
            runtime_id="ep-reattach-1",
            kernel_id="kern-saved",
            accelerator="L4",
            variant="GPU",
            high_ram=True,
            auth_user_index=0,
        )
    )
    remote_assignment = Assignment(
        endpoint="ep-reattach-1",
        accelerator="L4",
        variant="GPU",
        runtime_proxy_url="https://mock-proxy.local/tun/m/ep-reattach-1",
        runtime_proxy_token="fresh-tfe-jwt-token",
        token_expires_in=3600,
    )
    mock_tfe = MagicMock()
    mock_tfe.__aenter__ = AsyncMock(return_value=mock_tfe)
    mock_tfe.__aexit__ = AsyncMock(return_value=None)
    mock_tfe.list_assignments = AsyncMock(return_value=[remote_assignment])

    with (
        patch("colab_mcp.cli_session.ColabTFEClient", return_value=mock_tfe),
        patch("colab_mcp.runtime.tools._probe_kernel_status", new=AsyncMock(return_value=("kern-live", "idle"))),
    ):
        endpoint, user_idx = await ensure_cli_runtime_attached()

    assert endpoint == "ep-reattach-1"
    assert user_idx == 0
    rec = get_session_manager().get_session("ep-reattach-1")
    assert rec is not None
    assert rec.kernel_id == "kern-live"
    assert rec.keepalive_daemon is None


@pytest.mark.asyncio
async def test_ensure_cli_runtime_attached_clears_stale_pointer_when_expired(mock_keystore_dir: Path):
    save_cli_session(CLISessionState(runtime_id="ep-expired"))
    mock_tfe = MagicMock()
    mock_tfe.__aenter__ = AsyncMock(return_value=mock_tfe)
    mock_tfe.__aexit__ = AsyncMock(return_value=None)
    mock_tfe.list_assignments = AsyncMock(return_value=[])

    with patch("colab_mcp.cli_session.ColabTFEClient", return_value=mock_tfe):
        with pytest.raises(RuntimeDisconnectedError):
            await ensure_cli_runtime_attached()

    assert not get_cli_session_file().is_file()


# ============================================================================
# 2. Output Formatting, Plot Saving & Semantic Exit Codes
# ============================================================================


def test_semantic_exit_code_classification():
    assert classify_exception(AuthenticationRequiredError("login")) == CLIExitCode.AUTH_REQUIRED
    assert classify_exception(HighRamNotAvailableError("pro")) == CLIExitCode.QUOTA_OR_CAPACITY
    assert classify_exception(ComputeUnitsExhaustedError("ccu")) == CLIExitCode.QUOTA_OR_CAPACITY
    assert classify_exception(CloudCapacityError("503")) == CLIExitCode.QUOTA_OR_CAPACITY
    assert classify_exception(RuntimeDisconnectedError("gone")) == CLIExitCode.RUNTIME_DISCONNECTED
    assert classify_exception(KernelExecutionTimeoutError("timeout")) == CLIExitCode.EXECUTION_ERROR
    assert classify_exception(StdinInputRequiredError("prompt")) == CLIExitCode.EXECUTION_ERROR
    assert classify_exception(ValueError("bad arg")) == CLIExitCode.USAGE_ERROR
    assert classify_exception(RuntimeError("generic")) == CLIExitCode.ERROR


def test_save_image_blocks_decodes_png_and_jpeg(tmp_path: Path):
    blocks = [
        TextContent(type="text", text="hello"),
        ImageContent(type="image", data=SAMPLE_PNG_B64, mime_type="image/png"),
        ImageContent(type="image", data=SAMPLE_PNG_B64, mime_type="image/jpeg"),
    ]
    saved = save_image_blocks(blocks, tmp_path / "plots")
    assert len(saved) == 2
    assert saved[0].name == "plot_001.png"
    assert saved[1].name == "plot_002.jpg"
    assert saved[0].read_bytes() == base64.b64decode(SAMPLE_PNG_B64)


# ============================================================================
# 3. MCP Host Auto-Configurator (`setup-mcp`)
# ============================================================================


def test_setup_mcp_configure_dry_run_preserves_existing_and_removes(tmp_path: Path):
    cfg = tmp_path / "cursor" / "mcp.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        json.dumps({"mcpServers": {"other-tool": {"command": "npx", "args": ["foo"]}}}, indent=2),
        encoding="utf-8",
    )

    dry = configure_mcp_clients(
        client="cursor",
        config_path=str(cfg),
        command_mode="uvx",
        dry_run=True,
    )
    assert dry.status == "dry_run"
    assert dry.results[0].action == "added"
    # File untouched on dry_run
    assert "colab" not in json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]

    applied = configure_mcp_clients(
        client="cursor",
        config_path=str(cfg),
        command_mode="uvx",
        dry_run=False,
    )
    assert applied.status == "configured"
    written = json.loads(cfg.read_text(encoding="utf-8"))
    assert "other-tool" in written["mcpServers"]
    assert written["mcpServers"]["colab"]["command"] == "uvx"

    removed = configure_mcp_clients(
        client="cursor",
        config_path=str(cfg),
        remove=True,
    )
    assert removed.results[0].action == "removed"
    after_rm = json.loads(cfg.read_text(encoding="utf-8"))
    assert "colab" not in after_rm["mcpServers"]
    assert "other-tool" in after_rm["mcpServers"]


def test_setup_mcp_vscode_uses_servers_key(tmp_path: Path):
    vsc_cfg = tmp_path / "vscode_mcp.json"
    res = configure_mcp_clients(
        client="vscode",
        config_path=str(vsc_cfg),
        command_mode="python",
    )
    assert res.results[0].root_key == "servers"
    data = json.loads(vsc_cfg.read_text(encoding="utf-8"))
    assert data["servers"]["colab"]["type"] == "stdio"


def test_setup_mcp_cli_dry_run(tmp_path: Path):
    runner = CliRunner()
    cfg = tmp_path / "claude.json"
    res = runner.invoke(
        cli_app,
        ["setup-mcp", "--client", "claude-desktop", "--config-path", str(cfg), "--dry-run", "--json"],
    )
    assert res.exit_code == 0
    payload = json.loads(res.stdout)
    assert payload["status"] == "dry_run"
    assert payload["results"][0]["client"] == "claude-desktop"
    assert not cfg.exists()


# ============================================================================
# 4. Flat CLI Command Coverage & MCP Parity
# ============================================================================


def test_cli_24_commands_registered():
    expected_commands = {
        "version",
        "run",
        "setup-mcp",
        "login",
        "status",
        "logout",
        "set-secret",
        "list-secrets",
        "delete-secret",
        "list-runtimes",
        "list-sessions",
        "connect-runtime",
        "runtime-status",
        "runtime-url",
        "system-metrics",
        "disconnect",
        "execute-code",
        "job",
        "execute-notebook",
        "interrupt",
        "list-files",
        "upload-file",
        "download-file",
        "delete-file",
        "drive-list",
        "drive-download",
        "drive-upload",
        "ssh",
    }
    registered = {cmd.name for cmd in cli_app.registered_commands if cmd.name}
    assert expected_commands.issubset(registered)


def test_cli_connect_runtime_persists_session_and_disconnect_clears(mock_keystore_dir: Path):
    runner = CliRunner()
    conn_resp = ConnectRuntimeResponse(
        runtime_id="ep-cli-conn-1",
        status="connected",
        accelerator="T4",
        variant="GPU",
        high_ram=False,
        reused_existing=False,
        kernel_id="kern-abc",
        kernel_state="idle",
    )
    disc_resp = DisconnectRuntimeResponse(
        runtime_id="ep-cli-conn-1",
        status="disconnected",
        terminated_runtime_ids=["ep-cli-conn-1"],
        terminated_count=1,
        message="Disconnected.",
    )

    with patch("colab_mcp.runtime.tools.connect_runtime_impl", new=AsyncMock(return_value=conn_resp)):
        res_conn = runner.invoke(cli_app, ["connect-runtime", "--accelerator", "T4", "--json"])

    assert res_conn.exit_code == 0
    assert json.loads(res_conn.stdout)["runtime_id"] == "ep-cli-conn-1"
    saved = load_cli_session()
    assert saved is not None and saved.runtime_id == "ep-cli-conn-1"

    with patch("colab_mcp.runtime.tools.disconnect_runtime_impl", new=AsyncMock(return_value=disc_resp)) as mock_disc:
        res_disc = runner.invoke(cli_app, ["disconnect", "--json"])

    assert res_disc.exit_code == 0
    mock_disc.assert_awaited_once_with(runtime_id="ep-cli-conn-1", disconnect_all=False, auth_user_index=0)
    assert load_cli_session() is None


def test_cli_runtime_status_system_metrics_and_interrupt(mock_keystore_dir: Path):
    runner = CliRunner()
    rt_resp = RuntimeStatusResponse(
        runtime_id="ep-cli-1",
        status="active",
        kernel_state="idle",
        metrics={"accelerator": "L4", "high_ram": True},
    )
    metrics_resp = SystemMetricsResponse(
        runtime_id="ep-cli-1",
        gpus=[
            GpuMetric(
                index=0,
                name="NVIDIA L4",
                gpu_utilization_percent=60.0,
                vram_used_mb=8192.0,
                vram_total_mb=23034.0,
                temperature_celsius=62.0,
            )
        ],
        ram_total_mb=54272.0,
        ram_used_mb=10240.0,
        ram_free_mb=44032.0,
        ram_utilization_percent=18.9,
        disk_path="/content",
        disk_total_gb=100.0,
        disk_used_gb=20.0,
        disk_free_gb=80.0,
        disk_utilization_percent=20.0,
    )
    intr_resp = {
        "runtime_id": "ep-cli-1",
        "status": "interrupted",
        "kernel_state": "idle",
        "kernel_restarted": False,
        "message": "SIGINT dispatched",
    }

    with (
        patch("colab_mcp.cli.ensure_cli_runtime_attached", new=AsyncMock(return_value=("ep-cli-1", 0))),
        patch("colab_mcp.runtime.tools.get_runtime_status_impl", new=AsyncMock(return_value=rt_resp)),
        patch("colab_mcp.runtime.tools.get_system_metrics_impl", new=AsyncMock(return_value=metrics_resp)),
        patch("colab_mcp.kernel.tools.interrupt_execution_impl", new=AsyncMock(return_value=intr_resp)),
    ):
        r1 = runner.invoke(cli_app, ["runtime-status"])
        r2 = runner.invoke(cli_app, ["system-metrics"])
        r3 = runner.invoke(cli_app, ["system-metrics", "--json"])
        r4 = runner.invoke(cli_app, ["interrupt", "--json"])

    assert r1.exit_code == 0 and "ep-cli-1" in r1.stdout
    assert r2.exit_code == 0 and "NVIDIA L4" in r2.stdout
    assert r3.exit_code == 0 and json.loads(r3.stdout)["gpus"][0]["name"] == "NVIDIA L4"
    assert r4.exit_code == 0 and json.loads(r4.stdout)["status"] == "interrupted"


def test_cli_execute_code_inline_file_pipe_save_images_and_errors(tmp_path: Path):
    runner = CliRunner()
    plot_dir = tmp_path / "out_plots"
    script_file = tmp_path / "train.py"
    script_file.write_text("print('from file')\n", encoding="utf-8")

    blocks = [
        TextContent(type="text", text="epoch 1 acc=0.95\n"),
        ImageContent(type="image", data=SAMPLE_PNG_B64, mime_type="image/png"),
    ]
    mock_exec = AsyncMock(return_value=(blocks, False))

    with (
        patch("colab_mcp.cli.ensure_cli_runtime_attached", new=AsyncMock(return_value=("ep-cli-1", 0))),
        patch("colab_mcp.kernel.tools.execute_code_impl", new=mock_exec),
    ):
        r_inline = runner.invoke(
            cli_app,
            [
                "execute-code",
                "print(1)",
                "--env",
                "SEED=42",
                "--stdin-input",
                "yes",
                "--save-images-dir",
                str(plot_dir),
                "--json",
            ],
        )
        r_file = runner.invoke(cli_app, ["execute-code", "--file", str(script_file)])
        r_pipe = runner.invoke(cli_app, ["execute-code", "-"], input="print('piped')\n")

    assert r_inline.exit_code == 0
    inline_json = json.loads(r_inline.stdout)
    assert inline_json["has_error"] is False
    assert (plot_dir / "plot_001.png").is_file()
    assert r_file.exit_code == 0 and "epoch 1" in r_file.stdout
    assert r_pipe.exit_code == 0 and "epoch 1" in r_pipe.stdout

    # Failing kernel code exits with code 13 (EXECUTION_ERROR)
    with (
        patch("colab_mcp.cli.ensure_cli_runtime_attached", new=AsyncMock(return_value=("ep-cli-1", 0))),
        patch(
            "colab_mcp.kernel.tools.execute_code_impl",
            new=AsyncMock(return_value=([TextContent(type="text", text="ZeroDivisionError")], True)),
        ),
    ):
        r_err = runner.invoke(cli_app, ["execute-code", "1/0", "--non-interactive"])
    assert r_err.exit_code == int(CLIExitCode.EXECUTION_ERROR)

    # Disconnected runtime exits with code 12
    with patch(
        "colab_mcp.cli.ensure_cli_runtime_attached",
        new=AsyncMock(side_effect=RuntimeDisconnectedError("No active runtime")),
    ):
        r_disc = runner.invoke(cli_app, ["execute-code", "print(1)", "--json"])
    assert r_disc.exit_code == int(CLIExitCode.RUNTIME_DISCONNECTED)


def test_cli_execute_notebook_and_exit_code_on_cell_failure(tmp_path: Path):
    runner = CliRunner()
    nb_file = tmp_path / "nb.ipynb"
    nb_file.write_text("{}", encoding="utf-8")

    ok_resp = NotebookExecutionResponse(
        total_cells=1,
        executed_cells=1,
        successful_cells=1,
        failed_cells=0,
        skipped_cells=0,
        total_duration_seconds=0.5,
        kernel_reset_performed=True,
        cell_summaries=[CellExecutionSummary(cell_index=0, execution_count=1, status="success", duration_seconds=0.5)],
        saved_to=None,
    )
    fail_resp = ok_resp.model_copy(
        update={
            "successful_cells": 0,
            "failed_cells": 1,
            "cell_summaries": [
                CellExecutionSummary(
                    cell_index=0,
                    execution_count=1,
                    status="failed",
                    duration_seconds=0.2,
                    error_name="ValueError",
                    error_value="boom",
                )
            ],
        }
    )

    with (
        patch("colab_mcp.cli.ensure_cli_runtime_attached", new=AsyncMock(return_value=("ep-cli-1", 0))),
        patch("colab_mcp.notebook.runner.execute_notebook_impl", new=AsyncMock(return_value=ok_resp)),
    ):
        r_ok = runner.invoke(cli_app, ["execute-notebook", str(nb_file)])
    assert r_ok.exit_code == 0
    assert "completed" in r_ok.stdout.lower()

    with (
        patch("colab_mcp.cli.ensure_cli_runtime_attached", new=AsyncMock(return_value=("ep-cli-1", 0))),
        patch("colab_mcp.notebook.runner.execute_notebook_impl", new=AsyncMock(return_value=fail_resp)),
    ):
        r_fail = runner.invoke(cli_app, ["execute-notebook", str(nb_file), "--json"])
    assert r_fail.exit_code == int(CLIExitCode.EXECUTION_ERROR)


def test_cli_files_and_drive_commands(tmp_path: Path):
    runner = CliRunner()
    lf_resp = ListFilesResponse(
        runtime_id="ep-cli-1",
        remote_path="/content",
        entries=[RemoteFileEntry(name="train.py", path="/content/train.py", type="file", size_bytes=512)],
        total_entries=1,
    )
    up_resp = UploadFileResponse(
        runtime_id="ep-cli-1",
        local_path=str(tmp_path / "a.txt"),
        remote_path="/content/a.txt",
        size_bytes=12,
        sha256="abc123",
        transfer_method="contents_api",
    )
    dl_resp = DownloadFileResponse(
        runtime_id="ep-cli-1",
        remote_path="/content/out.zip",
        local_path=str(tmp_path / "out.zip"),
        size_bytes=2048,
        sha256="def456",
        transfer_method="contents_api",
    )
    del_resp = DeleteFileResponse(
        runtime_id="ep-cli-1",
        remote_path="/content/old.txt",
        recursive=False,
        delete_method="contents_api",
    )
    dr_list = DriveListResponse(
        folder="MyDrive",
        files=[
            DriveFileEntry(
                id="id-1",
                name="ckpt.pt",
                mime_type="application/octet-stream",
                size_bytes=1024,
                modified_time=datetime.now(timezone.utc).isoformat(),
                is_folder=False,
            )
        ],
        total_files=1,
    )
    dr_tr = DriveTransferResponse(
        status="uploaded",
        file_id="id-2",
        file_name="ckpt.pt",
        size_bytes=1024,
        sha256="999aaa",
        local_path=None,
        runtime_path="/content/ckpt.pt",
    )

    with (
        patch("colab_mcp.cli.ensure_cli_runtime_attached", new=AsyncMock(return_value=("ep-cli-1", 0))),
        patch("colab_mcp.files.tools.list_files_impl", new=AsyncMock(return_value=lf_resp)),
        patch("colab_mcp.files.tools.upload_file_impl", new=AsyncMock(return_value=up_resp)),
        patch("colab_mcp.files.tools.download_file_impl", new=AsyncMock(return_value=dl_resp)),
        patch("colab_mcp.files.tools.delete_file_impl", new=AsyncMock(return_value=del_resp)),
        patch("colab_mcp.files.tools.drive_list_impl", new=AsyncMock(return_value=dr_list)),
        patch("colab_mcp.files.tools.drive_download_impl", new=AsyncMock(return_value=dr_tr)),
        patch("colab_mcp.files.tools.drive_upload_impl", new=AsyncMock(return_value=dr_tr)),
    ):
        assert runner.invoke(cli_app, ["list-files"]).exit_code == 0
        assert runner.invoke(cli_app, ["upload-file", str(tmp_path / "a.txt"), "/content/a.txt", "--json"]).exit_code == 0
        assert runner.invoke(cli_app, ["download-file", "/content/out.zip", str(tmp_path / "out.zip")]).exit_code == 0
        assert runner.invoke(cli_app, ["delete-file", "/content/old.txt", "--json"]).exit_code == 0
        r_dlist = runner.invoke(cli_app, ["drive-list"])
        assert r_dlist.exit_code == 0 and "ckpt.pt" in r_dlist.stdout
        assert runner.invoke(cli_app, ["drive-download", "id-1", "--local-path", str(tmp_path / "ckpt.pt")]).exit_code == 0
        assert runner.invoke(cli_app, ["drive-upload", "--runtime-path", "/content/ckpt.pt", "--json"]).exit_code == 0
