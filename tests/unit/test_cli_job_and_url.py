"""Unit tests for ephemeral `job`, `runtime-url`, Accelerator.G4, and proxy param helper."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import TextContent
from typer.testing import CliRunner

from colab_mcp.cli import app as cli_app
from colab_mcp.cli_job import build_job_payload, extract_job_result, strip_shebang
from colab_mcp.cli_session import load_cli_session
from colab_mcp.runtime.models import (
    Accelerator,
    Assignment,
    ConnectRuntimeResponse,
    DisconnectRuntimeResponse,
    Variant,
    get_accelerator_catalog,
    resolve_variant,
)
from colab_mcp.runtime.session_manager import SessionManager


def test_strip_shebang_and_build_job_payload(tmp_path: Path):
    script = tmp_path / "train.py"
    script.write_text(
        "#!/usr/bin/env -S gcolab-mcp job --accelerator T4\nimport sys\nprint(sys.argv)\n",
        encoding="utf-8",
    )
    assert strip_shebang("#!/usr/bin/python3\nprint(1)\n") == "print(1)\n"

    payload = build_job_payload(script, ["--epochs", "5", "--json"])
    assert "sys.argv = ['train.py', '--epochs', '5', '--json']" in payload
    assert "__name__ = '__main__'" in payload
    assert "#!/" not in payload
    assert "compile(" in payload


def test_extract_job_result_handles_sys_exit_and_strips_sentinel():
    blocks = [
        TextContent(type="text", text="epoch=1\n__COLAB_MCP_JOB_EXIT__=42\n"),
    ]
    cleaned, code = extract_job_result(blocks, has_error=False)
    assert code == 42
    assert len(cleaned) == 1
    assert cleaned[0].text == "epoch=1\n"

    blocks_zero = [TextContent(type="text", text="__COLAB_MCP_JOB_EXIT__=0\n")]
    cleaned_z, code_z = extract_job_result(blocks_zero, has_error=False)
    assert code_z == 0
    assert cleaned_z == []

    _, code_err = extract_job_result([TextContent(type="text", text="Traceback...")], has_error=True)
    assert code_err == 13


def test_cli_job_missing_script_aborts_before_connect(tmp_path: Path):
    runner = CliRunner()
    connect_mock = AsyncMock()
    with patch("colab_mcp.runtime.tools.connect_runtime_impl", connect_mock):
        res = runner.invoke(cli_app, ["job", str(tmp_path / "missing.py")])
    assert res.exit_code != 0
    assert not connect_mock.called


def test_cli_job_teardown_runs_even_on_execution_exception(
    mock_keystore_dir: Path,
    tmp_path: Path,
):
    script = tmp_path / "crash.py"
    script.write_text("print('before crash')\n", encoding="utf-8")

    conn_resp = ConnectRuntimeResponse(
        runtime_id="ep-job-crash",
        accelerator="NONE",
        variant="DEFAULT",
        kernel_id="k-crash",
    )
    disc_resp = DisconnectRuntimeResponse(
        runtime_id="ep-job-crash",
        status="disconnected",
        terminated_runtime_ids=["ep-job-crash"],
        terminated_count=1,
        message="Disconnected.",
    )
    with (
        patch("colab_mcp.runtime.tools.connect_runtime_impl", AsyncMock(return_value=conn_resp)),
        patch(
            "colab_mcp.kernel.tools.execute_code_impl",
            AsyncMock(side_effect=RuntimeError("kernel crashed")),
        ),
        patch(
            "colab_mcp.runtime.tools.disconnect_runtime_impl", AsyncMock(return_value=disc_resp)
        ) as m_disc,
    ):
        runner = CliRunner()
        res = runner.invoke(cli_app, ["job", str(script)])

    assert res.exit_code == 1
    m_disc.assert_awaited_once()


def test_cli_job_provisions_forwards_args_and_disconnects_on_exit_code(
    mock_keystore_dir: Path,
    tmp_path: Path,
):
    script = tmp_path / "task.py"
    script.write_text("#!/usr/bin/env python3\nimport sys\nsys.exit(7)\n", encoding="utf-8")

    conn_resp = ConnectRuntimeResponse(
        runtime_id="ep-job-1",
        accelerator="T4",
        variant="GPU",
        kernel_id="k-job-1",
    )
    disc_resp = DisconnectRuntimeResponse(
        runtime_id="ep-job-1",
        status="disconnected",
        terminated_runtime_ids=["ep-job-1"],
        terminated_count=1,
        message="Disconnected.",
    )
    seen = {}

    async def fake_exec(**kwargs):
        seen["code"] = kwargs["code"]
        seen["env"] = kwargs.get("env")
        return [TextContent(type="text", text="hello-job\n__COLAB_MCP_JOB_EXIT__=7\n")], False

    with (
        patch("colab_mcp.runtime.tools.connect_runtime_impl", AsyncMock(return_value=conn_resp)) as m_conn,
        patch("colab_mcp.kernel.tools.execute_code_impl", AsyncMock(side_effect=fake_exec)),
        patch(
            "colab_mcp.runtime.tools.disconnect_runtime_impl", AsyncMock(return_value=disc_resp)
        ) as m_disc,
    ):
        runner = CliRunner()
        res = runner.invoke(
            cli_app,
            ["job", "--accelerator", "T4", "--env", "WANDB_MODE=disabled", str(script), "--json", "--lr", "0.1"],
        )

    assert res.exit_code == 7
    assert "hello-job" in res.stdout
    assert "__COLAB_MCP_JOB_EXIT__" not in res.stdout
    assert "sys.argv = ['task.py', '--json', '--lr', '0.1']" in seen["code"]
    assert seen["env"] == {"WANDB_MODE": "disabled"}
    assert m_conn.call_args.kwargs["reuse_existing"] is False
    m_disc.assert_awaited_once()
    assert load_cli_session() is None


def test_cli_job_keep_preserves_session_and_skips_disconnect(
    mock_keystore_dir: Path,
    tmp_path: Path,
):
    script = tmp_path / "keep_task.py"
    script.write_text("print('kept')\n", encoding="utf-8")

    conn_resp = ConnectRuntimeResponse(
        runtime_id="ep-job-keep",
        accelerator="NONE",
        variant="DEFAULT",
        kernel_id="k-keep",
    )
    with (
        patch("colab_mcp.runtime.tools.connect_runtime_impl", AsyncMock(return_value=conn_resp)),
        patch(
            "colab_mcp.kernel.tools.execute_code_impl",
            AsyncMock(return_value=([TextContent(type="text", text="kept\n")], False)),
        ),
        patch("colab_mcp.runtime.tools.disconnect_runtime_impl", AsyncMock()) as m_disc,
    ):
        runner = CliRunner()
        res = runner.invoke(cli_app, ["job", "--keep", "--json", str(script)])

    assert res.exit_code == 0
    m_disc.assert_not_called()
    saved = load_cli_session()
    assert saved is not None and saved.runtime_id == "ep-job-keep"
    payload = json.loads(res.stdout)
    assert payload["exit_code"] == 0
    assert payload["has_error"] is False


def test_cli_runtime_url_and_accelerator_g4(mock_keystore_dir: Path):
    runner = CliRunner()
    with (
        patch(
            "colab_mcp.cli.ensure_cli_runtime_attached",
            AsyncMock(return_value=("m-s-url-123", 0)),
        ),
        patch("colab_mcp.runtime.session_manager.get_session_manager") as m_mgr,
        patch("colab_mcp.cli.webbrowser.open") as m_open,
    ):
        mgr_inst = m_mgr.return_value
        mgr_inst.get_runtime_proxy_params = AsyncMock(
            return_value=(
                "m-s-url-123",
                "https://proxy.colab.test",
                "tok",
                datetime.now(timezone.utc) + timedelta(hours=1),
            )
        )
        res = runner.invoke(cli_app, ["runtime-url", "--open"])

    assert res.exit_code == 0
    expected = (
        "https://colab.research.google.com/notebooks/empty.ipynb"
        "?dbu=%2Ftun%2Fm%2Fm-s-url-123#datalabBackendUrl=https://colab.research.google.com/tun/m/m-s-url-123"
    )
    assert res.stdout.strip() == expected
    m_open.assert_called_once_with(expected)

    assert resolve_variant("G4") == Variant.GPU
    catalog = {c.accelerator: c for c in get_accelerator_catalog("PRO")}
    assert "G4" in catalog and catalog["G4"].variant == "GPU"
    free_catalog = {c.accelerator: c for c in get_accelerator_catalog("FREE")}
    assert free_catalog["G4"].available is False


@pytest.mark.asyncio
async def test_session_manager_get_runtime_proxy_params_skips_kernel_discovery():
    mgr = SessionManager()
    mgr.register_session(
        endpoint="ep-no-kernel",
        assignment=Assignment(
            endpoint="ep-no-kernel",
            accelerator="T4",
            variant="GPU",
            runtime_proxy_url="https://proxy.colab.test",
            runtime_proxy_token="tok-xyz",
            token_expires_in=3600,
        ),
    )
    with patch.object(mgr, "_discover_or_create_kernel", AsyncMock()) as m_kernel:
        ep, url, tok, _ = await mgr.get_runtime_proxy_params("ep-no-kernel")
    assert (ep, url, tok) == ("ep-no-kernel", "https://proxy.colab.test", "tok-xyz")
    m_kernel.assert_not_called()
