"""
Unit tests for encrypted user secrets vault, dynamic egress redaction,
headless google.colab popup shims, CLI secret commands, and system metrics.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from colab_mcp.auth.secrets import (
    delete_secret,
    get_all_secrets,
    get_secret,
    get_secrets_file,
    list_secrets,
    mask_secret_value,
    resolve_secret_interpolation,
    set_secret,
)
from colab_mcp.auth.tools import (
    colab_delete_secret_impl,
    colab_list_secrets_impl,
    colab_set_secret_impl,
)
from colab_mcp.cli import app as cli_app
from colab_mcp.kernel.colab_shims import (
    build_colab_bootstrap_code,
    build_system_metrics_probe_code,
)
from colab_mcp.kernel.models import RawExecutionResult
from colab_mcp.runtime.tools import get_system_metrics_impl
from colab_mcp.server import create_server
from colab_mcp.utils.redaction import (
    clear_dynamic_secrets,
    redact_credentials,
    redact_data,
)


def test_secrets_vault_crud_encryption_and_redaction(
    mock_keystore_dir: Path,
    mock_keyring: dict,
    mock_machine_id: bytes,
):
    clear_dynamic_secrets()
    secret_val = "hf_SuperSecretToken987654"
    meta = set_secret("HF_TOKEN", secret_val)

    assert meta.name == "HF_TOKEN"
    assert meta.length == len(secret_val)
    assert secret_val not in meta.masked_preview
    assert meta.masked_preview == mask_secret_value(secret_val)
    assert get_secrets_file().is_file()

    assert get_secret("HF_TOKEN") == secret_val
    assert get_all_secrets() == {"HF_TOKEN": secret_val}
    listed = list_secrets()
    assert len(listed) == 1
    assert listed[0].name == "HF_TOKEN"

    # Global redaction scrubs newly stored secrets
    assert secret_val not in redact_credentials(f"Leaked: token={secret_val}")
    assert "[REDACTED]" in redact_credentials(f"Leaked: token={secret_val}")
    redacted_dict = redact_data({"env": {"HF_TOKEN": secret_val}, "ok": "safe"})
    assert redacted_dict == {"env": {"HF_TOKEN": "[REDACTED]"}, "ok": "safe"}

    # $SECRET:<NAME> interpolation
    assert resolve_secret_interpolation("$SECRET:HF_TOKEN") == secret_val
    assert resolve_secret_interpolation("plain-value") == "plain-value"
    with pytest.raises(ValueError, match="NO_SUCH_KEY"):
        resolve_secret_interpolation("$SECRET:NO_SUCH_KEY")

    # Delete removes from vault and redaction registry
    assert delete_secret("HF_TOKEN") is True
    assert get_secret("HF_TOKEN") is None
    assert delete_secret("HF_TOKEN") is False


def test_secrets_fallback_file_works_when_keyring_fails(
    mock_keystore_dir: Path,
    mock_machine_id: bytes,
    monkeypatch: pytest.MonkeyPatch,
):
    import keyring

    def _boom(*args, **kwargs):
        raise RuntimeError("No OS keyring daemon")

    monkeypatch.setattr(keyring, "get_password", _boom)
    monkeypatch.setattr(keyring, "set_password", _boom)

    set_secret("WANDB_API_KEY", "wandb_key_123456")
    assert get_secrets_file().is_file()
    assert get_secret("WANDB_API_KEY") == "wandb_key_123456"
    assert delete_secret("WANDB_API_KEY") is True


@pytest.mark.parametrize("bad_name", ["", "123_BAD", "bad-name", "has space", "a;b"])
def test_secret_name_validation_rejects_invalid_identifiers(bad_name: str):
    with pytest.raises(ValueError, match="Secret name"):
        set_secret(bad_name, "valid-secret-value")


def test_headless_colab_shims_patch_userdata_drive_auth_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    google_pkg = types.ModuleType("google")
    colab_pkg = types.ModuleType("google.colab")
    userdata_mod = types.ModuleType("google.colab.userdata")
    drive_mod = types.ModuleType("google.colab.drive")
    auth_mod = types.ModuleType("google.colab.auth")
    files_mod = types.ModuleType("google.colab.files")

    class SecretNotFoundError(Exception):
        pass

    userdata_mod.SecretNotFoundError = SecretNotFoundError
    userdata_mod.get = lambda _k: (_ for _ in ()).throw(RuntimeError("browser modal blocked"))
    drive_mod.mount = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("drive FUSE blocked"))
    auth_mod.authenticate_user = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("auth popup blocked"))
    files_mod.upload = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("upload widget blocked"))
    files_mod.download = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("download widget blocked"))

    colab_pkg.userdata = userdata_mod
    colab_pkg.drive = drive_mod
    colab_pkg.auth = auth_mod
    colab_pkg.files = files_mod
    google_pkg.colab = colab_pkg

    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.colab", colab_pkg)
    monkeypatch.setitem(sys.modules, "google.colab.userdata", userdata_mod)
    monkeypatch.setitem(sys.modules, "google.colab.drive", drive_mod)
    monkeypatch.setitem(sys.modules, "google.colab.auth", auth_mod)
    monkeypatch.setitem(sys.modules, "google.colab.files", files_mod)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("CUSTOM_VAR", raising=False)
    monkeypatch.delenv("GOOGLE_OAUTH_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDSDK_AUTH_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("COLAB_MCP_DRIVE_MOUNTPOINT", raising=False)

    ns: dict = {}
    code = build_colab_bootstrap_code(
        secrets={"HF_TOKEN": "hf_shim_test_token"},
        env_overrides={"CUSTOM_VAR": "custom_value"},
        access_token="ya29.headless_oauth_token",
    )
    exec(compile(code, "<shim-test>", "exec"), ns)

    # 1. userdata.get + os.environ
    assert os.environ["HF_TOKEN"] == "hf_shim_test_token"
    assert os.environ["CUSTOM_VAR"] == "custom_value"
    assert userdata_mod.get("HF_TOKEN") == "hf_shim_test_token"
    with pytest.raises(SecretNotFoundError, match="colab_set_secret"):
        userdata_mod.get("UNSET_KEY")

    # 2. drive.mount (Option A)
    mount_path = tmp_path / "drive"
    drive_mod.mount(str(mount_path))
    assert (mount_path / "MyDrive").is_dir()
    assert os.environ["COLAB_MCP_DRIVE_MOUNTPOINT"] == str(mount_path.resolve())

    # 3. auth.authenticate_user
    auth_mod.authenticate_user(project_id="my-gcp-proj")
    assert os.environ["GOOGLE_OAUTH_ACCESS_TOKEN"] == "ya29.headless_oauth_token"
    assert os.environ["GOOGLE_CLOUD_PROJECT"] == "my-gcp-proj"

    # 4. files.download + files.upload via stdin fallback
    sample_file = tmp_path / "weights.bin"
    sample_file.write_bytes(b"\x01\x02\x03")
    files_mod.download(str(sample_file))
    with pytest.raises(FileNotFoundError):
        files_mod.download(str(tmp_path / "missing.bin"))

    orig_input = __builtins__["input"] if isinstance(__builtins__, dict) else __builtins__.input
    monkeypatch.setattr("builtins.input", lambda _prompt="": str(sample_file))
    try:
        uploaded = files_mod.upload()
    finally:
        monkeypatch.setattr("builtins.input", orig_input)
    assert uploaded == {"weights.bin": b"\x01\x02\x03"}

    # 5. Secret deletion updates kernel namespace and os.environ
    rm_code = build_colab_bootstrap_code(removed_secrets=["HF_TOKEN"])
    exec(compile(rm_code, "<shim-rm-test>", "exec"), ns)
    assert "HF_TOKEN" not in os.environ
    with pytest.raises(SecretNotFoundError):
        userdata_mod.get("HF_TOKEN")


def test_cli_secret_commands_roundtrip(
    mock_keystore_dir: Path,
    mock_keyring: dict,
    mock_machine_id: bytes,
):
    runner = CliRunner()
    res_set = runner.invoke(cli_app, ["set-secret", "OPENAI_API_KEY", "--value", "sk-test-123456"])
    assert res_set.exit_code == 0
    payload_set = json.loads(res_set.stdout)
    assert payload_set["name"] == "OPENAI_API_KEY"
    assert "sk-test-123456" not in res_set.stdout

    res_list = runner.invoke(cli_app, ["list-secrets"])
    assert res_list.exit_code == 0
    payload_list = json.loads(res_list.stdout)
    assert payload_list["count"] == 1
    assert payload_list["secrets"][0]["name"] == "OPENAI_API_KEY"

    res_del = runner.invoke(cli_app, ["delete-secret", "OPENAI_API_KEY"])
    assert res_del.exit_code == 0
    assert json.loads(res_del.stdout)["existed"] is True


@pytest.mark.asyncio
async def test_mcp_secret_tools_via_server(
    mock_keystore_dir: Path,
    mock_keyring: dict,
    mock_machine_id: bytes,
):
    server = create_server("secrets-mcp-test")
    tools = {t.name for t in await server.list_tools()}
    assert {"colab_set_secret", "colab_list_secrets", "colab_delete_secret", "colab_send_input", "colab_get_system_metrics"}.issubset(tools)

    set_res = await colab_set_secret_impl("HF_TOKEN", "hf_mcp_secret_abcdef")
    assert set_res.status == "stored"
    list_res = await colab_list_secrets_impl()
    assert list_res.count == 1
    assert list_res.secrets[0].name == "HF_TOKEN"
    del_res = await colab_delete_secret_impl("HF_TOKEN")
    assert del_res.existed is True


def test_system_metrics_probe_code_runs_locally():
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(compile(build_system_metrics_probe_code(), "<metrics-probe>", "exec"), {})
    data = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert "gpus" in data
    assert "disk_path" in data
    assert data["disk_total_gb"] is not None


@pytest.mark.asyncio
async def test_get_system_metrics_impl_parses_kernel_json():
    metrics_json = json.dumps(
        {
            "gpus": [
                {
                    "index": 0,
                    "name": "NVIDIA L4",
                    "gpu_utilization_percent": 42.0,
                    "vram_used_mb": 4096.0,
                    "vram_total_mb": 23034.0,
                    "temperature_celsius": 55.0,
                }
            ],
            "ram_total_mb": 54272.0,
            "ram_used_mb": 12000.0,
            "ram_free_mb": 42272.0,
            "ram_utilization_percent": 22.1,
            "disk_path": "/content",
            "disk_total_gb": 78.2,
            "disk_used_gb": 18.4,
            "disk_free_gb": 59.8,
            "disk_utilization_percent": 23.5,
        }
    )
    mock_client = AsyncMock()
    mock_client.runtime_id = "gpu-l4-ep-1"
    mock_client.execute_raw.return_value = RawExecutionResult(
        status="ok",
        iopub_messages=[
            {"header": {"msg_type": "stream"}, "content": {"name": "stdout", "text": metrics_json + "\n"}}
        ],
    )

    with patch("colab_mcp.runtime.tools.get_session_manager") as mock_mgr:
        resp = await get_system_metrics_impl(
            runtime_id="gpu-l4-ep-1",
            _client_getter=AsyncMock(return_value=mock_client),
        )
    assert mock_mgr.called
    assert resp.runtime_id == "gpu-l4-ep-1"
    assert len(resp.gpus) == 1
    assert resp.gpus[0].name == "NVIDIA L4"
    assert resp.gpus[0].vram_used_mb == 4096.0
    assert resp.ram_total_mb == 54272.0
    assert resp.disk_free_gb == 59.8
