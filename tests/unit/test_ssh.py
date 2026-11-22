"""Unit and loopback wire tests for SSH-over-WebSocket runtime access."""

from __future__ import annotations

import asyncio
import io
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets
from typer.testing import CliRunner

from colab_mcp.cli import app as cli_app
from colab_mcp.cli_output import CLIExitCode
from colab_mcp.runtime.models import ConnectRuntimeResponse
from colab_mcp.runtime.ssh import (
    ColabSSHError,
    bridge_ws_stdio,
    build_proxy_command,
    build_ssh_subprocess_args,
    build_ssh_ws_url,
    connect_ssh_websocket,
    explain_ssh_handshake_failure,
    render_ssh_config_snippet,
    resolve_ssh_private_key,
    resolve_ssh_pubkey,
)


def test_resolve_ssh_pubkey_prefers_ed25519_and_skips_rsa(tmp_path: Path, monkeypatch):
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    (ssh_dir / "id_rsa.pub").write_text("ssh-rsa AAAA_IGNORED\n", encoding="utf-8")
    (ssh_dir / "id_ed25519.pub").write_text("ssh-ed25519 AAAA_ED25519 user@host\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert resolve_ssh_pubkey() == "ssh-ed25519 AAAA_ED25519 user@host"

    (ssh_dir / "id_ed25519.pub").unlink()
    (ssh_dir / "id_ecdsa.pub").write_text("ecdsa-sha2-nistp256 AAAA_ECDSA\n", encoding="utf-8")
    assert resolve_ssh_pubkey() == "ecdsa-sha2-nistp256 AAAA_ECDSA"

    (ssh_dir / "id_ecdsa.pub").unlink()
    with pytest.raises(ColabSSHError, match="id_ed25519.pub"):
        resolve_ssh_pubkey()


def test_resolve_ssh_pubkey_explicit_identity_and_keygen_fallback(tmp_path: Path, monkeypatch):
    priv = tmp_path / "custom_key"
    pub = tmp_path / "custom_key.pub"
    priv.write_text("PRIVATE", encoding="utf-8")
    pub.write_text("ssh-ed25519 CUSTOM_PUB\n", encoding="utf-8")

    # Passing private key picks up sibling .pub without running ssh-keygen
    assert resolve_ssh_pubkey(str(priv)) == "ssh-ed25519 CUSTOM_PUB"
    assert resolve_ssh_private_key(str(pub)) == str(priv)

    # Without sibling .pub, invokes `ssh-keygen -y -f`
    pub.unlink()
    fake_proc = MagicMock(stdout="ssh-ed25519 FROM_KEYGEN\n")
    with patch("colab_mcp.runtime.ssh.subprocess.run", return_value=fake_proc) as run_mock:
        assert resolve_ssh_pubkey(str(priv)) == "ssh-ed25519 FROM_KEYGEN"
        run_mock.assert_called_once()


def test_build_ws_url_and_handshake_explanations():
    ws_url = build_ssh_ws_url("https://proxy.colab.test/", "tok-123", mode="cmd")
    assert ws_url.startswith("wss://proxy.colab.test/colab/ssh?")
    assert "colab-runtime-proxy-token=tok-123" in ws_url
    assert "colab-client-agent=colab-mcp" in ws_url
    assert "colab-client-agent=ssh-mode-cmd" in ws_url

    assert "unsupported key type" in explain_ssh_handshake_failure(400, b"unsupported key type")
    assert "HTTP 401" in explain_ssh_handshake_failure(401)
    assert "HTTP 404" in explain_ssh_handshake_failure(404)
    limit_msg = explain_ssh_handshake_failure(
        429,
        json.dumps({"error": "already-active-session", "limit": 2}).encode("utf-8"),
    )
    assert "2 of 2 active SSH sessions" in limit_msg
    assert "--force" in limit_msg
    assert "HTTP 502" in explain_ssh_handshake_failure(502)


def test_build_proxy_command_and_ssh_config_snippet(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sys, "executable", str(tmp_path / "Python with spaces" / "python.exe"))
    cmd = build_proxy_command(
        runtime_id="ep-ssh-1",
        identity=str(tmp_path / "my keys" / "id_ed25519.pub"),
        mode="proxy",
        rm=True,
    )
    assert "ssh" in cmd
    assert "--proxy-mode" in cmd
    assert "--runtime-id" in cmd
    assert "ep-ssh-1" in cmd
    assert "--rm" in cmd

    snippet = render_ssh_config_snippet(host_alias="my-colab", accelerator="T4", rm=True)
    assert "Host my-colab" in snippet
    assert "User root" in snippet
    assert "ProxyCommand " in snippet


def test_build_ssh_subprocess_args_tty_and_command(monkeypatch, tmp_path: Path):
    priv = tmp_path / "id_ed25519"
    pub = tmp_path / "id_ed25519.pub"
    priv.write_text("PRIV", encoding="utf-8")
    pub.write_text("ssh-ed25519 PUB", encoding="utf-8")
    monkeypatch.setattr("colab_mcp.runtime.ssh.shutil.which", lambda _: "/usr/bin/ssh")

    interactive = build_ssh_subprocess_args("proxy-cmd", identity=str(pub), force_tty=True)
    assert "-t" in interactive
    assert "-i" in interactive and str(priv) in interactive
    assert "root@colab-runtime" in interactive
    assert "exec ${SHELL:-/bin/bash} -l" in interactive[-1]

    remote_cmd = build_ssh_subprocess_args(
        "proxy-cmd",
        remote_command=["ls", "-la", "my file.txt"],
        force_tty=False,
    )
    assert "-T" in remote_cmd
    assert "'my file.txt'" in remote_cmd[-1] or '"my file.txt"' in remote_cmd[-1]


@pytest.mark.asyncio
async def test_ssh_websocket_wire_contract_and_stdio_bridge():
    received_headers = {}
    received_path = ""
    echoed_frames = []

    async def server_handler(ws):
        nonlocal received_headers, received_path
        req = ws.request
        received_path = req.path
        received_headers = dict(req.headers)
        async for msg in ws:
            echoed_frames.append(msg)
            await ws.send(b"srv:" + (msg if isinstance(msg, bytes) else msg.encode("utf-8")))
            break

    async with websockets.serve(server_handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        ws = await connect_ssh_websocket(
            f"http://127.0.0.1:{port}",
            "proxy-tok-99",
            "ssh-ed25519 WIRETEST",
            mode="interactive",
        )
        in_stream = io.BytesIO(b"hello-ssh-bytes")
        out_stream = io.BytesIO()
        rc = await bridge_ws_stdio(ws, stdin_stream=in_stream, stdout_stream=out_stream)

    assert rc == 0
    assert "/colab/ssh?" in received_path
    assert "colab-runtime-proxy-token=proxy-tok-99" in received_path
    assert "ssh-mode-interactive" in received_path
    lower_headers = {k.lower(): v for k, v in received_headers.items()}
    assert lower_headers["x-colab-ssh-pubkey"] == "ssh-ed25519 WIRETEST"
    assert lower_headers["x-colab-client-agent"] == "colab-mcp"
    assert "x-colab-runtime-proxy-token" not in lower_headers
    assert echoed_frames == [b"hello-ssh-bytes"]
    assert out_stream.getvalue() == b"srv:hello-ssh-bytes"


def test_cli_ssh_config_and_missing_binary_guard(tmp_path: Path, monkeypatch):
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    pub = ssh_dir / "id_ed25519.pub"
    pub.write_text("ssh-ed25519 CFG_KEY\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    runner = CliRunner()
    cfg_res = runner.invoke(cli_app, ["ssh", "--ssh-config", "--accelerator", "T4", "--rm"])
    assert cfg_res.exit_code == 0
    assert "Host colab" in cfg_res.stdout
    assert "--proxy-mode" in cfg_res.stdout

    # Combined --proxy-mode and positional CMD rejects before any runtime call
    bad_res = runner.invoke(cli_app, ["ssh", "--proxy-mode", "--", "ls"])
    assert bad_res.exit_code == int(CLIExitCode.USAGE_ERROR)

    # Missing OpenSSH client fails before attempting VM provisioning
    monkeypatch.setattr("shutil.which", lambda _: None)
    connect_mock = AsyncMock(
        return_value=ConnectRuntimeResponse(runtime_id="never", accelerator="NONE", variant="DEFAULT")
    )
    with patch("colab_mcp.runtime.tools.connect_runtime_impl", connect_mock):
        no_ssh = runner.invoke(cli_app, ["ssh"])
    assert no_ssh.exit_code != 0
    assert not connect_mock.called


@pytest.mark.asyncio
async def test_ssh_handshake_rejection_returns_actionable_error():
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    async def reject_429(_conn, _req):
        body = json.dumps({"error": "already-active-session", "limit": 2}).encode("utf-8")
        return Response(429, "Too Many Requests", Headers([("Content-Type", "application/json")]), body)

    async def noop_handler(_ws):
        return None

    async with websockets.serve(noop_handler, "127.0.0.1", 0, process_request=reject_429) as server:
        port = server.sockets[0].getsockname()[1]
        with pytest.raises(ColabSSHError, match="2 of 2 active SSH sessions"):
            await connect_ssh_websocket(
                f"http://127.0.0.1:{port}",
                "tok",
                "ssh-ed25519 TEST",
            )


def test_cli_ssh_proxy_rm_and_cmd_exit_propagation(tmp_path: Path, monkeypatch):
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    (ssh_dir / "id_ed25519.pub").write_text("ssh-ed25519 TESTKEY\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    runner = CliRunner()
    disc_mock = AsyncMock()
    with (
        patch("colab_mcp.cli.ensure_cli_runtime_attached", AsyncMock(return_value=("ep-proxy-1", 0))),
        patch("colab_mcp.runtime.session_manager.get_session_manager") as m_mgr,
        patch("colab_mcp.runtime.ssh.connect_ssh_websocket", AsyncMock(return_value=MagicMock())),
        patch("colab_mcp.runtime.ssh.bridge_ws_stdio", AsyncMock(side_effect=RuntimeError("boom"))),
        patch("colab_mcp.cli._disconnect_ssh_runtime", disc_mock),
    ):
        m_mgr.return_value.get_runtime_proxy_params = AsyncMock(
            return_value=("ep-proxy-1", "https://proxy.colab.test", "tok", None)
        )
        res = runner.invoke(cli_app, ["ssh", "--proxy-mode", "--rm"])

    assert res.exit_code != 0
    disc_mock.assert_awaited_once_with("ep-proxy-1", 0)

    # `-- CMD` uses attached runtime and forwards subprocess exit code cleanly
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ssh")
    with (
        patch("colab_mcp.cli.ensure_cli_runtime_attached", AsyncMock(return_value=("ep-cmd-1", 0))),
        patch("colab_mcp.cli.subprocess.call", return_value=7) as m_call,
    ):
        cmd_res = runner.invoke(cli_app, ["ssh", "--", "ls", "-r"])
    assert cmd_res.exit_code == 7
    assert "ls -r" in m_call.call_args.args[0][-1]
