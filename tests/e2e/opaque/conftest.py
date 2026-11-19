"""
Opaque-box E2E test fixtures:
- Fail-closed isolation (in-memory KeyringBackend, isolated USERPROFILE/HOME/CLOUDSDK_CONFIG,
  deterministic machine ID, SessionManager shutdown + reset, stdout protection reset)
- Local Colab Twin lifecycle + selective host-based HTTP/WS/browser interception
- Unified `mcp_session(transport="memory"|"sse"|"stdio")` context manager factory

ZERO imports from `colab_mcp.kernel` or `colab_mcp.notebook`.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Literal, Optional
from urllib.parse import parse_qs, urlparse, urlunparse

import httpx
import keyring
import keyring.backend
import keyring.errors
import pytest
import requests
import uvicorn
import websockets
from mcp import ClientSession
from mcp.client._memory import InMemoryTransport
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, get_default_environment, stdio_client
from typer.testing import CliRunner

from colab_mcp.auth.oauth import _ACTIVE_REMOTE_FLOWS
from colab_mcp.runtime.session_manager import get_session_manager, reset_session_manager
from colab_mcp.server import create_server
from colab_mcp.utils.logging import unprotect_stdout

from .colab_twin import TWIN_EXTERNAL_HOSTS, ColabTwin
from .helpers import require_cli_command


class InMemoryKeyringBackend(keyring.backend.KeyringBackend):
    """Process-local, isolated in-memory KeyringBackend (never touches host WinVault/Keychain)."""

    priority = 10

    def __init__(self) -> None:
        self._store: Dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> Optional[str]:
        return self._store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        if (service, username) not in self._store:
            raise keyring.errors.PasswordDeleteError("Password not found")
        del self._store[(service, username)]


def _reset_optional_m3_caches() -> None:
    """Reset M3 in-memory caches if M3 modules are loaded, without importing them."""
    kernel_tools_mod = sys.modules.get("colab_mcp.kernel.tools")
    if kernel_tools_mod is not None:
        for attr in ("reset_kernel_tools", "reset_kernel_state"):
            fn = getattr(kernel_tools_mod, attr, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
        get_reg = getattr(kernel_tools_mod, "get_execution_registry", None)
        if callable(get_reg):
            try:
                reg = get_reg()
                clear_fn = getattr(reg, "clear", None)
                if callable(clear_fn):
                    clear_fn()
                elif hasattr(reg, "_executions") and isinstance(reg._executions, dict):
                    reg._executions.clear()
            except Exception:
                pass
        for dict_attr in ("_ACTIVE_KERNEL_CLIENTS", "_KERNEL_CLIENTS", "_CLIENTS"):
            d = getattr(kernel_tools_mod, dict_attr, None)
            if isinstance(d, dict):
                d.clear()


@pytest.fixture(autouse=True)
async def isolated_opaque_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[Path, None]:
    """Fail-closed isolation across keyring, filesystem, env, singletons, and stdout."""
    _ACTIVE_REMOTE_FLOWS.clear()
    unprotect_stdout()
    reset_session_manager()
    _reset_optional_m3_caches()

    cred_dir = tmp_path / ".colabmcp"
    cred_dir.mkdir(parents=True, exist_ok=True)
    gcloud_dir = tmp_path / "gcloud"
    gcloud_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CLOUDSDK_CONFIG", str(gcloud_dir))
    monkeypatch.setenv("NO_GCE_CHECK", "true")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("COLAB_MCP_KEY", raising=False)

    monkeypatch.setattr("colab_mcp.auth.keystore.get_credentials_dir", lambda: cred_dir)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        "colab_mcp.auth.keystore.get_machine_id_bytes",
        lambda: b"DETERMINISTIC_TEST_MACHINE_ID_BYTES_12345",
    )

    prev_keyring = keyring.get_keyring()
    mem_keyring = InMemoryKeyringBackend()
    keyring.set_keyring(mem_keyring)

    try:
        yield tmp_path
    finally:
        _ACTIVE_REMOTE_FLOWS.clear()
        try:
            mgr = get_session_manager()
            await mgr.shutdown()
        except Exception:
            pass
        reset_session_manager()
        _reset_optional_m3_caches()
        keyring.set_keyring(prev_keyring)
        unprotect_stdout()


@pytest.fixture
def colab_twin(monkeypatch: pytest.MonkeyPatch) -> ColabTwin:
    """
    Start Local Colab Twin (HTTP router + real RFC 6455 WS server) and install
    selective host-based network interception (fail-closed on unknown external hosts).
    """
    twin = ColabTwin().start()

    orig_async_send = httpx.AsyncClient.send
    orig_sync_send = httpx.Client.send
    orig_requests_send = requests.Session.send
    orig_ws_connect = websockets.connect

    def _classify_url(url_str: str) -> str:
        p = urlparse(str(url_str))
        host = (p.hostname or "").lower()
        if host in TWIN_EXTERNAL_HOSTS or (host in ("127.0.0.1", "localhost") and p.port == twin.cdp_port):
            return "twin"
        if host in ("127.0.0.1", "localhost", "::1"):
            return "local_passthrough"
        return "blocked"

    async def _intercepted_async_send(self: httpx.AsyncClient, request: httpx.Request, *args: Any, **kwargs: Any) -> httpx.Response:
        route = _classify_url(str(request.url))
        if route == "twin":
            return twin.handle_httpx_request(request)
        if route == "local_passthrough":
            return await orig_async_send(self, request, *args, **kwargs)
        raise RuntimeError(f"Offline opaque E2E guard blocked outbound HTTP request: {request.url}")

    def _intercepted_sync_send(self: httpx.Client, request: httpx.Request, *args: Any, **kwargs: Any) -> httpx.Response:
        route = _classify_url(str(request.url))
        if route == "twin":
            return twin.handle_httpx_request(request)
        if route == "local_passthrough":
            return orig_sync_send(self, request, *args, **kwargs)
        raise RuntimeError(f"Offline opaque E2E guard blocked outbound HTTP request: {request.url}")

    def _intercepted_requests_send(self: requests.Session, request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:
        route = _classify_url(request.url or "")
        if route == "twin":
            return twin.handle_requests_prepared(request)
        if route == "local_passthrough":
            return orig_requests_send(self, request, **kwargs)
        raise RuntimeError(f"Offline opaque E2E guard blocked outbound requests call: {request.url}")

    def _intercepted_ws_connect(uri: str, *args: Any, **kwargs: Any) -> Any:
        p = urlparse(str(uri))
        host = (p.hostname or "").lower()
        if p.scheme == "wss" and host == "colab.research.google.com":
            rewritten = urlunparse(("ws", f"127.0.0.1:{twin.ws_port}", p.path, p.params, p.query, p.fragment))
            return orig_ws_connect(rewritten, *args, **kwargs)
        if host in ("127.0.0.1", "localhost"):
            return orig_ws_connect(uri, *args, **kwargs)
        raise RuntimeError(f"Offline opaque E2E guard blocked outbound WebSocket: {uri}")

    opened_urls: List[str] = []

    def _intercepted_webbrowser_open(url: str, *args: Any, **kwargs: Any) -> bool:
        opened_urls.append(url)
        p = urlparse(url)
        qs = parse_qs(p.query)
        redirect_uri = qs.get("redirect_uri", [""])[0]
        state = qs.get("state", [""])[0]
        if redirect_uri.startswith(("http://127.0.0.1:", "http://localhost:")):
            code = twin.issue_auth_code("twin-loopback-auth-code", auth_user_index=0)

            def _hit_callback() -> None:
                time.sleep(0.05)
                try:
                    with httpx.Client(timeout=3.0) as c:
                        c.get(redirect_uri, params={"state": state, "code": code})
                except Exception:
                    pass

            threading.Thread(target=_hit_callback, daemon=True).start()
        return True

    monkeypatch.setattr(httpx.AsyncClient, "send", _intercepted_async_send)
    monkeypatch.setattr(httpx.Client, "send", _intercepted_sync_send)
    monkeypatch.setattr(requests.Session, "send", _intercepted_requests_send)
    monkeypatch.setattr(websockets, "connect", _intercepted_ws_connect)
    monkeypatch.setattr(webbrowser, "open", _intercepted_webbrowser_open)
    twin.opened_browser_urls = opened_urls  # type: ignore[attr-defined]

    try:
        yield twin
    finally:
        twin.stop()


@pytest.fixture
def cli_runner() -> CliRunner:
    """Isolated Typer CliRunner."""
    return CliRunner()


@pytest.fixture
def mcp_session(colab_twin: ColabTwin, tmp_path: Path) -> Callable[..., Any]:
    """
    Async context manager factory creating an initialized MCP ClientSession
    over 'memory', 'sse', or 'stdio' transport.
    """

    @asynccontextmanager
    async def _factory(
        transport: Literal["memory", "sse", "stdio"] = "memory",
        server_name: str = "colab-mcp-opaque-e2e",
    ) -> AsyncGenerator[ClientSession, None]:
        if transport == "memory":
            server = create_server(server_name)
            async with InMemoryTransport(server) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session
            return

        if transport == "sse":
            server = create_server(server_name)
            app = server.sse_app()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            cfg = uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                log_level="error",
                timeout_graceful_shutdown=1,
            )
            uv_srv = uvicorn.Server(cfg)

            def _run_sse() -> None:
                uv_srv.run(sockets=[sock])

            thread = threading.Thread(target=_run_sse, daemon=True, name="opaque-e2e-sse")
            thread.start()
            deadline = time.monotonic() + 5.0
            while not uv_srv.started and time.monotonic() < deadline:
                time.sleep(0.01)
            if not uv_srv.started:
                uv_srv.should_exit = True
                thread.join(timeout=2.0)
                sock.close()
                raise RuntimeError("Local uvicorn SSE server failed to start")

            try:
                async with sse_client(f"http://127.0.0.1:{port}/sse") as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        yield session
            finally:
                uv_srv.should_exit = True
                thread.join(timeout=5.0)
                try:
                    sock.close()
                except OSError:
                    pass
            return

        if transport == "stdio":
            require_cli_command("run")
            base_env = get_default_environment()
            env = colab_twin.build_subprocess_env(tmp_path, base_env=base_env, home_dir=tmp_path)
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "colab_mcp", "run", "--transport", "stdio"],
                env=env,
            )
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session
            return

        raise ValueError(f"Unsupported transport: {transport!r}")

    return _factory
