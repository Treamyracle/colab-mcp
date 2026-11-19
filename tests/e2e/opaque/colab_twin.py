"""
Stateful Local Colab Twin for offline, deterministic, opaque-box E2E testing.

Emulates:
1. Google OAuth 2.0 (`/token`, `/oauth2/v3/userinfo`)
2. Chrome DevTools Protocol (`/json/version`)
3. Google Colab Tunnel Frontend (`/tun/m/assignments`, `/ccu-info`, 2-step XSRF
   `/assign`, `/unassign/{ep}`, `/{ep}/keep-alive/`)
4. Jupyter Kernel REST (`/api/kernels`, `/interrupt`, `/restart`)
5. Real RFC 6455 WebSocket server (`websockets.serve` on `127.0.0.1:0`) speaking
   Jupyter Messaging Protocol v5.3/v5.4 (`shell`, `iopub`, `control`) with
   persistent per-kernel Python execution, ANSI tracebacks, multimodal
   `display_data`, cooperative + thread-level SIGINT, Tier-2 `/restart` wipe,
   `Sec-WebSocket-Protocol` HTTP 400 rejection, and genuine WS 1006 aborts.

ZERO imports from `colab_mcp.kernel` or `colab_mcp.notebook`.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import builtins as py_builtins
import ctypes
import io
import json
import re
import socket
import sys
import threading
import time
import traceback
import types
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple
from urllib.parse import parse_qs, urlparse

import httpx
import requests
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse
from starlette.routing import Route
import websockets
from websockets.asyncio.server import ServerConnection, serve as ws_serve

# Valid 1x1 PNG (RGBA) and 1x1 JPEG bytes
_TINY_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
)
_TINY_PNG_B64 = base64.b64encode(_TINY_PNG_BYTES).decode("ascii")

# Minimal valid 1x1 JPEG
_TINY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////"
    "////////////////////////////////////////////wgALCAABAAEBAREA/8QAFAABAAAAAAAA"
    "AAAAAAAAAAAAAP/aAAgBAQABPxA="
)

TWIN_EXTERNAL_HOSTS: frozenset[str] = frozenset({
    "colab.research.google.com",
    "colab.clients6.google.com",
    "oauth2.googleapis.com",
    "www.googleapis.com",
    "accounts.google.com",
    "sdk.cloud.google.com",
})

ACCELERATOR_RATES: Dict[str, float] = {
    "NONE": 0.0,
    "T4": 1.96,
    "L4": 3.84,
    "A100": 5.42,
    "H100": 13.50,
    "V5E1": 1.96,
    "V6E1": 4.50,
}


@dataclass
class AssignFault:
    """Configurable fault injection for /tun/m/assign."""

    status: int
    body: Any
    on_step: Literal["get", "post", "both"] = "post"
    one_shot: bool = True
    match_high_ram: Optional[bool] = None
    match_accelerator: Optional[str] = None


@dataclass
class TwinFaultConfig:
    """Runtime fault injection & protocol configuration for ColabTwin."""

    assign_error: Optional[AssignFault] = None
    oauth_refresh_mode: Literal["ok", "invalid_grant", "network_error"] = "ok"
    oauth_exchange_error: Optional[Tuple[int, Dict[str, Any]]] = None
    keepalive_mode: Literal["ok", "read_timeout", "http_404", "http_500", "connect_error"] = "ok"
    ccu_info_error: Optional[int] = None
    xssi_prefix: bool = False
    hang_on_sigint: bool = False
    drop_ws_1006_after_chunks: Optional[int] = None
    kernel_restarting_polls: int = 0
    default_proxy_token_ttl: int = 3600
    auto_create_kernel: bool = True
    unassign_404: bool = False
    shortcut_assign_step1: bool = False
    cdp_ready: bool = True
    enforce_tier_limits: bool = True
    max_assignments_per_user: Optional[int] = None
    inject_osc_in_traceback: bool = False


@dataclass
class TwinState:
    """In-memory state & telemetry recorded by ColabTwin."""

    user_emails: Dict[int, str] = field(
        default_factory=lambda: {
            0: "twin-user0@example.com",
            1: "twin-user1@example.com",
        }
    )
    user_tiers: Dict[int, str] = field(
        default_factory=lambda: {
            0: "FREE",
            1: "PRO",
        }
    )
    ccu_balances: Dict[int, float] = field(
        default_factory=lambda: {
            0: 0.0,
            1: 100.0,
        }
    )
    issued_auth_codes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    oauth_tokens: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pending_xsrf: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    assignments: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Telemetry logs for opaque assertions
    http_requests: List[Dict[str, Any]] = field(default_factory=list)
    ws_handshakes: List[Dict[str, Any]] = field(default_factory=list)
    keepalive_pings: List[Dict[str, Any]] = field(default_factory=list)
    unassigned_endpoints: List[str] = field(default_factory=list)
    restarted_kernels: List[str] = field(default_factory=list)
    interrupted_kernels: List[str] = field(default_factory=list)
    ws_frames_sent: List[Dict[str, Any]] = field(default_factory=list)
    ws_frames_received: List[Dict[str, Any]] = field(default_factory=list)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _build_jupyter_msg(
    *,
    msg_type: str,
    channel: str,
    content: Dict[str, Any],
    parent_header: Optional[Dict[str, Any]] = None,
    session_id: str = "",
) -> Dict[str, Any]:
    return {
        "header": {
            "msg_id": str(uuid.uuid4()),
            "username": "kernel",
            "session": session_id or str(uuid.uuid4()),
            "msg_type": msg_type,
            "version": "5.3",
            "date": _utc_iso(),
        },
        "parent_header": dict(parent_header or {}),
        "metadata": {},
        "content": content,
        "channel": channel,
        "buffers": [],
    }


class _StreamingWriter(io.TextIOBase):
    """Thread-safe stream writer that forwards stdout/stderr chunks to Jupyter iopub."""

    def __init__(self, name: str, emit_fn: Callable[[str, str], None]) -> None:
        super().__init__()
        self._name = name
        self._emit_fn = emit_fn

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        if s:
            self._emit_fn(self._name, s)
        return len(s)

    def flush(self) -> None:
        return None


class TwinKernel:
    """Stateful in-memory Python execution kernel speaking Jupyter v5.3/v5.4."""

    def __init__(self, kernel_id: str, endpoint: str, twin: "ColabTwin") -> None:
        self.kernel_id = kernel_id
        self.endpoint = endpoint
        self.twin = twin
        self.execution_state: str = "idle"
        self.execution_count: int = 0
        self.namespace: Dict[str, Any] = {}
        self.restart_epoch: int = 0
        self._lock = threading.Lock()
        self._interrupt_event = threading.Event()
        self._active_thread_id: Optional[int] = None
        self._active_connections: Set[ServerConnection] = set()
        self._ws_loop: Optional[asyncio.AbstractEventLoop] = None
        self._chunks_emitted_since_connect: int = 0

    def register_ws(self, ws: ServerConnection, loop: asyncio.AbstractEventLoop) -> None:
        self._ws_loop = loop
        self._active_connections.add(ws)
        self._chunks_emitted_since_connect = 0

    def unregister_ws(self, ws: ServerConnection) -> None:
        self._active_connections.discard(ws)

    def _schedule_send(self, msg: Dict[str, Any]) -> None:
        loop = self._ws_loop
        if loop is None or loop.is_closed():
            return
        payload = json.dumps(msg)
        self.twin.state.ws_frames_sent.append(msg)

        async def _send_all() -> None:
            for ws in list(self._active_connections):
                try:
                    await ws.send(payload)
                except Exception:
                    self._active_connections.discard(ws)

            drop_after = self.twin.faults.drop_ws_1006_after_chunks
            if drop_after is not None and msg.get("header", {}).get("msg_type") in (
                "stream",
                "display_data",
                "execute_result",
                "status",
            ):
                self._chunks_emitted_since_connect += 1
                if self._chunks_emitted_since_connect >= drop_after:
                    self.twin.faults.drop_ws_1006_after_chunks = None
                    for ws in list(self._active_connections):
                        transport = getattr(ws, "transport", None)
                        if transport is not None:
                            transport.abort()

        asyncio.run_coroutine_threadsafe(_send_all(), loop)

    def _interruptible_sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, float(seconds))
        while True:
            if self._interrupt_event.is_set() and not self.twin.faults.hang_on_sigint:
                raise KeyboardInterrupt("Interrupted by SIGINT")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.02, remaining))

    def _make_exec_builtins(
        self,
        emit_display: Callable[[Dict[str, Any], Optional[Dict[str, Any]]], None],
    ) -> Dict[str, Any]:
        real_import = py_builtins.__import__
        twin_sleep = self._interruptible_sleep

        def _make_pyplot_module() -> types.ModuleType:
            plt_mod = types.ModuleType("matplotlib.pyplot")
            plt_mod._plots = []

            def plot(*args: Any, **kwargs: Any) -> list[Any]:
                plt_mod._plots.append((args, kwargs))
                return []

            def title(label: str, *args: Any, **kwargs: Any) -> None:
                plt_mod._title = label

            def xlabel(*args: Any, **kwargs: Any) -> None:
                return None

            def ylabel(*args: Any, **kwargs: Any) -> None:
                return None

            def figure(*args: Any, **kwargs: Any) -> None:
                return None

            def close(*args: Any, **kwargs: Any) -> None:
                return None

            def show(*args: Any, **kwargs: Any) -> None:
                emit_display(
                    {
                        "image/png": _TINY_PNG_B64,
                        "text/plain": "<Figure size 640x480 with 1 Axes>",
                    },
                    {},
                )
                plt_mod._plots.clear()

            plt_mod.plot = plot
            plt_mod.title = title
            plt_mod.xlabel = xlabel
            plt_mod.ylabel = ylabel
            plt_mod.figure = figure
            plt_mod.close = close
            plt_mod.show = show
            return plt_mod

        def _make_matplotlib_module() -> types.ModuleType:
            mpl_mod = types.ModuleType("matplotlib")
            mpl_mod.pyplot = _make_pyplot_module()
            return mpl_mod

        def _make_torch_module() -> types.ModuleType:
            torch_mod = types.ModuleType("torch")
            nn_mod = types.ModuleType("torch.nn")

            class _Linear:
                def __init__(self, in_f: int, out_f: int) -> None:
                    self.in_features = in_f
                    self.out_features = out_f

            nn_mod.Linear = _Linear
            torch_mod.nn = nn_mod
            return torch_mod

        def _custom_import(
            name: str,
            globals: Any = None,
            locals: Any = None,
            fromlist: Any = (),
            level: int = 0,
        ) -> Any:
            if name == "matplotlib.pyplot":
                mpl = _make_matplotlib_module()
                return mpl if not fromlist else mpl.pyplot
            if name == "matplotlib":
                mpl = _make_matplotlib_module()
                if fromlist and "pyplot" in fromlist:
                    return mpl
                return mpl
            if name == "torch":
                return _make_torch_module()
            mod = real_import(name, globals, locals, fromlist, level)
            if name == "time":
                wrapped_time = types.ModuleType("time")
                for attr in dir(mod):
                    setattr(wrapped_time, attr, getattr(mod, attr))
                wrapped_time.sleep = twin_sleep
                return wrapped_time
            return mod

        b = dict(vars(py_builtins))
        b["__import__"] = _custom_import
        return b

    def execute_request(self, req: Dict[str, Any]) -> None:
        """Run a Jupyter execute_request on a worker thread so WS control stays live."""
        parent_header = dict(req.get("header") or {})
        session_id = parent_header.get("session", "")
        content = req.get("content") or {}
        code = str(content.get("code", ""))
        silent = bool(content.get("silent", False))
        store_history = bool(content.get("store_history", True))

        def _worker() -> None:
            with self._lock:
                epoch_at_start = self.restart_epoch
                self._interrupt_event.clear()
                self._active_thread_id = threading.get_ident()
                self.execution_state = "busy"
                if not silent and store_history:
                    self.execution_count += 1
                exec_count = self.execution_count

                def _is_stale() -> bool:
                    return self.restart_epoch != epoch_at_start

                def _emit(msg_type: str, channel: str, msg_content: Dict[str, Any]) -> None:
                    if _is_stale():
                        return
                    self._schedule_send(
                        _build_jupyter_msg(
                            msg_type=msg_type,
                            channel=channel,
                            content=msg_content,
                            parent_header=parent_header,
                            session_id=session_id,
                        )
                    )

                _emit("status", "iopub", {"execution_state": "busy"})
                if not silent:
                    _emit("execute_input", "iopub", {"code": code, "execution_count": exec_count})

                def _emit_stream(stream_name: str, text: str) -> None:
                    _emit("stream", "iopub", {"name": stream_name, "text": text})

                def _emit_display(data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
                    _emit(
                        "display_data",
                        "iopub",
                        {"data": data, "metadata": metadata or {}},
                    )

                def _twin_display(
                    mime_or_data: Any,
                    value: Any = None,
                    metadata: Optional[Dict[str, Any]] = None,
                ) -> None:
                    if isinstance(mime_or_data, dict):
                        _emit_display(mime_or_data, metadata)
                    else:
                        _emit_display({str(mime_or_data): value}, metadata)

                self.namespace["__builtins__"] = self._make_exec_builtins(_emit_display)
                self.namespace["twin_display"] = _twin_display
                self.namespace["TINY_PNG_B64"] = _TINY_PNG_B64
                self.namespace["TINY_JPEG_B64"] = _TINY_JPEG_B64

                out_writer = _StreamingWriter("stdout", _emit_stream)
                err_writer = _StreamingWriter("stderr", _emit_stream)

                old_stdout, old_stderr = sys.stdout, sys.stderr
                status_reply = "ok"
                error_payload: Optional[Dict[str, Any]] = None

                try:
                    sys.stdout = out_writer
                    sys.stderr = err_writer
                    tree = ast.parse(code, mode="exec")
                    if tree.body and isinstance(tree.body[-1], ast.Expr):
                        last_expr = tree.body.pop()
                        if tree.body:
                            exec(compile(tree, "<colab-twin>", "exec"), self.namespace)
                        expr_node = ast.Expression(last_expr.value)
                        ast.fix_missing_locations(expr_node)
                        val = eval(compile(expr_node, "<colab-twin>", "eval"), self.namespace)
                        if val is not None and not silent:
                            _emit(
                                "execute_result",
                                "iopub",
                                {
                                    "execution_count": exec_count,
                                    "data": {"text/plain": repr(val)},
                                    "metadata": {},
                                },
                            )
                    else:
                        exec(compile(tree, "<colab-twin>", "exec"), self.namespace)
                except BaseException as exc:
                    status_reply = "error"
                    ename = type(exc).__name__
                    evalue = str(exc)
                    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ansi_tb: List[str] = [
                        f"\x1b[31m{'-' * 75}\x1b[0m",
                        f"\x1b[31m{ename}\x1b[0m\x1b[36m: {evalue}\x1b[0m",
                    ]
                    if self.twin.faults.inject_osc_in_traceback:
                        ansi_tb.append("\x1b]8;;https://colab.research.google.com\x07Colab Trace\x1b]8;;\x07")
                    for line in tb_lines:
                        ansi_tb.append(f"\x1b[33m{line.rstrip()}\x1b[0m")
                    error_payload = {
                        "ename": ename,
                        "evalue": evalue,
                        "traceback": ansi_tb,
                    }
                    _emit("error", "iopub", error_payload)
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                    self._active_thread_id = None
                    if not _is_stale():
                        self.execution_state = "idle"
                        _emit("status", "iopub", {"execution_state": "idle"})
                        reply_content: Dict[str, Any] = {
                            "status": status_reply,
                            "execution_count": exec_count,
                        }
                        if error_payload is not None:
                            reply_content.update(error_payload)
                        _emit("execute_reply", "shell", reply_content)

        threading.Thread(target=_worker, daemon=True, name=f"twin-kernel-{self.kernel_id[:8]}").start()

    def interrupt(self) -> None:
        self.twin.state.interrupted_kernels.append(self.kernel_id)
        if self.twin.faults.hang_on_sigint:
            # Ignore SIGINT to force Tier-2 escalation after 5.0s
            return
        self._interrupt_event.set()
        tid = self._active_thread_id
        if tid is not None:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(tid),
                ctypes.py_object(KeyboardInterrupt),
            )

    def restart(self) -> None:
        self.twin.state.restarted_kernels.append(self.kernel_id)
        self.restart_epoch += 1
        self._interrupt_event.set()
        tid = self._active_thread_id
        if tid is not None:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(tid),
                ctypes.py_object(KeyboardInterrupt),
            )
        self.namespace.clear()
        self.execution_count = 0
        self.execution_state = "idle"
        loop = self._ws_loop
        if loop is not None and not loop.is_closed():
            async def _close_ws() -> None:
                for ws in list(self._active_connections):
                    try:
                        await ws.close(code=1001, reason="Kernel restarted")
                    except Exception:
                        pass
            asyncio.run_coroutine_threadsafe(_close_ws(), loop)


class ColabTwin:
    """Offline Local Colab Twin combining HTTP router + real RFC 6455 WS server."""

    def __init__(self, cdp_port: int = 9222) -> None:
        self.cdp_port = cdp_port
        self.faults = TwinFaultConfig()
        self.state = TwinState()

        self._ws_loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_ready = threading.Event()
        self._ws_stop_event: Optional[asyncio.Event] = None
        self.ws_port: int = 0

        self._http_server: Optional[uvicorn.Server] = None
        self._http_thread: Optional[threading.Thread] = None
        self._http_sock: Optional[socket.socket] = None
        self.http_port: int = 0

    # ------------------------------------------------------------------
    # Lifecycle: WebSocket Server & Optional HTTP Bridge
    # ------------------------------------------------------------------

    def start(self) -> "ColabTwin":
        if self._ws_thread is not None and self._ws_thread.is_alive():
            return self
        self._ws_ready.clear()

        def _run_ws_loop() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._ws_loop = loop
            self._ws_stop_event = asyncio.Event()

            async def _serve() -> None:
                async with ws_serve(
                    self._ws_handler,
                    host="127.0.0.1",
                    port=0,
                    subprotocols=None,
                    max_size=50 * 1024 * 1024,
                    process_request=self._ws_process_request,
                ) as srv:
                    self.ws_port = srv.sockets[0].getsockname()[1]
                    self._ws_ready.set()
                    assert self._ws_stop_event is not None
                    await self._ws_stop_event.wait()

            try:
                loop.run_until_complete(_serve())
            finally:
                loop.close()

        self._ws_thread = threading.Thread(target=_run_ws_loop, daemon=True, name="colab-twin-ws")
        self._ws_thread.start()
        if not self._ws_ready.wait(timeout=5.0):
            raise RuntimeError("ColabTwin WebSocket server failed to start within 5s")
        return self

    def stop(self) -> None:
        self.stop_http_bridge()
        if self._ws_loop is not None and self._ws_stop_event is not None and not self._ws_loop.is_closed():
            self._ws_loop.call_soon_threadsafe(self._ws_stop_event.set)
        if self._ws_thread is not None:
            self._ws_thread.join(timeout=5.0)
            self._ws_thread = None
        self._ws_loop = None

    def start_http_bridge(self) -> int:
        """Start an on-demand Starlette/uvicorn HTTP bridge on 127.0.0.1:0 for subprocesses."""
        if self._http_thread is not None and self._http_thread.is_alive():
            return self.http_port

        twin = self

        async def _proxy_all(request: StarletteRequest) -> StarletteResponse:
            body = await request.body()
            orig_host = request.headers.get("x-twin-original-host", "colab.research.google.com")
            query = f"?{request.url.query}" if request.url.query else ""
            full_url = f"https://{orig_host}{request.url.path}{query}"
            headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
            status, resp_headers, resp_body = twin.handle_http(
                request.method, full_url, headers, body
            )
            return StarletteResponse(
                content=resp_body,
                status_code=status,
                headers={
                    k: v
                    for k, v in resp_headers.items()
                    if k.lower() not in ("content-length", "transfer-encoding")
                },
            )

        app = Starlette(
            routes=[
                Route("/{path:path}", _proxy_all, methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]),
            ]
        )
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        self.http_port = sock.getsockname()[1]
        self._http_sock = sock

        cfg = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.http_port,
            log_level="error",
            timeout_graceful_shutdown=1,
        )
        srv = uvicorn.Server(cfg)
        self._http_server = srv

        def _run_http() -> None:
            srv.run(sockets=[sock])

        self._http_thread = threading.Thread(target=_run_http, daemon=True, name="colab-twin-http")
        self._http_thread.start()
        deadline = time.monotonic() + 5.0
        while not srv.started and time.monotonic() < deadline:
            time.sleep(0.01)
        if not srv.started:
            raise RuntimeError("ColabTwin HTTP bridge failed to start")
        return self.http_port

    def stop_http_bridge(self) -> None:
        if self._http_server is not None:
            self._http_server.should_exit = True
        if self._http_thread is not None:
            self._http_thread.join(timeout=5.0)
            self._http_thread = None
        if self._http_sock is not None:
            try:
                self._http_sock.close()
            except OSError:
                pass
            self._http_sock = None
        self._http_server = None

    def __enter__(self) -> "ColabTwin":
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # WebSocket Handshake & Handler
    # ------------------------------------------------------------------

    def _extract_ws_endpoint_and_kernel(self, path: str) -> Tuple[Optional[str], Optional[str]]:
        m = re.match(r"^/tun/m/([^/]+)/api/kernels/([^/]+)/channels(?:\?.*)?$", path)
        if m:
            return m.group(1), m.group(2)
        return None, None

    async def _ws_process_request(self, connection: ServerConnection, request: Any) -> Any:
        headers_lower = {k.lower(): v for k, v in request.headers.items()}
        record: Dict[str, Any] = {
            "path": request.path,
            "headers": dict(request.headers),
            "accepted": False,
            "status": 400,
            "reason": "",
        }

        if "sec-websocket-protocol" in headers_lower:
            record["status"] = 400
            record["reason"] = "Sec-WebSocket-Protocol must not be sent"
            self.state.ws_handshakes.append(record)
            return connection.respond(400, "Sec-WebSocket-Protocol header is forbidden by Colab TFE")

        endpoint, kernel_id = self._extract_ws_endpoint_and_kernel(request.path)
        if not endpoint or not kernel_id:
            record["status"] = 404
            record["reason"] = "Unknown WebSocket path"
            self.state.ws_handshakes.append(record)
            return connection.respond(404, "Kernel channel not found")

        assignment = self.state.assignments.get(endpoint)
        if assignment is None:
            record["status"] = 404
            record["reason"] = "Unknown endpoint"
            self.state.ws_handshakes.append(record)
            return connection.respond(404, "Runtime endpoint not assigned")

        proxy_token = headers_lower.get("x-colab-runtime-proxy-token")
        if not proxy_token or proxy_token != assignment["proxy_token"]:
            record["status"] = 403
            record["reason"] = "Invalid or missing X-Colab-Runtime-Proxy-Token"
            self.state.ws_handshakes.append(record)
            return connection.respond(403, "Invalid X-Colab-Runtime-Proxy-Token")

        kernel = assignment["kernels"].get(kernel_id)
        if kernel is None:
            record["status"] = 404
            record["reason"] = "Unknown kernel_id"
            self.state.ws_handshakes.append(record)
            return connection.respond(404, "Kernel not found")

        record["accepted"] = True
        record["status"] = 101
        self.state.ws_handshakes.append(record)
        return None

    async def _ws_handler(self, websocket: ServerConnection) -> None:
        path = websocket.request.path if websocket.request else ""
        endpoint, kernel_id = self._extract_ws_endpoint_and_kernel(path)
        assignment = self.state.assignments.get(endpoint or "")
        kernel: Optional[TwinKernel] = assignment["kernels"].get(kernel_id or "") if assignment else None
        if kernel is None:
            await websocket.close(code=1008, reason="Kernel not found")
            return

        loop = asyncio.get_running_loop()
        kernel.register_ws(websocket, loop)
        try:
            async for raw in websocket:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                msg = json.loads(raw)
                self.state.ws_frames_received.append(msg)
                header = msg.get("header") or {}
                msg_type = header.get("msg_type")
                channel = msg.get("channel", "shell")

                if msg_type == "execute_request":
                    kernel.execute_request(msg)
                elif msg_type == "interrupt_request" or channel == "control" and msg_type == "interrupt_request":
                    kernel.interrupt()
                    reply = _build_jupyter_msg(
                        msg_type="interrupt_reply",
                        channel="control",
                        content={"status": "ok"},
                        parent_header=header,
                        session_id=header.get("session", ""),
                    )
                    self.state.ws_frames_sent.append(reply)
                    await websocket.send(json.dumps(reply))
                elif msg_type == "kernel_info_request":
                    reply = _build_jupyter_msg(
                        msg_type="kernel_info_reply",
                        channel="shell",
                        content={
                            "status": "ok",
                            "protocol_version": "5.3",
                            "implementation": "colab-twin",
                            "language_info": {"name": "python", "version": "3.13"},
                        },
                        parent_header=header,
                        session_id=header.get("session", ""),
                    )
                    self.state.ws_frames_sent.append(reply)
                    await websocket.send(json.dumps(reply))
        except websockets.ConnectionClosed:
            pass
        finally:
            kernel.unregister_ws(websocket)

    async def drop_active_ws_1006(self) -> int:
        """Abruptly abort all active WS TCP transports to trigger genuine WS 1006."""
        dropped = 0
        if self._ws_loop is None or self._ws_loop.is_closed():
            return 0
        fut = asyncio.run_coroutine_threadsafe(self._abort_all_ws(), self._ws_loop)
        dropped = await asyncio.wrap_future(fut)
        return dropped

    async def _abort_all_ws(self) -> int:
        count = 0
        for assignment in self.state.assignments.values():
            for kernel in assignment["kernels"].values():
                for ws in list(kernel._active_connections):
                    transport = getattr(ws, "transport", None)
                    if transport is not None:
                        transport.abort()
                        count += 1
        return count

    # ------------------------------------------------------------------
    # Helpers for Issuing Tokens & Inspecting State
    # ------------------------------------------------------------------

    def issue_auth_code(
        self,
        code: str,
        auth_user_index: int = 0,
        email: Optional[str] = None,
    ) -> str:
        resolved_email = email or self.state.user_emails.get(
            auth_user_index, f"twin-user{auth_user_index}@example.com"
        )
        self.state.issued_auth_codes[code] = {
            "auth_user_index": auth_user_index,
            "email": resolved_email,
        }
        return code

    def get_active_kernels(self) -> List[TwinKernel]:
        kernels: List[TwinKernel] = []
        for assignment in self.state.assignments.values():
            kernels.extend(assignment["kernels"].values())
        return kernels

    # ------------------------------------------------------------------
    # HTTP Router (shared by httpx, requests, and Starlette bridge)
    # ------------------------------------------------------------------

    def _json_response(
        self,
        status: int,
        payload: Any,
        *,
        xssi: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, Dict[str, str], bytes]:
        text = json.dumps(payload)
        if xssi:
            text = ")]}'\n" + text
        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        return status, headers, text.encode("utf-8")

    def _check_tfe_auth(self, headers_lower: Dict[str, str]) -> Optional[int]:
        auth = headers_lower.get("authorization", "")
        if auth.startswith("Bearer "):
            token = auth[len("Bearer "):].strip()
            meta = self.state.oauth_tokens.get(token)
            if meta is not None:
                return int(meta.get("auth_user_index", 0))
            return int(headers_lower.get("x-goog-authuser", "0") or "0")
        if auth.startswith("SAPISIDHASH "):
            cookie = headers_lower.get("cookie", "")
            if "SAPISID=" in cookie:
                return int(headers_lower.get("x-goog-authuser", "0") or "0")
        return None

    def _serialize_assignment(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "endpoint": rec["endpoint"],
            "accelerator": rec["accelerator"],
            "variant": rec["variant"],
            "runtimeProxyInfo": {
                "url": rec["proxy_url"],
                "token": rec["proxy_token"],
                "tokenExpiresInSeconds": rec["token_ttl"],
            },
        }

    def handle_http(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: bytes = b"",
    ) -> Tuple[int, Dict[str, str], bytes]:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        path = parsed.path
        qs = parse_qs(parsed.query)
        headers_lower = {k.lower(): v for k, v in headers.items()}
        method_upper = method.upper()

        def _record(status: int) -> None:
            self.state.http_requests.append(
                {
                    "method": method_upper,
                    "url": url,
                    "host": host,
                    "path": path,
                    "query": {k: v[0] if len(v) == 1 else v for k, v in qs.items()},
                    "headers": dict(headers),
                    "status": status,
                }
            )

        # 1. Chrome CDP readiness
        if host in ("127.0.0.1", "localhost") and path == "/json/version":
            if not self.faults.cdp_ready:
                _record(503)
                return self._json_response(503, {"error": "CDP not ready"})
            _record(200)
            return self._json_response(
                200,
                {
                    "Browser": "Chrome/130.0.0.0",
                    "Protocol-Version": "1.3",
                    "webSocketDebuggerUrl": f"ws://127.0.0.1:{self.cdp_port}/devtools/browser/twin",
                },
            )

        # 2. Google OAuth2 Token Endpoint
        if host == "oauth2.googleapis.com" and path == "/token" and method_upper == "POST":
            form = parse_qs(body.decode("utf-8", errors="replace"))
            grant_type = form.get("grant_type", [""])[0]

            if grant_type == "authorization_code":
                code = form.get("code", [""])[0]
                if self.faults.oauth_exchange_error is not None:
                    st, err_body = self.faults.oauth_exchange_error
                    _record(st)
                    return self._json_response(st, err_body)
                if not code or code in ("invalid_code", "bad_code", "expired_code"):
                    _record(400)
                    return self._json_response(
                        400,
                        {"error": "invalid_grant", "error_description": "Invalid authorization code."},
                    )
                code_meta = self.state.issued_auth_codes.get(code, {})
                user_idx = int(code_meta.get("auth_user_index", 0))
                email = code_meta.get(
                    "email",
                    self.state.user_emails.get(user_idx, f"twin-user{user_idx}@example.com"),
                )
                access_token = f"ya29.twin_{user_idx}_{uuid.uuid4().hex[:16]}"
                refresh_token = f"1//0twin_{user_idx}_{uuid.uuid4().hex[:16]}"
                self.state.oauth_tokens[access_token] = {
                    "auth_user_index": user_idx,
                    "email": email,
                    "refresh_token": refresh_token,
                }
                self.state.oauth_tokens[refresh_token] = {
                    "auth_user_index": user_idx,
                    "email": email,
                }
                _record(200)
                return self._json_response(
                    200,
                    {
                        "access_token": access_token,
                        "expires_in": 3600,
                        "token_type": "Bearer",
                        "scope": "openid https://www.googleapis.com/auth/colaboratory",
                        "refresh_token": refresh_token,
                    },
                )

            if grant_type == "refresh_token":
                mode = self.faults.oauth_refresh_mode
                if mode == "invalid_grant":
                    _record(400)
                    return self._json_response(
                        400,
                        {"error": "invalid_grant", "error_description": "Token has been expired or revoked."},
                    )
                if mode == "network_error":
                    raise httpx.ConnectError("Simulated OAuth token refresh network error")
                rtok = form.get("refresh_token", [""])[0]
                meta = self.state.oauth_tokens.get(rtok, {})
                user_idx = int(meta.get("auth_user_index", 0))
                email = meta.get(
                    "email",
                    self.state.user_emails.get(user_idx, f"twin-user{user_idx}@example.com"),
                )
                new_access = f"ya29.twin_refreshed_{user_idx}_{uuid.uuid4().hex[:16]}"
                self.state.oauth_tokens[new_access] = {
                    "auth_user_index": user_idx,
                    "email": email,
                    "refresh_token": rtok,
                }
                _record(200)
                return self._json_response(
                    200,
                    {
                        "access_token": new_access,
                        "expires_in": 3600,
                        "token_type": "Bearer",
                        "scope": "openid https://www.googleapis.com/auth/colaboratory",
                    },
                )

            _record(400)
            return self._json_response(400, {"error": "unsupported_grant_type"})

        # 3. Google Userinfo Endpoint
        if host == "www.googleapis.com" and path == "/oauth2/v3/userinfo" and method_upper == "GET":
            auth = headers_lower.get("authorization", "")
            token = auth[len("Bearer "):].strip() if auth.startswith("Bearer ") else ""
            meta = self.state.oauth_tokens.get(token)
            email = (
                meta["email"]
                if meta
                else self.state.user_emails.get(0, "twin-user0@example.com")
            )
            _record(200)
            return self._json_response(200, {"email": email, "email_verified": True})

        # 4. Jupyter Proxy REST: /tun/m/{endpoint}/api/kernels...
        m_kernels = re.match(r"^/tun/m/([^/]+)/api/kernels(?:/([^/]+)(?:/(interrupt|restart))?)?$", path)
        if m_kernels:
            endpoint = m_kernels.group(1)
            kernel_id = m_kernels.group(2)
            action = m_kernels.group(3)
            assignment = self.state.assignments.get(endpoint)
            if assignment is None:
                _record(404)
                return self._json_response(404, {"error": "Endpoint not found"})
            proxy_token = headers_lower.get("x-colab-runtime-proxy-token", "")
            if proxy_token != assignment["proxy_token"]:
                _record(403)
                return self._json_response(403, {"error": "Forbidden: invalid proxy token"})

            kernels: Dict[str, TwinKernel] = assignment["kernels"]
            if kernel_id is None:
                if method_upper == "GET":
                    if self.faults.kernel_restarting_polls > 0:
                        self.faults.kernel_restarting_polls -= 1
                        payload = [
                            {"id": k.kernel_id, "name": "python3", "execution_state": "restarting"}
                            for k in kernels.values()
                        ]
                        _record(200)
                        return self._json_response(200, payload)
                    payload = [
                        {"id": k.kernel_id, "name": "python3", "execution_state": k.execution_state}
                        for k in kernels.values()
                    ]
                    _record(200)
                    return self._json_response(200, payload)
                if method_upper == "POST":
                    new_kid = str(uuid.uuid4())
                    k = TwinKernel(new_kid, endpoint, self)
                    kernels[new_kid] = k
                    _record(201)
                    return self._json_response(
                        201,
                        {"id": new_kid, "name": "python3", "execution_state": k.execution_state},
                    )
            else:
                k = kernels.get(kernel_id)
                if k is None:
                    _record(404)
                    return self._json_response(404, {"error": "Kernel not found"})
                if action == "interrupt" and method_upper == "POST":
                    k.interrupt()
                    _record(204)
                    return 204, {}, b""
                if action == "restart" and method_upper == "POST":
                    k.restart()
                    _record(200)
                    return self._json_response(
                        200,
                        {"id": k.kernel_id, "name": "python3", "execution_state": k.execution_state},
                    )
                if action is None and method_upper == "GET":
                    _record(200)
                    return self._json_response(
                        200,
                        {"id": k.kernel_id, "name": "python3", "execution_state": k.execution_state},
                    )

        # 5. Colab TFE Keep-Alive: GET /tun/m/{endpoint}/keep-alive/
        m_ka = re.match(r"^/tun/m/([^/]+)/keep-alive/?$", path)
        if m_ka and method_upper == "GET":
            endpoint = m_ka.group(1)
            if headers_lower.get("x-colab-tunnel") != "Google":
                _record(400)
                return self._json_response(400, {"error": "Missing X-Colab-Tunnel: Google"})
            self.state.keepalive_pings.append(
                {
                    "endpoint": endpoint,
                    "headers": dict(headers),
                    "mode": self.faults.keepalive_mode,
                    "timestamp": time.time(),
                }
            )
            mode = self.faults.keepalive_mode
            if mode == "read_timeout":
                raise httpx.ReadTimeout("Simulated Colab keep-alive socket hold")
            if mode == "connect_error":
                raise httpx.ConnectError("Simulated keep-alive connect error")
            if mode == "http_404":
                _record(404)
                return self._json_response(404, {"error": "Endpoint gone"})
            if mode == "http_500":
                _record(500)
                return self._json_response(500, {"error": "Internal error"})
            _record(200)
            return self._json_response(200, {"status": "alive"})

        # 6. Colab TFE: /tun/m/...
        if path.startswith("/tun/m/"):
            auth_user = self._check_tfe_auth(headers_lower)
            if auth_user is None:
                _record(401)
                return self._json_response(401, {"error": "Unauthorized"})
            if "authuser" in qs:
                auth_user = int(qs["authuser"][0])

            # GET /tun/m/assignments
            if path == "/tun/m/assignments" and method_upper == "GET":
                items = [
                    self._serialize_assignment(rec)
                    for rec in self.state.assignments.values()
                    if rec["auth_user_index"] == auth_user
                ]
                _record(200)
                return self._json_response(200, items, xssi=self.faults.xssi_prefix)

            # GET /tun/m/ccu-info
            if path == "/tun/m/ccu-info" and method_upper == "GET":
                if self.faults.ccu_info_error is not None:
                    st = self.faults.ccu_info_error
                    _record(st)
                    return self._json_response(st, {"error": "CCU service error"})
                tier = self.state.user_tiers.get(auth_user, "FREE")
                balance = self.state.ccu_balances.get(auth_user, 0.0 if tier == "FREE" else 100.0)
                user_assignments = [
                    rec for rec in self.state.assignments.values() if rec["auth_user_index"] == auth_user
                ]
                burn_rate = sum(
                    ACCELERATOR_RATES.get(rec["accelerator"], 0.0) for rec in user_assignments
                )
                _record(200)
                return self._json_response(
                    200,
                    {
                        "user_tier": tier,
                        "compute_units_remaining": balance,
                        "consumption_rate_per_hour": burn_rate,
                        "active_assignments_count": len(user_assignments),
                        "auto_refill_enabled": False,
                    },
                    xssi=self.faults.xssi_prefix,
                )

            # GET / POST /tun/m/assign
            if path == "/tun/m/assign" and method_upper in ("GET", "POST"):
                nbh = qs.get("nbh", [""])[0]
                if nbh and len(nbh) != 44:
                    _record(400)
                    return self._json_response(400, {"error": "Invalid nbh length; expected 44 chars"})

                accelerator = qs.get("accelerator", ["NONE"])[0].upper()
                variant = qs.get("variant", ["DEFAULT"])[0].upper()
                high_ram = qs.get("shape", [""])[0] == "hm"

                # Check injected assign_error
                fault = self.faults.assign_error
                if fault is not None:
                    step_match = (
                        fault.on_step == "both"
                        or (fault.on_step == "get" and method_upper == "GET")
                        or (fault.on_step == "post" and method_upper == "POST")
                    )
                    hr_match = fault.match_high_ram is None or fault.match_high_ram == high_ram
                    acc_match = fault.match_accelerator is None or fault.match_accelerator.upper() == accelerator
                    if step_match and hr_match and acc_match:
                        if fault.one_shot:
                            self.faults.assign_error = None
                        _record(fault.status)
                        return self._json_response(fault.status, fault.body, xssi=self.faults.xssi_prefix)

                # Natural tier limit checks
                tier = self.state.user_tiers.get(auth_user, "FREE")
                if self.faults.enforce_tier_limits and tier == "FREE" and high_ram:
                    _record(400)
                    return self._json_response(
                        400,
                        {"error": {"code": 400, "message": "High-RAM shape is not available on Free Tier"}},
                    )

                if self.faults.max_assignments_per_user is not None:
                    active_count = sum(
                        1 for rec in self.state.assignments.values() if rec["auth_user_index"] == auth_user
                    )
                    if active_count >= self.faults.max_assignments_per_user:
                        _record(412)
                        return self._json_response(
                            412,
                            {"error": {"code": "TOO_MANY_ASSIGNMENTS", "message": "Too many assignments"}},
                        )

                def _create_assignment() -> Dict[str, Any]:
                    ep = f"colab-twin-{uuid.uuid4().hex[:12]}"
                    proxy_token = f"ya29.twin_proxy_{uuid.uuid4().hex[:16]}"
                    kernels: Dict[str, TwinKernel] = {}
                    if self.faults.auto_create_kernel:
                        kid = str(uuid.uuid4())
                        kernels[kid] = TwinKernel(kid, ep, self)
                    rec = {
                        "endpoint": ep,
                        "auth_user_index": auth_user,
                        "accelerator": accelerator,
                        "variant": variant,
                        "high_ram": high_ram,
                        "proxy_url": f"https://colab.research.google.com/tun/m/{ep}",
                        "proxy_token": proxy_token,
                        "token_ttl": self.faults.default_proxy_token_ttl,
                        "created_at": time.time(),
                        "kernels": kernels,
                    }
                    self.state.assignments[ep] = rec
                    return rec

                if method_upper == "GET":
                    if self.faults.shortcut_assign_step1:
                        rec = _create_assignment()
                        _record(200)
                        return self._json_response(
                            200, self._serialize_assignment(rec), xssi=self.faults.xssi_prefix
                        )
                    xsrf = f"xsrf-{uuid.uuid4().hex[:16]}"
                    self.state.pending_xsrf[xsrf] = {"auth_user_index": auth_user, "op": "assign"}
                    _record(200)
                    return self._json_response(
                        200,
                        {"token": xsrf},
                        xssi=self.faults.xssi_prefix,
                        extra_headers={"Set-Cookie": f"XSRF-TOKEN={xsrf}; Path=/"},
                    )

                # POST /tun/m/assign
                xsrf = headers_lower.get("x-goog-colab-token", "")
                if not xsrf or xsrf not in self.state.pending_xsrf:
                    _record(403)
                    return self._json_response(403, {"error": "Invalid or missing X-Goog-Colab-Token"})
                self.state.pending_xsrf.pop(xsrf, None)
                rec = _create_assignment()
                _record(200)
                return self._json_response(
                    200, self._serialize_assignment(rec), xssi=self.faults.xssi_prefix
                )

            # GET / POST /tun/m/unassign/{endpoint}
            m_unassign = re.match(r"^/tun/m/unassign/([^/]+)$", path)
            if m_unassign and method_upper in ("GET", "POST"):
                endpoint = m_unassign.group(1)
                if self.faults.unassign_404 or endpoint not in self.state.assignments:
                    _record(404)
                    return self._json_response(404, {"error": "Runtime not found"})
                if method_upper == "GET":
                    xsrf = f"unassign-xsrf-{uuid.uuid4().hex[:16]}"
                    self.state.pending_xsrf[xsrf] = {"endpoint": endpoint, "op": "unassign"}
                    _record(200)
                    return self._json_response(200, {"token": xsrf}, xssi=self.faults.xssi_prefix)

                xsrf = headers_lower.get("x-goog-colab-token", "")
                if not xsrf or xsrf not in self.state.pending_xsrf:
                    _record(403)
                    return self._json_response(403, {"error": "Invalid X-Goog-Colab-Token"})
                self.state.pending_xsrf.pop(xsrf, None)
                rec = self.state.assignments.pop(endpoint, None)
                if rec is not None:
                    for k in rec["kernels"].values():
                        k.restart()
                self.state.unassigned_endpoints.append(endpoint)
                _record(200)
                return self._json_response(
                    200,
                    {"status": "unassigned", "endpoint": endpoint},
                    xssi=self.faults.xssi_prefix,
                )

        _record(404)
        return self._json_response(404, {"error": f"Unhandled twin route: {method_upper} {url}"})

    # ------------------------------------------------------------------
    # Client Adapters (httpx, requests, and stdio subprocess bootstrap)
    # ------------------------------------------------------------------

    def handle_httpx_request(self, request: httpx.Request) -> httpx.Response:
        body = request.content or b""
        status, headers, resp_body = self.handle_http(
            request.method,
            str(request.url),
            dict(request.headers),
            body,
        )
        return httpx.Response(
            status_code=status,
            headers=headers,
            content=resp_body,
            request=request,
        )

    def handle_requests_prepared(self, prepared: requests.PreparedRequest) -> requests.Response:
        body = prepared.body or b""
        if isinstance(body, str):
            body = body.encode("utf-8")
        status, headers, resp_body = self.handle_http(
            prepared.method or "GET",
            prepared.url or "",
            dict(prepared.headers or {}),
            body,
        )
        resp = requests.Response()
        resp.status_code = status
        resp.url = prepared.url or ""
        resp.headers.update(headers)
        resp._content = resp_body
        resp.request = prepared
        return resp

    def build_subprocess_env(
        self,
        tmp_path: Path,
        base_env: Optional[Dict[str, str]] = None,
        home_dir: Optional[Path] = None,
    ) -> Dict[str, str]:
        """
        Build isolated environment + sitecustomize.py bootstrap for stdio_client
        subprocesses so they route all external twin traffic to local bridges and
        never touch host keyring, registry MachineGuid, or ~/.colabmcp.
        """
        http_port = self.start_http_bridge()
        bootstrap_dir = tmp_path / "_twin_bootstrap"
        bootstrap_dir.mkdir(parents=True, exist_ok=True)
        resolved_home = home_dir or tmp_path
        resolved_home.mkdir(parents=True, exist_ok=True)

        sitecustomize_code = f'''
import os
from urllib.parse import urlparse, urlunparse

_HTTP_PORT = int(os.environ.get("COLAB_TWIN_HTTP_PORT", "{http_port}"))
_WS_PORT = int(os.environ.get("COLAB_TWIN_WS_PORT", "{self.ws_port}"))
_CDP_PORT = int(os.environ.get("COLAB_TWIN_CDP_PORT", "{self.cdp_port}"))
_TWIN_HOSTS = {sorted(TWIN_EXTERNAL_HOSTS)!r}

try:
    import colab_mcp.auth.keystore as _ks
    _ks.get_machine_id_bytes = lambda: b"DETERMINISTIC_TEST_MACHINE_ID_BYTES_12345"
except Exception:
    pass

def _rewrite_http_url(url_str):
    p = urlparse(str(url_str))
    host = (p.hostname or "").lower()
    if host in _TWIN_HOSTS or (host in ("127.0.0.1", "localhost") and p.port == _CDP_PORT):
        new_url = urlunparse(("http", f"127.0.0.1:{{_HTTP_PORT}}", p.path, p.params, p.query, p.fragment))
        return new_url, host
    if host in ("127.0.0.1", "localhost", "::1"):
        return None, host
    raise RuntimeError(f"Offline twin guard blocked outbound HTTP request to unmocked host: {{url_str}}")

try:
    import httpx
    _orig_async_send = httpx.AsyncClient.send
    _orig_sync_send = httpx.Client.send

    async def _wrapped_async_send(self, request, *args, **kwargs):
        new_url, orig_host = _rewrite_http_url(request.url)
        if new_url is not None:
            request.url = httpx.URL(new_url)
            request.headers["X-Twin-Original-Host"] = orig_host
            request.headers["Host"] = f"127.0.0.1:{{_HTTP_PORT}}"
        return await _orig_async_send(self, request, *args, **kwargs)

    def _wrapped_sync_send(self, request, *args, **kwargs):
        new_url, orig_host = _rewrite_http_url(request.url)
        if new_url is not None:
            request.url = httpx.URL(new_url)
            request.headers["X-Twin-Original-Host"] = orig_host
            request.headers["Host"] = f"127.0.0.1:{{_HTTP_PORT}}"
        return _orig_sync_send(self, request, *args, **kwargs)

    httpx.AsyncClient.send = _wrapped_async_send
    httpx.Client.send = _wrapped_sync_send
except Exception:
    pass

try:
    import requests
    _orig_req_send = requests.Session.send

    def _wrapped_req_send(self, request, **kwargs):
        new_url, orig_host = _rewrite_http_url(request.url)
        if new_url is not None:
            request.url = new_url
            request.headers["X-Twin-Original-Host"] = orig_host
        return _orig_req_send(self, request, **kwargs)

    requests.Session.send = _wrapped_req_send
except Exception:
    pass

try:
    import websockets
    _orig_ws_connect = websockets.connect

    def _wrapped_ws_connect(uri, *args, **kwargs):
        p = urlparse(str(uri))
        host = (p.hostname or "").lower()
        if p.scheme == "wss" and host == "colab.research.google.com":
            rewritten = urlunparse(("ws", f"127.0.0.1:{{_WS_PORT}}", p.path, p.params, p.query, p.fragment))
            return _orig_ws_connect(rewritten, *args, **kwargs)
        if host in ("127.0.0.1", "localhost"):
            return _orig_ws_connect(uri, *args, **kwargs)
        raise RuntimeError(f"Offline twin guard blocked outbound WebSocket to unmocked host: {{uri}}")

    websockets.connect = _wrapped_ws_connect
except Exception:
    pass
'''
        (bootstrap_dir / "sitecustomize.py").write_text(sitecustomize_code, encoding="utf-8")

        repo_src = str(Path(__file__).resolve().parents[3] / "src")
        env = dict(base_env or {})
        env.update(
            {
                "PYTHONPATH": f"{bootstrap_dir}{';' if sys.platform == 'win32' else ':'}{repo_src}",
                "USERPROFILE": str(resolved_home),
                "HOME": str(resolved_home),
                "CLOUDSDK_CONFIG": str(resolved_home / "gcloud"),
                "NO_GCE_CHECK": "true",
                "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
                "COLAB_TWIN_HTTP_PORT": str(http_port),
                "COLAB_TWIN_WS_PORT": str(self.ws_port),
                "COLAB_TWIN_CDP_PORT": str(self.cdp_port),
            }
        )
        env.pop("COLAB_MCP_KEY", None)
        env.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        return env
