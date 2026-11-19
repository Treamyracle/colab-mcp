"""
Jupyter Kernel REST Discovery, RFC 6455 WebSocket Bridge, Multiplexed Execution Client,
Tiered Interruption, and 6-Stage Auto-Reconnection (TECH_SPEC §5.1-5.6).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

import httpx
import websockets
from mcp.types import ImageContent, TextContent

from colab_mcp.kernel.correlator import KernelMessageCorrelator
from colab_mcp.kernel.mime_converter import (
    MCPContentBlock,
    convert_iopub_to_mcp_blocks,
    sanitize_ansi_text,
    sanitize_ansi_traceback,
)
from colab_mcp.kernel.models import (
    ExecutionEvent,
    InterruptExecutionResponse,
    KernelConnectionError,
    KernelExecutionTimeoutError,
    KernelState,
    RawExecutionResult,
    build_execute_request,
    build_interrupt_request,
)
from colab_mcp.runtime.models import KernelConnectionParams
from colab_mcp.runtime.session_manager import get_session_manager
from colab_mcp.utils.logging import get_logger
from colab_mcp.utils.redaction import redact_credentials

logger = get_logger(__name__)

DEFAULT_WS_PING_INTERVAL = 20
DEFAULT_WS_PING_TIMEOUT = 20
DEFAULT_WS_MAX_SIZE = 50 * 1024 * 1024  # 50 MB


def _build_ws_url(
    proxy_url: str,
    kernel_id: str,
    session_id: Optional[str] = None,
) -> str:
    """Convert an HTTP(S) proxy URL into a WS(S) Jupyter kernel channels URL."""
    base = proxy_url.rstrip("/")
    if base.startswith("https://"):
        ws_base = "wss://" + base[len("https://") :]
    elif base.startswith("http://"):
        ws_base = "ws://" + base[len("http://") :]
    else:
        ws_base = base
    url = f"{ws_base}/api/kernels/{kernel_id}/channels"
    if session_id:
        return f"{url}?session_id={session_id}"
    return url


def _select_ws_headers_param() -> str:
    """
    Determine whether the active ``websockets.connect`` expects
    ``additional_headers`` (v13+/v16+) or ``extra_headers`` (<v13).
    Falls back to ``websockets.asyncio.client.connect`` when mocked with AsyncMock.
    """
    try:
        params = inspect.signature(websockets.connect).parameters
        if "additional_headers" in params:
            return "additional_headers"
        if "extra_headers" in params:
            return "extra_headers"
    except (TypeError, ValueError):
        pass

    try:
        import websockets.asyncio.client as ws_asyncio_client  # type: ignore[import-untyped]

        async_params = inspect.signature(ws_asyncio_client.connect).parameters
        if "additional_headers" in async_params:
            return "additional_headers"
        if "extra_headers" in async_params:
            return "extra_headers"
    except Exception:
        pass

    return "additional_headers"


async def discover_or_create_kernel(
    proxy_url: str,
    proxy_token: str,
    client_agent: str = "colab-mcp",
    timeout: float = 15.0,
) -> str:
    """
    Discover an existing Jupyter kernel via ``GET /api/kernels`` or create a new
    ``python3`` kernel via ``POST /api/kernels`` (TECH_SPEC §5.1).
    """
    kernels_url = f"{proxy_url.rstrip('/')}/api/kernels"
    headers = {
        "X-Colab-Runtime-Proxy-Token": proxy_token,
        "X-Colab-Client-Agent": client_agent,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as http:
            resp = await http.get(kernels_url, headers=headers)
            resp.raise_for_status()
            kernels = resp.json()
            if isinstance(kernels, list) and len(kernels) > 0:
                first = kernels[0]
                if isinstance(first, dict) and first.get("id"):
                    return str(first["id"])

            create_resp = await http.post(
                kernels_url,
                headers=headers,
                json={"name": "python3"},
            )
            create_resp.raise_for_status()
            created = create_resp.json()
            if isinstance(created, dict) and created.get("id"):
                return str(created["id"])
    except KernelConnectionError:
        raise
    except Exception as exc:
        raise KernelConnectionError(
            f"Failed to discover or create Jupyter kernel at {kernels_url}: {exc}"
        ) from exc

    raise KernelConnectionError("Kernel discovery/creation response did not contain a valid kernel id.")


resolve_or_create_kernel = discover_or_create_kernel


async def connect_colab_kernel_ws(
    proxy_url: str,
    proxy_token: str,
    kernel_id: str,
    client_agent: str = "colab-mcp",
    ping_interval: float = DEFAULT_WS_PING_INTERVAL,
    ping_timeout: float = DEFAULT_WS_PING_TIMEOUT,
    max_size: int = DEFAULT_WS_MAX_SIZE,
    session_id: Optional[str] = None,
) -> Any:
    """
    Establish a resilient RFC 6455 compliant WebSocket tunnel to a Colab Jupyter kernel.

    Always sets ``subprotocols=None`` so no empty ``Sec-WebSocket-Protocol`` header
    is emitted, disables permessage-deflate compression for Colab's proxy, includes
    ``?session_id=``, and dynamically adapts between ``additional_headers`` and
    ``extra_headers``.
    """
    resolved_session_id = session_id or uuid.uuid4().hex
    ws_url = _build_ws_url(proxy_url, kernel_id, session_id=resolved_session_id)
    headers = {
        "X-Colab-Runtime-Proxy-Token": proxy_token,
        "X-Colab-Client-Agent": client_agent,
    }
    headers_param = _select_ws_headers_param()
    ws_kwargs: Dict[str, Any] = {
        "subprotocols": None,
        # Google Colab's reverse proxy stalls the WebSocket opening handshake when
        # permessage-deflate is advertised, so disable compression explicitly.
        "compression": None,
        "ping_interval": ping_interval,
        "ping_timeout": ping_timeout,
        "max_size": max_size,
        headers_param: headers,
    }
    return await websockets.connect(ws_url, **ws_kwargs)


connect_kernel_ws = connect_colab_kernel_ws


class KernelWebSocketClient:
    """
    Multiplexed Jupyter kernel WebSocket client managing execution correlation,
    streaming events, tiered interruption/restart, and 6-stage auto-reconnection.
    """

    def __init__(
        self,
        proxy_url: str,
        proxy_token: str,
        kernel_id: Optional[str] = None,
        runtime_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        client_agent: str = "colab-mcp",
        session_id: Optional[str] = None,
        correlator: Optional[KernelMessageCorrelator] = None,
        session_manager: Optional[Any] = None,
    ) -> None:
        self.proxy_url = proxy_url.rstrip("/")
        self.proxy_token = proxy_token
        self.kernel_id = kernel_id
        self.runtime_id = runtime_id or endpoint
        self.endpoint = self.runtime_id
        self.client_agent = client_agent
        self.session_id = session_id or uuid.uuid4().hex
        self.correlator = correlator or KernelMessageCorrelator()
        self._session_manager = session_manager

        self._ws: Any = None
        self._reader_task: Optional[asyncio.Task[Any]] = None
        self._reconnect_task: Optional[asyncio.Task[Any]] = None
        self._closing: bool = False
        self._kernel_state: str = KernelState.DISCONNECTED.value
        self._idle_event: asyncio.Event = asyncio.Event()
        self._connect_lock: asyncio.Lock = asyncio.Lock()
        self._send_lock: asyncio.Lock = asyncio.Lock()
        self._reconnect_lock: asyncio.Lock = asyncio.Lock()

    @classmethod
    def from_connection_params(
        cls,
        params: KernelConnectionParams,
        **kwargs: Any,
    ) -> "KernelWebSocketClient":
        """Instantiate a client directly from M2 ``KernelConnectionParams``."""
        return cls(
            proxy_url=params.proxy_url,
            proxy_token=params.proxy_token,
            kernel_id=params.kernel_id,
            runtime_id=params.endpoint,
            client_agent=params.client_agent,
            **kwargs,
        )

    def _get_mgr(self) -> Any:
        if self._session_manager is not None:
            return self._session_manager
        try:
            return get_session_manager()
        except Exception:
            return None

    def _record_activity(self) -> None:
        mgr = self._get_mgr()
        if mgr is not None and hasattr(mgr, "record_activity"):
            try:
                mgr.record_activity(endpoint=self.runtime_id)
            except Exception:
                pass

    def _sync_session_state(
        self,
        *,
        kernel_id: Optional[str] = None,
        kernel_state: Optional[str] = None,
    ) -> None:
        if not self.runtime_id:
            return
        mgr = self._get_mgr()
        if mgr is None or not hasattr(mgr, "get_session"):
            return
        try:
            session = mgr.get_session(self.runtime_id)
            if session is not None:
                if kernel_id is not None:
                    session.kernel_id = kernel_id
                if kernel_state is not None:
                    session.kernel_state = kernel_state
        except Exception:
            pass

    def _http_headers(self) -> Dict[str, str]:
        return {
            "X-Colab-Runtime-Proxy-Token": self.proxy_token,
            "X-Colab-Client-Agent": self.client_agent,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    @property
    def kernel_state(self) -> str:
        return self._kernel_state

    @property
    def is_connected(self) -> bool:
        """Return True if an active open WebSocket connection exists."""
        ws = self._ws
        if ws is None or self._closing:
            return False

        state = getattr(ws, "state", None)
        if state is not None:
            state_name = getattr(state, "name", None)
            if isinstance(state_name, str):
                return state_name.upper() == "OPEN"
            if isinstance(state, int):
                return state == 1

        closed = getattr(ws, "closed", None)
        if isinstance(closed, bool):
            return not closed

        open_attr = getattr(ws, "open", None)
        if isinstance(open_attr, bool):
            return open_attr

        return True

    async def connect(self) -> None:
        """Discover kernel if missing, establish WebSocket, and start background reader."""
        async with self._connect_lock:
            if self.is_connected:
                return

            self._closing = False
            if not self.kernel_id:
                self.kernel_id = await discover_or_create_kernel(
                    proxy_url=self.proxy_url,
                    proxy_token=self.proxy_token,
                    client_agent=self.client_agent,
                )

            self._ws = await connect_colab_kernel_ws(
                proxy_url=self.proxy_url,
                proxy_token=self.proxy_token,
                kernel_id=self.kernel_id,
                client_agent=self.client_agent,
                session_id=self.session_id,
            )
            self._kernel_state = KernelState.IDLE.value
            self._idle_event.set()
            await self._cancel_reader_task()
            self._reader_task = asyncio.create_task(self._reader_loop())
            self._sync_session_state(
                kernel_id=self.kernel_id,
                kernel_state=self._kernel_state,
            )

    async def _cancel_reader_task(self) -> None:
        task = self._reader_task
        self._reader_task = None
        if task is None or task.done():
            return
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        if task is current:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def _cancel_reconnect_task(self) -> None:
        task = self._reconnect_task
        self._reconnect_task = None
        if task is None or task.done():
            return
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        if task is current:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def disconnect(self) -> None:
        """Cleanly close the WebSocket connection and stop background reader tasks."""
        self._closing = True
        await self._cancel_reconnect_task()
        await self._cancel_reader_task()

        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                await ws.close()
            except Exception as exc:
                logger.debug("Error closing kernel WebSocket: %s", exc)

        self.correlator.flush_all(
            terminal_message={"_internal_type": "connection_lost", "reason": "disconnected"}
        )
        self._kernel_state = KernelState.DISCONNECTED.value
        self._idle_event.clear()
        self._sync_session_state(kernel_state=self._kernel_state)

    def _schedule_reconnect(self) -> None:
        if self._closing:
            return
        if self._reconnect_task is not None and not self._reconnect_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        async def _run_reconnect() -> None:
            try:
                await self.reconnect()
            except Exception as exc:
                logger.warning("Background kernel reconnection failed: %s", exc)

        self._reconnect_task = loop.create_task(_run_reconnect())

    async def _reader_loop(self) -> None:
        """Continuously read multiplexed Jupyter frames and dispatch via correlator."""
        ws = self._ws
        try:
            while ws is not None and not self._closing:
                raw = await ws.recv()
                # Guard against AsyncMock .recv() returning non-wire objects in unit tests
                if not isinstance(raw, (str, bytes, bytearray)):
                    break

                if isinstance(raw, (bytes, bytearray)):
                    text = raw.decode("utf-8", errors="replace")
                else:
                    text = raw

                try:
                    msg = json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    continue

                if not isinstance(msg, dict):
                    continue

                header = msg.get("header") if isinstance(msg.get("header"), dict) else {}
                msg_type = header.get("msg_type") or msg.get("msg_type")
                content = msg.get("content") if isinstance(msg.get("content"), dict) else {}

                if msg_type in ("stream", "display_data", "execute_result", "execute_reply"):
                    self._record_activity()

                if msg_type == "status":
                    exec_state = content.get("execution_state")
                    if exec_state is not None:
                        state_str = str(exec_state)
                        self._kernel_state = state_str
                        if state_str == KernelState.IDLE.value:
                            self._idle_event.set()
                        elif state_str in (
                            KernelState.BUSY.value,
                            KernelState.STARTING.value,
                            KernelState.RESTARTING.value,
                            KernelState.AUTORESTARTING.value,
                        ):
                            self._idle_event.clear()
                        self._sync_session_state(kernel_state=state_str)

                self.correlator.route_incoming_message(msg)

                if (
                    msg_type == "status"
                    and content.get("execution_state")
                    in (KernelState.RESTARTING.value, KernelState.AUTORESTARTING.value)
                    and not self._closing
                ):
                    logger.warning("Kernel reported '%s'; initiating auto-reconnection.", content.get("execution_state"))
                    self.correlator.flush_all(
                        terminal_message={
                            "_internal_type": "connection_lost",
                            "reason": "kernel_restarting",
                        }
                    )
                    self._schedule_reconnect()
                    break

        except asyncio.CancelledError:
            return
        except websockets.exceptions.ConnectionClosed as exc:
            if not self._closing:
                logger.warning("Kernel WebSocket closed abnormally (%s); triggering reconnect.", exc)
                self._kernel_state = KernelState.DISCONNECTED.value
                self._idle_event.clear()
                self._sync_session_state(kernel_state=self._kernel_state)
                self.correlator.flush_all(
                    terminal_message={
                        "_internal_type": "connection_lost",
                        "reason": f"ws_closed:{getattr(exc, 'code', 'unknown')}",
                    }
                )
                self._schedule_reconnect()
        except Exception as exc:
            if not self._closing:
                logger.warning("Kernel WebSocket reader terminated with error: %s", exc)
                self._kernel_state = KernelState.DISCONNECTED.value
                self._idle_event.clear()
                self._sync_session_state(kernel_state=self._kernel_state)
                self.correlator.flush_all(
                    terminal_message={"_internal_type": "connection_lost", "reason": str(exc)}
                )
                self._schedule_reconnect()

    async def _send_frame(self, frame: Dict[str, Any]) -> None:
        async with self._send_lock:
            if not self.is_connected or self._ws is None:
                raise KernelConnectionError("Kernel WebSocket is not connected.")
            await self._ws.send(json.dumps(frame))

    async def _dispatch_timeout_sigint(self) -> None:
        """Best-effort out-of-band SIGINT when an execution hits timeout_seconds."""
        if self.is_connected and self._ws is not None:
            try:
                frame = build_interrupt_request(session_id=self.session_id)
                await self._send_frame(frame)
            except Exception:
                pass
        try:
            await self.interrupt()
        except Exception:
            pass

    async def execute_raw(
        self,
        code: str,
        timeout_seconds: int = 300,
        store_history: bool = True,
        stop_on_error: bool = True,
        silent: bool = False,
        on_message: Optional[Callable[[Dict[str, Any]], Any]] = None,
        execution_id: Optional[str] = None,
    ) -> RawExecutionResult:
        """
        Execute ``code`` over Jupyter WebSocket and capture raw ``iopub``/``shell`` frames.

        Terminates only when BOTH ``execute_reply`` (shell) AND ``status: idle`` (iopub)
        have been observed, regardless of arrival order.
        """
        self._record_activity()
        if not self.is_connected:
            await self.connect()

        frame = build_execute_request(
            code=code,
            session_id=self.session_id,
            silent=silent,
            store_history=store_history,
            allow_stdin=False,
            stop_on_error=stop_on_error,
        )
        msg_id = str(frame["header"]["msg_id"])
        queue = self.correlator.register_execution(msg_id)
        start_mono = time.monotonic()

        iopub_messages: List[Dict[str, Any]] = []
        reply_message: Optional[Dict[str, Any]] = None
        execution_count: Optional[int] = None
        reply_status: str = "ok"
        has_error = False
        error_name: Optional[str] = None
        error_value: Optional[str] = None
        traceback_lines: Optional[List[str]] = None

        got_execute_reply = False
        got_idle_status = False

        try:
            await self._send_frame(frame)
            loop = asyncio.get_running_loop()
            deadline = loop.time() + float(timeout_seconds)

            while not (got_execute_reply and got_idle_status):
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError()

                msg = await asyncio.wait_for(queue.get(), timeout=remaining)
                if not isinstance(msg, dict):
                    continue

                if msg.get("_internal_type") == "connection_lost":
                    raise KernelConnectionError(
                        f"Kernel WebSocket connection lost during execution: {msg.get('reason', 'unknown')}"
                    )

                if on_message is not None:
                    res = on_message(msg)
                    if inspect.isawaitable(res):
                        await res

                header = msg.get("header") if isinstance(msg.get("header"), dict) else {}
                msg_type = header.get("msg_type") or msg.get("msg_type")
                content = msg.get("content") if isinstance(msg.get("content"), dict) else {}

                if msg_type in ("stream", "display_data", "execute_result", "execute_reply"):
                    self._record_activity()

                if msg_type == "status":
                    if content.get("execution_state") == KernelState.IDLE.value:
                        got_idle_status = True
                elif msg_type == "execute_reply":
                    got_execute_reply = True
                    reply_message = msg
                    raw_status = str(content.get("status", "ok"))
                    if raw_status in ("ok", "error", "abort"):
                        reply_status = raw_status
                    if content.get("execution_count") is not None:
                        execution_count = int(content["execution_count"])
                    if raw_status == "error":
                        has_error = True
                        error_name = error_name or content.get("ename")
                        error_value = error_value or content.get("evalue")
                        if traceback_lines is None and isinstance(content.get("traceback"), list):
                            traceback_lines = [str(line) for line in content["traceback"]]
                    elif raw_status == "abort":
                        has_error = True
                elif msg_type in ("stream", "display_data", "execute_result"):
                    iopub_messages.append(msg)
                    if msg_type == "execute_result" and content.get("execution_count") is not None:
                        execution_count = int(content["execution_count"])
                elif msg_type == "error":
                    iopub_messages.append(msg)
                    has_error = True
                    error_name = str(content.get("ename", "Error"))
                    error_value = str(content.get("evalue", ""))
                    raw_tb = content.get("traceback")
                    if isinstance(raw_tb, list):
                        traceback_lines = [str(line) for line in raw_tb]

        except asyncio.TimeoutError as exc:
            await self._dispatch_timeout_sigint()
            raise KernelExecutionTimeoutError(
                f"Execution exceeded timeout of {timeout_seconds}s; SIGINT interrupt dispatched."
            ) from exc
        finally:
            self.correlator.complete_execution(msg_id)
            self._record_activity()

        duration = round(max(0.0, time.monotonic() - start_mono), 4)
        final_status: str = "error" if (has_error and reply_status == "ok") else reply_status
        if final_status not in ("ok", "error", "abort", "timed_out", "interrupted"):
            final_status = "error" if has_error else "ok"

        return RawExecutionResult(
            execution_id=execution_id,
            msg_id=msg_id,
            status=final_status,  # type: ignore[arg-type]
            execution_count=execution_count,
            duration_seconds=duration,
            iopub_messages=iopub_messages,
            reply_message=reply_message,
            has_error=has_error,
            error_name=error_name,
            error_value=error_value,
            traceback=traceback_lines,
        )

    async def execute(
        self,
        code: str,
        timeout_seconds: int = 300,
        store_history: bool = True,
        max_text_lines: int = 1000,
        max_output_bytes: int = 102400,
        max_images: int = 5,
        include_images: bool = True,
        stop_on_error: bool = True,
    ) -> List[MCPContentBlock]:
        """Execute ``code`` and return chronological MCP ``[TextContent | ImageContent]`` blocks."""
        result = await self.execute_raw(
            code=code,
            timeout_seconds=timeout_seconds,
            store_history=store_history,
            stop_on_error=stop_on_error,
        )
        blocks = convert_iopub_to_mcp_blocks(
            result.iopub_messages,
            include_images=include_images,
            max_text_lines=max_text_lines,
            max_output_bytes=max_output_bytes,
            max_images=max_images,
        )
        if result.has_error and not any(
            (m.get("header") or {}).get("msg_type") == "error" for m in result.iopub_messages
        ):
            tb_text = (
                sanitize_ansi_traceback(result.traceback)
                if result.traceback
                else f"{result.error_name or 'Error'}: {result.error_value or ''}"
            )
            blocks.append(TextContent(type="text", text=redact_credentials(tb_text)))
        return blocks

    async def execute_snippet(
        self,
        code: str,
        timeout_seconds: int = 300,
        store_history: bool = True,
        stop_on_error: bool = True,
    ) -> AsyncIterator[ExecutionEvent]:
        """
        Asynchronously stream real-time ``ExecutionEvent`` chunks across ``shell`` and ``iopub``
        (TECH_SPEC §2.4 Contract 3).
        """
        self._record_activity()
        if not self.is_connected:
            await self.connect()

        frame = build_execute_request(
            code=code,
            session_id=self.session_id,
            silent=False,
            store_history=store_history,
            allow_stdin=False,
            stop_on_error=stop_on_error,
        )
        msg_id = str(frame["header"]["msg_id"])
        queue = self.correlator.register_execution(msg_id)
        execution_count: Optional[int] = None
        got_execute_reply = False
        got_idle_status = False

        try:
            await self._send_frame(frame)
            yield ExecutionEvent(event_type="started", raw_message=frame)

            loop = asyncio.get_running_loop()
            deadline = loop.time() + float(timeout_seconds)

            while not (got_execute_reply and got_idle_status):
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError()

                msg = await asyncio.wait_for(queue.get(), timeout=remaining)
                if not isinstance(msg, dict):
                    continue

                if msg.get("_internal_type") == "connection_lost":
                    raise KernelConnectionError(
                        f"Kernel WebSocket connection lost during streaming: {msg.get('reason', 'unknown')}"
                    )

                header = msg.get("header") if isinstance(msg.get("header"), dict) else {}
                msg_type = header.get("msg_type") or msg.get("msg_type")
                content = msg.get("content") if isinstance(msg.get("content"), dict) else {}

                if msg_type in ("stream", "display_data", "execute_result", "execute_reply"):
                    self._record_activity()

                if msg_type == "status":
                    if content.get("execution_state") == KernelState.IDLE.value:
                        got_idle_status = True

                elif msg_type == "execute_reply":
                    got_execute_reply = True
                    if content.get("execution_count") is not None:
                        execution_count = int(content["execution_count"])

                elif msg_type == "stream":
                    sname = str(content.get("name", "stdout")).lower()
                    stream_lit = "stderr" if sname == "stderr" else "stdout"
                    cleaned = redact_credentials(sanitize_ansi_text(str(content.get("text", ""))))
                    yield ExecutionEvent(
                        event_type="stream",
                        stream_name=stream_lit,  # type: ignore[arg-type]
                        text=cleaned,
                        raw_message=msg,
                    )

                elif msg_type in ("display_data", "execute_result"):
                    data = content.get("data") if isinstance(content.get("data"), dict) else {}
                    if content.get("execution_count") is not None:
                        execution_count = int(content["execution_count"])
                    ev_type = "execute_result" if msg_type == "execute_result" else "display_data"

                    img_mime: Optional[str] = None
                    img_b64: Optional[str] = None
                    for mime in ("image/png", "image/jpeg"):
                        val = data.get(mime)
                        if isinstance(val, str) and val.strip():
                            img_mime = mime
                            img_b64 = val.strip()
                            break

                    if img_mime and img_b64:
                        yield ExecutionEvent(
                            event_type=ev_type,  # type: ignore[arg-type]
                            mime_type=img_mime,
                            data_base64=img_b64,
                            execution_count=execution_count,
                            raw_message=msg,
                        )
                    elif "text/plain" in data and data["text/plain"] is not None:
                        yield ExecutionEvent(
                            event_type=ev_type,  # type: ignore[arg-type]
                            mime_type="text/plain",
                            text=redact_credentials(sanitize_ansi_text(str(data["text/plain"]))),
                            execution_count=execution_count,
                            raw_message=msg,
                        )

                elif msg_type == "error":
                    raw_tb = content.get("traceback")
                    cleaned_tb = (
                        redact_credentials(sanitize_ansi_traceback([str(l) for l in raw_tb]))
                        if isinstance(raw_tb, list)
                        else None
                    )
                    yield ExecutionEvent(
                        event_type="error",
                        error_name=str(content.get("ename", "Error")),
                        error_value=redact_credentials(str(content.get("evalue", ""))),
                        sanitized_traceback=cleaned_tb,
                        execution_count=execution_count,
                        raw_message=msg,
                    )

            yield ExecutionEvent(
                event_type="completed",
                execution_count=execution_count,
            )

        except asyncio.TimeoutError:
            await self._dispatch_timeout_sigint()
            yield ExecutionEvent(
                event_type="error",
                error_name="TimeoutError",
                error_value=f"Execution exceeded timeout of {timeout_seconds}s; SIGINT dispatched.",
                execution_count=execution_count,
            )
        finally:
            self.correlator.complete_execution(msg_id)
            self._record_activity()

    async def _discover_kernel_if_needed(self, http_client: httpx.AsyncClient) -> str:
        if self.kernel_id:
            return self.kernel_id
        kernels_url = f"{self.proxy_url}/api/kernels"
        headers = self._http_headers()
        resp = await http_client.get(kernels_url, headers=headers)
        resp.raise_for_status()
        kernels = resp.json()
        if isinstance(kernels, list) and kernels and isinstance(kernels[0], dict) and kernels[0].get("id"):
            self.kernel_id = str(kernels[0]["id"])
            return self.kernel_id
        create_resp = await http_client.post(kernels_url, headers=headers, json={"name": "python3"})
        create_resp.raise_for_status()
        created = create_resp.json()
        if isinstance(created, dict) and created.get("id"):
            self.kernel_id = str(created["id"])
            return self.kernel_id
        raise KernelConnectionError("Unable to discover or create kernel for operation.")

    async def interrupt(
        self,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> InterruptExecutionResponse:
        """
        Tier 1 non-destructive SIGINT interrupt (TECH_SPEC §5.6).

        Preserves all in-memory kernel variables (``state_reset=False``, ``kernel_restarted=False``).
        """
        self._record_activity()
        if http_client is not None:
            return await self._interrupt_with_client(http_client)

        async with httpx.AsyncClient(timeout=15.0) as http:
            return await self._interrupt_with_client(http)

    async def _interrupt_with_client(
        self,
        http: httpx.AsyncClient,
    ) -> InterruptExecutionResponse:
        kernel_id = await self._discover_kernel_if_needed(http)
        interrupt_url = f"{self.proxy_url}/api/kernels/{kernel_id}/interrupt"

        if self.is_connected and self._ws is not None:
            try:
                await self._send_frame(build_interrupt_request(session_id=self.session_id))
            except Exception:
                pass

        resp = await http.post(interrupt_url, headers=self._http_headers())
        resp.raise_for_status()

        return InterruptExecutionResponse(
            runtime_id=self.runtime_id,
            status="interrupted",
            state_reset=False,
            kernel_restarted=False,
            kernel_state=self._kernel_state if self._kernel_state != "disconnected" else "idle",
            warning=None,
            message="SIGINT interrupt signal dispatched; execution halted while preserving in-memory state.",
        )

    async def restart(
        self,
        http_client: Optional[httpx.AsyncClient] = None,
        wait_for_ready: bool = True,
    ) -> InterruptExecutionResponse:
        """
        Tier 2 destructive kernel restart (TECH_SPEC §5.6).

        Resets all in-memory variables and re-establishes WebSocket if previously connected.
        """
        self._record_activity()
        if http_client is not None:
            return await self._restart_with_client(http_client, wait_for_ready=wait_for_ready)

        async with httpx.AsyncClient(timeout=15.0) as http:
            return await self._restart_with_client(http, wait_for_ready=wait_for_ready)

    async def _restart_with_client(
        self,
        http: httpx.AsyncClient,
        wait_for_ready: bool = True,
    ) -> InterruptExecutionResponse:
        kernel_id = await self._discover_kernel_if_needed(http)
        restart_url = f"{self.proxy_url}/api/kernels/{kernel_id}/restart"

        resp = await http.post(restart_url, headers=self._http_headers())
        resp.raise_for_status()

        try:
            payload = resp.json()
            if isinstance(payload, dict) and payload.get("id"):
                self.kernel_id = str(payload["id"])
        except Exception:
            pass

        had_ws = self._ws is not None
        self.correlator.flush_all(
            terminal_message={"_internal_type": "connection_lost", "reason": "kernel_restarted"}
        )

        if had_ws and wait_for_ready:
            await self._reconnect_with_client(http, max_retries=5, initial_backoff=0.2)
        else:
            self._kernel_state = KernelState.IDLE.value
            self._idle_event.set()
            self._sync_session_state(kernel_id=self.kernel_id, kernel_state=KernelState.IDLE.value)

        return InterruptExecutionResponse(
            runtime_id=self.runtime_id,
            status="restarted",
            state_reset=True,
            kernel_restarted=True,
            kernel_state=KernelState.IDLE.value,
            warning=(
                "SIGINT did not halt kernel within timeout. Kernel was forcefully restarted; "
                "all in-memory variables and imported modules have been reset."
            ),
            message="Kernel restarted and WebSocket bridge automatically reconnected.",
        )

    async def interrupt_with_escalation(
        self,
        timeout: float = 5.0,
    ) -> InterruptExecutionResponse:
        """
        Execute Tier 1 SIGINT interrupt, waiting up to ``timeout`` seconds for ``idle``.
        If the kernel remains unresponsive, escalate to Tier 2 destructive ``/restart``.
        Uses a single ``httpx.AsyncClient`` instance across all HTTP calls.
        """
        self._record_activity()
        async with httpx.AsyncClient(timeout=15.0) as http:
            tier1_resp = await self._interrupt_with_client(http)

            became_idle = False
            if self.is_connected:
                if self._kernel_state == KernelState.IDLE.value:
                    became_idle = True
                else:
                    try:
                        await asyncio.wait_for(self._idle_event.wait(), timeout=timeout)
                        became_idle = True
                    except asyncio.TimeoutError:
                        became_idle = False
            else:
                kernel_id = self.kernel_id
                kernels_url = f"{self.proxy_url}/api/kernels"
                loop = asyncio.get_running_loop()
                deadline = loop.time() + max(0.0, float(timeout))
                while loop.time() < deadline:
                    try:
                        probe = await http.get(kernels_url, headers=self._http_headers())
                        if probe.status_code == 200 and probe.content:
                            data = probe.json()
                            if isinstance(data, list) and data:
                                target = next(
                                    (
                                        k
                                        for k in data
                                        if isinstance(k, dict) and k.get("id") == kernel_id
                                    ),
                                    data[0] if isinstance(data[0], dict) else None,
                                )
                                if target and target.get("execution_state") == KernelState.IDLE.value:
                                    became_idle = True
                                    break
                    except Exception:
                        pass
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    await asyncio.sleep(min(0.05, remaining))

            if became_idle:
                self._kernel_state = KernelState.IDLE.value
                self._sync_session_state(kernel_state=KernelState.IDLE.value)
                tier1_resp.kernel_state = KernelState.IDLE.value
                return tier1_resp

            logger.warning(
                "Kernel %s did not return to idle within %.2fs after SIGINT; escalating to restart.",
                self.kernel_id,
                timeout,
            )
            return await self._restart_with_client(http, wait_for_ready=True)

    async def reconnect(
        self,
        max_retries: int = 5,
        initial_backoff: float = 0.5,
    ) -> bool:
        """
        6-stage WebSocket auto-reconnection state machine (TECH_SPEC §5.6).

        Opens a single ``httpx.AsyncClient`` session across all retry iterations.
        """
        async with self._reconnect_lock:
            async with httpx.AsyncClient(timeout=15.0) as http:
                return await self._reconnect_with_client(
                    http,
                    max_retries=max_retries,
                    initial_backoff=initial_backoff,
                )

    async def _reconnect_with_client(
        self,
        http: httpx.AsyncClient,
        max_retries: int = 5,
        initial_backoff: float = 0.5,
    ) -> bool:
        # Stage 1: Detection
        self._kernel_state = KernelState.RECONNECTING.value
        self._idle_event.clear()
        self._sync_session_state(kernel_state=KernelState.RECONNECTING.value)

        # Stage 2: Teardown stale WebSocket and reader task cleanly
        self._closing = True
        await self._cancel_reader_task()
        old_ws = self._ws
        self._ws = None
        if old_ws is not None:
            try:
                await old_ws.close()
            except Exception:
                pass

        # Stage 3: Poll GET /api/kernels with exponential backoff until idle
        kernels_url = f"{self.proxy_url}/api/kernels"
        ready = False
        last_error: Optional[Exception] = None

        for attempt in range(max(1, int(max_retries))):
            try:
                resp = await http.get(kernels_url, headers=self._http_headers())
                resp.raise_for_status()
                kernels = resp.json()

                if isinstance(kernels, list) and len(kernels) == 0:
                    create_resp = await http.post(
                        kernels_url,
                        headers=self._http_headers(),
                        json={"name": "python3"},
                    )
                    create_resp.raise_for_status()
                    created = create_resp.json()
                    if isinstance(created, dict) and created.get("id"):
                        self.kernel_id = str(created["id"])
                        ready = True
                        break
                elif isinstance(kernels, list) and len(kernels) > 0:
                    matching = next(
                        (
                            k
                            for k in kernels
                            if isinstance(k, dict) and k.get("id") == self.kernel_id
                        ),
                        kernels[0] if isinstance(kernels[0], dict) else None,
                    )
                    if matching is not None:
                        if matching.get("id"):
                            self.kernel_id = str(matching["id"])
                        state = matching.get("execution_state")
                        if state in (None, KernelState.IDLE.value):
                            ready = True
                            break
            except Exception as exc:
                last_error = exc

            if attempt < max_retries - 1:
                delay = float(initial_backoff) * (2 ** attempt)
                await asyncio.sleep(delay)

        if not ready or not self.kernel_id:
            self._kernel_state = KernelState.DISCONNECTED.value
            self._sync_session_state(kernel_state=KernelState.DISCONNECTED.value)
            raise KernelConnectionError(
                f"Kernel WebSocket reconnection failed after {max_retries} attempts: {last_error}"
            )

        # Stage 4: WebSocket Re-Handshake
        self._closing = False
        self._ws = await connect_colab_kernel_ws(
            proxy_url=self.proxy_url,
            proxy_token=self.proxy_token,
            kernel_id=self.kernel_id,
            client_agent=self.client_agent,
            session_id=self.session_id,
        )

        # Stage 5: State Synchronization (flush stale queues & start fresh reader)
        self.correlator.flush_all(
            terminal_message={"_internal_type": "connection_lost", "reason": "reconnected"}
        )
        self._reader_task = asyncio.create_task(self._reader_loop())

        # Stage 6: Notification & Resume
        self._kernel_state = KernelState.IDLE.value
        self._idle_event.set()
        self._sync_session_state(
            kernel_id=self.kernel_id,
            kernel_state=KernelState.IDLE.value,
        )
        logger.info(
            "Kernel WebSocket reconnected cleanly (runtime=%s, kernel_id=%s).",
            self.runtime_id or "default",
            self.kernel_id,
        )
        return True


__all__ = [
    "DEFAULT_WS_MAX_SIZE",
    "DEFAULT_WS_PING_INTERVAL",
    "DEFAULT_WS_PING_TIMEOUT",
    "KernelWebSocketClient",
    "connect_colab_kernel_ws",
    "connect_kernel_ws",
    "discover_or_create_kernel",
    "resolve_or_create_kernel",
]
