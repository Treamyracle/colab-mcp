"""
Cross-platform SSH-over-WebSocket bridge for Google Colab runtimes (``/colab/ssh``).

Supports interactive SSH shells landing in ``/content``, OpenSSH ``ProxyCommand``
bridging for VS Code / Cursor Remote-SSH and ``scp``/``rsync``, one-shot remote
commands (``-- CMD``), and idempotent ephemeral ``--rm`` runtime teardown.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Sequence
from urllib.parse import urlencode, urlparse

import websockets

from colab_mcp.runtime.models import ColabRuntimeError
from colab_mcp.utils.logging import get_logger
from colab_mcp.utils.ws import select_ws_headers_param

logger = get_logger(__name__)

_SSH_PATH = "/colab/ssh"
_DEFAULT_REMOTE_DIR = "/content"
_SSH_HOST = "root@colab-runtime"
_PUBKEY_HEADER = "X-Colab-Ssh-Pubkey"
_CLIENT_AGENT_VALUE = "colab-mcp"
_PREFERRED_KEY_FILES = ("id_ed25519.pub", "id_ecdsa.pub")
_VALID_SSH_MODES = ("interactive", "proxy", "cmd")
_MAX_SSH_SESSIONS = 5


class ColabSSHError(ColabRuntimeError):
    """Raised when an SSH key or WebSocket handshake fails cleanly."""


def _pubkey_from_private_key(key_path: Path) -> str:
    """Derive an OpenSSH public key line using ``ssh-keygen -y -f``."""
    if not key_path.exists():
        raise FileNotFoundError(f"SSH key file not found: '{key_path}'.")
    try:
        proc = subprocess.run(
            ["ssh-keygen", "-y", "-f", str(key_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ColabSSHError(
            "OpenSSH `ssh-keygen` executable was not found on PATH; pass a `.pub` file via --identity."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise ColabSSHError(f"Failed to derive public key from '{key_path}': {detail}") from exc

    pubkey = proc.stdout.strip()
    if not pubkey:
        raise ColabSSHError(f"`ssh-keygen -y -f` returned an empty public key for '{key_path}'.")
    return pubkey


def resolve_ssh_pubkey(identity: Optional[str] = None) -> str:
    """
    Resolve an Ed25519 or ECDSA public key for ``X-Colab-Ssh-Pubkey``.

    Accepts either a `.pub` public key path or a private key path via *identity*,
    reading the matching `.pub` file directly when present before falling back to
    ``ssh-keygen -y -f``. When *identity* is omitted, scans ``~/.ssh`` for
    ``id_ed25519.pub`` then ``id_ecdsa.pub`` (intentionally skipping ``id_rsa``
    because Colab rejects RSA keys server-side).
    """
    if identity is not None and identity.strip():
        expanded = Path(identity).expanduser()
        if expanded.suffix == ".pub":
            pub_path = expanded
            priv_path = expanded.with_suffix("")
        else:
            priv_path = expanded
            pub_path = Path(f"{expanded}.pub")

        if pub_path.is_file():
            text = pub_path.read_text(encoding="utf-8").strip()
            if text:
                return text
        target = priv_path if priv_path.exists() else expanded
        return _pubkey_from_private_key(target)

    ssh_dir = Path.home() / ".ssh"
    for filename in _PREFERRED_KEY_FILES:
        candidate = ssh_dir / filename
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8").strip()
            if text:
                return text

    raise ColabSSHError(
        "No supported SSH public key found in ~/.ssh/ (checked id_ed25519.pub, id_ecdsa.pub). "
        "Colab rejects RSA keys; generate one with `ssh-keygen -t ed25519` or pass `--identity`."
    )


def resolve_ssh_private_key(identity: Optional[str] = None) -> Optional[str]:
    """Return sibling private key path for ``ssh -i`` if an explicit *.pub* path was passed."""
    if not identity or not identity.strip():
        return None
    expanded = Path(identity).expanduser()
    if expanded.suffix == ".pub":
        sibling = expanded.with_suffix("")
        return str(sibling) if sibling.exists() else None
    return str(expanded)


def build_ssh_ws_url(
    proxy_url: str,
    proxy_token: str,
    *,
    mode: str = "proxy",
) -> str:
    """Build the ``wss://.../colab/ssh`` URL and query parameters."""
    resolved_mode = mode if mode in _VALID_SSH_MODES else "proxy"
    parsed = urlparse(proxy_url.strip())
    scheme = "wss" if parsed.scheme == "https" else "ws"
    query = urlencode(
        [
            ("colab-runtime-proxy-token", proxy_token),
            ("colab-client-agent", _CLIENT_AGENT_VALUE),
            ("colab-client-agent", f"ssh-mode-{resolved_mode}"),
        ]
    )
    return f"{scheme}://{parsed.netloc}{_SSH_PATH}?{query}"


def _parse_session_limit(body: bytes) -> Optional[int]:
    try:
        data = json.loads(body.decode("utf-8", errors="replace"))
    except (ValueError, RecursionError):
        return None
    if not isinstance(data, dict) or data.get("error") != "already-active-session":
        return None
    limit = data.get("limit")
    if isinstance(limit, bool) or not isinstance(limit, int):
        return None
    return limit if 1 <= limit <= _MAX_SSH_SESSIONS else None


def explain_ssh_handshake_failure(status: Optional[int], body: bytes = b"") -> str:
    """Translate HTTP upgrade rejection statuses from ``/colab/ssh`` into actionable guidance."""
    snippet = body.decode("utf-8", errors="replace").strip()[:240]
    if status == 400:
        if "missing pubkey" in snippet:
            return "Server rejected SSH handshake: missing X-Colab-Ssh-Pubkey header."
        if "unsupported key type" in snippet:
            return (
                "Server rejected SSH public key: unsupported key type. Accepted types: "
                "ssh-ed25519 or ecdsa-sha2-nistp{256,384,521}. RSA keys are not accepted."
            )
        return f"Server rejected SSH handshake (HTTP 400): {snippet}."
    if status == 401:
        return (
            "SSH authentication failed (HTTP 401): runtime proxy token expired or invalid. "
            "Reconnect with `colab-mcp connect-runtime`."
        )
    if status == 403:
        return (
            "SSH forbidden (HTTP 403): runtime proxy token lacks permission for `/colab/ssh`."
        )
    if status == 404:
        return (
            "SSH endpoint unavailable (HTTP 404): this runtime does not expose `/colab/ssh`. "
            "Provision a fresh runtime (`colab-mcp connect-runtime --no-reuse-existing`)."
        )
    if status == 429:
        limit = _parse_session_limit(body)
        if limit is not None:
            return (
                f"SSH session limit reached (HTTP 429): {limit} of {limit} active SSH sessions "
                "in use on this runtime. Close another SSH/IDE session or pass `--force`."
            )
        return (
            "SSH session already active on this runtime (HTTP 429). Close the other session "
            "or pass `--force` to terminate stale `sshd` children."
        )
    if status == 502:
        return "Bad gateway (HTTP 502): remote `sshd` on the Colab VM is unreachable."
    if status is None:
        return f"SSH WebSocket upgrade failed: {snippet or 'connection closed'}."
    return f"SSH WebSocket upgrade rejected (HTTP {status}): {snippet}"


def _extract_ws_status_error(exc: Exception) -> tuple[Optional[int], bytes]:
    resp = getattr(exc, "response", None)
    if resp is not None:
        status = getattr(resp, "status_code", None)
        body = getattr(resp, "body", b"") or b""
        if isinstance(body, str):
            body = body.encode("utf-8", errors="replace")
        return status, body
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "resp_body", b"") or b""
    if isinstance(body, str):
        body = body.encode("utf-8", errors="replace")
    return status, body


async def connect_ssh_websocket(
    proxy_url: str,
    proxy_token: str,
    pubkey: str,
    *,
    mode: str = "proxy",
) -> Any:
    """Open the authenticated WebSocket stream to ``/colab/ssh``."""
    ws_url = build_ssh_ws_url(proxy_url, proxy_token, mode=mode)
    # Pass proxy token solely via the query string; sending X-Colab-Runtime-Proxy-Token
    # as an HTTP header causes Colab's proxy to route /colab/ssh to Jupyter (-> HTTP 404).
    headers = {
        _PUBKEY_HEADER: pubkey,
        "X-Colab-Client-Agent": _CLIENT_AGENT_VALUE,
    }
    headers_param = select_ws_headers_param(websockets.connect)
    ws_kwargs: Dict[str, Any] = {
        "subprotocols": None,
        "compression": None,
        "max_size": None,
        "ping_interval": 20,
        "ping_timeout": 20,
        headers_param: headers,
    }
    try:
        return await websockets.connect(ws_url, **ws_kwargs)
    except Exception as exc:
        status, body = _extract_ws_status_error(exc)
        if status is not None:
            raise ColabSSHError(explain_ssh_handshake_failure(status, body)) from exc
        raise ColabSSHError(f"Failed to open SSH WebSocket to runtime: {exc}") from exc


def _read_stdin_chunk(stdin_stream: BinaryIO) -> bytes:
    """Read raw available bytes cleanly on both live pipes and in-memory streams."""
    try:
        fd = stdin_stream.fileno()
        return os.read(fd, 8192)
    except Exception:
        read1 = getattr(stdin_stream, "read1", None)
        if callable(read1):
            return read1(8192)
        return stdin_stream.read(8192)


async def bridge_ws_stdio(
    ws: Any,
    *,
    stdin_stream: Optional[BinaryIO] = None,
    stdout_stream: Optional[BinaryIO] = None,
) -> int:
    """
    Bidirectionally bridge binary OpenSSH traffic between ``ws`` and local stdio.

    Returns ``0`` once either side closes.
    """
    in_buf = stdin_stream or sys.stdin.buffer
    out_buf = stdout_stream or sys.stdout.buffer
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    async def _send_and_signal(chunk: bytes) -> None:
        try:
            await ws.send(chunk)
        except Exception:
            stop_event.set()

    def _stdin_worker() -> None:
        try:
            while not stop_event.is_set():
                data = _read_stdin_chunk(in_buf)
                if not data:
                    break
                fut = asyncio.run_coroutine_threadsafe(_send_and_signal(data), loop)
                fut.result()
        except Exception:
            pass
        finally:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(stop_event.set)
                loop.call_soon_threadsafe(lambda: asyncio.create_task(_close_ws_quietly(ws)))

    reader_thread = threading.Thread(target=_stdin_worker, name="colab-ssh-stdin", daemon=True)
    reader_thread.start()

    try:
        while not stop_event.is_set():
            try:
                frame = await ws.recv()
            except websockets.exceptions.ConnectionClosed:
                break
            except StopAsyncIteration:
                break
            if frame is None:
                break
            if isinstance(frame, str):
                frame = frame.encode("utf-8")
            out_buf.write(frame)
            out_buf.flush()
    except (OSError, websockets.exceptions.WebSocketException):
        pass
    finally:
        stop_event.set()
        await _close_ws_quietly(ws)
    return 0


async def _close_ws_quietly(ws: Any) -> None:
    try:
        await ws.close()
    except Exception:
        pass


def build_proxy_command(
    *,
    runtime_id: Optional[str] = None,
    identity: Optional[str] = None,
    auth_user_index: Optional[int] = None,
    mode: str = "proxy",
    accelerator: Optional[str] = None,
    high_ram: bool = False,
    rm: bool = False,
) -> str:
    """
    Build the OS-quoted OpenSSH ``ProxyCommand`` invoking this CLI in ``--proxy-mode``.

    Uses ``subprocess.list2cmdline`` on native Windows and ``shlex.join`` on POSIX so
    paths containing spaces execute cleanly under non-login shells.
    """
    argv: List[str] = [sys.executable, "-m", "colab_mcp.cli", "ssh", "--proxy-mode"]
    if runtime_id:
        argv.extend(["--runtime-id", runtime_id])
    if identity:
        argv.extend(["--identity", identity])
    if auth_user_index is not None:
        argv.extend(["--auth-user-index", str(auth_user_index)])
    if accelerator:
        argv.extend(["--accelerator", accelerator])
    if high_ram:
        argv.append("--high-ram")
    if rm:
        argv.append("--rm")
    argv.extend(["--ssh-mode", mode])

    if sys.platform == "win32":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def build_ssh_subprocess_args(
    proxy_command: str,
    *,
    identity: Optional[str] = None,
    remote_command: Optional[Sequence[str]] = None,
    force_tty: bool = True,
) -> List[str]:
    """Assemble ``ssh`` CLI arguments for interactive shells or one-shot commands."""
    ssh_bin = shutil.which("ssh")
    if not ssh_bin:
        raise ColabSSHError(
            "OpenSSH client (`ssh`) was not found on PATH. Install OpenSSH or use `--proxy-mode`."
        )

    args: List[str] = [
        ssh_bin,
        "-o",
        f"ProxyCommand={proxy_command}",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        f"UserKnownHostsFile={os.devnull}",
        "-o",
        "LogLevel=ERROR",
    ]
    priv_key = resolve_ssh_private_key(identity)
    if priv_key:
        args.extend(["-i", priv_key])

    args.append("-t" if force_tty else "-T")
    args.append(_SSH_HOST)

    if remote_command is None:
        args.append(f"cd {_DEFAULT_REMOTE_DIR} 2>/dev/null; exec ${{SHELL:-/bin/bash}} -l")
    else:
        cmd_list = list(remote_command)
        line = cmd_list[0] if len(cmd_list) == 1 else shlex.join(cmd_list)
        args.append(f"cd {_DEFAULT_REMOTE_DIR} 2>/dev/null; {line}")

    return args


def render_ssh_config_snippet(
    *,
    host_alias: str = "colab",
    runtime_id: Optional[str] = None,
    identity: Optional[str] = None,
    accelerator: str = "T4",
    rm: bool = False,
) -> str:
    """Return a ready-to-paste ``~/.ssh/config`` entry for Cursor / VS Code Remote-SSH."""
    proxy = build_proxy_command(
        runtime_id=runtime_id,
        identity=identity,
        mode="proxy",
        accelerator=None if runtime_id else accelerator,
        rm=rm,
    )
    lines = [
        f"Host {host_alias}",
        f"    HostName {_SSH_HOST.split('@', 1)[1]}",
        "    User root",
        f"    ProxyCommand {proxy}",
        "    StrictHostKeyChecking no",
        f"    UserKnownHostsFile {os.devnull}",
        "    RequestTTY yes",
        "    RemoteCommand cd /content 2>/dev/null; exec bash -l",
    ]
    priv = resolve_ssh_private_key(identity)
    if priv:
        lines.append(f"    IdentityFile {priv}")
    return "\n".join(lines)


def install_rm_signal_handlers(cleanup_fn: Callable[[], None]) -> Callable[[], None]:
    """
    Install signal handlers so OpenSSH teardown signals (``SIGHUP``, ``SIGTERM``, ``SIGINT``)
    trigger *cleanup_fn* before exit. Returns a restore callback.
    """
    previous: Dict[int, Any] = {}
    sig_names = ["SIGINT", "SIGTERM"]
    if hasattr(signal, "SIGHUP"):
        sig_names.append("SIGHUP")

    def _handle(signum: int, _frame: Any) -> None:
        try:
            cleanup_fn()
        finally:
            if signum == signal.SIGINT:
                raise SystemExit(130)
            raise SystemExit(128 + signum)

    for name in sig_names:
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, _handle)
        except Exception:
            pass

    def _restore() -> None:
        for sig, prev in previous.items():
            try:
                signal.signal(sig, prev)
            except Exception:
                pass

    return _restore


__all__ = [
    "ColabSSHError",
    "bridge_ws_stdio",
    "build_proxy_command",
    "build_ssh_subprocess_args",
    "build_ssh_ws_url",
    "connect_ssh_websocket",
    "explain_ssh_handshake_failure",
    "install_rm_signal_handlers",
    "render_ssh_config_snippet",
    "resolve_ssh_private_key",
    "resolve_ssh_pubkey",
]
