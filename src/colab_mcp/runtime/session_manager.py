"""
Runtime Session Manager, Proxy Token Caching, and 30-Minute Inactivity Watchdog.

Tracks active Colab VM sessions, caches proxy JWT tokens (auto-refreshing on
expiry via GET /tun/m/assignments), discovers Jupyter kernels, and enforces the
1,800-second inactivity watchdog to prevent CCU runaway.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

from colab_mcp.runtime.client import ColabTFEClient
from colab_mcp.runtime.keepalive import KeepAliveDaemon
from colab_mcp.runtime.models import (
    Assignment,
    KernelConnectionParams,
    RuntimeDisconnectedError,
)
from colab_mcp.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_INACTIVITY_TIMEOUT_SECONDS = 1800.0
PROXY_TOKEN_REFRESH_BUFFER_SECONDS = 300.0


@dataclass
class RuntimeSessionRecord:
    """In-memory metadata for an active Colab runtime session."""

    endpoint: str
    auth_user_index: int = 0
    assignment: Optional[Assignment] = None
    accelerator: str = "NONE"
    variant: str = "DEFAULT"
    high_ram: bool = False
    kernel_id: Optional[str] = None
    kernel_state: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_activity_time: float = field(default_factory=time.time)
    token_acquired_at: float = field(default_factory=time.time)
    keepalive_daemon: Optional[Any] = None
    status: str = "CONNECTED"


class SessionManager:
    """
    Manages Colab runtime sessions, proxy token refresh, and inactivity watchdog.

    Supports both the single-session watchdog API and the multi-session
    registry API used across runtime tools, logout teardown, and M3 kernels.
    """

    def __init__(
        self,
        inactivity_timeout_seconds: float = DEFAULT_INACTIVITY_TIMEOUT_SECONDS,
        unassign_handler: Optional[Callable[[str], Awaitable[bool]]] = None,
    ) -> None:
        self.inactivity_timeout_seconds = inactivity_timeout_seconds
        self.unassign_handler = unassign_handler
        self._last_activity_time: float = time.time()
        self._active_endpoint: Optional[str] = None
        self._keepalive_daemon: Optional[Any] = None
        self._session_status: str = "DISCONNECTED"
        self._sessions: Dict[str, RuntimeSessionRecord] = {}
        self._watchdog_task: Optional[asyncio.Task[Any]] = None
        self._endpoint_locks: Dict[str, asyncio.Lock] = {}
        self._tracked_daemons: Dict[int, Any] = {}
        self._superseded_daemons: Dict[int, Any] = {}
        self._cleanup_tasks: set[asyncio.Task[Any]] = set()

    def _get_endpoint_lock(self, endpoint: str) -> asyncio.Lock:
        lock = self._endpoint_locks.get(endpoint)
        if lock is None:
            lock = asyncio.Lock()
            self._endpoint_locks[endpoint] = lock
        return lock

    def _track_daemon(self, daemon: Optional[Any]) -> None:
        if daemon is not None:
            self._tracked_daemons[id(daemon)] = daemon

    def _schedule_superseded_daemon_stop(self, daemon: Optional[Any]) -> None:
        if daemon is None:
            return
        self._track_daemon(daemon)
        self._superseded_daemons[id(daemon)] = daemon
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._stop_daemon_safely(daemon))
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    @property
    def last_activity_time(self) -> float:
        return self._last_activity_time

    @last_activity_time.setter
    def last_activity_time(self, value: float) -> None:
        self._last_activity_time = float(value)
        if self._active_endpoint and self._active_endpoint in self._sessions:
            self._sessions[self._active_endpoint].last_activity_time = self._last_activity_time

    @property
    def active_endpoint(self) -> Optional[str]:
        return self._active_endpoint

    @active_endpoint.setter
    def active_endpoint(self, endpoint: Optional[str]) -> None:
        self._active_endpoint = endpoint
        if endpoint is not None:
            if endpoint not in self._sessions:
                self._sessions[endpoint] = RuntimeSessionRecord(
                    endpoint=endpoint,
                    last_activity_time=self._last_activity_time,
                    keepalive_daemon=self._keepalive_daemon,
                )
            self._session_status = "CONNECTED"

    @property
    def keepalive_daemon(self) -> Optional[Any]:
        return self._keepalive_daemon

    @keepalive_daemon.setter
    def keepalive_daemon(self, daemon: Optional[Any]) -> None:
        self._keepalive_daemon = daemon
        self._track_daemon(daemon)
        if self._active_endpoint and self._active_endpoint in self._sessions:
            self._sessions[self._active_endpoint].keepalive_daemon = daemon

    @property
    def session_status(self) -> str:
        return self._session_status

    @session_status.setter
    def session_status(self, value: str) -> None:
        self._session_status = value

    @property
    def status(self) -> str:
        return self._session_status

    @status.setter
    def status(self, value: str) -> None:
        self._session_status = value

    @property
    def idle_seconds(self) -> float:
        """Elapsed seconds since the most recent recorded activity."""
        return max(0.0, time.time() - self._last_activity_time)

    def get_idle_seconds(self) -> float:
        """Return elapsed seconds since last recorded activity."""
        return self.idle_seconds

    def record_activity(
        self,
        timestamp: Optional[float] = None,
        endpoint: Optional[str] = None,
    ) -> None:
        """Update activity timestamp to reset the 30-minute inactivity watchdog."""
        ts = float(timestamp) if timestamp is not None else time.time()
        if endpoint is None or endpoint == self._active_endpoint:
            self._last_activity_time = ts
        target_endpoint = endpoint or self._active_endpoint
        if target_endpoint and target_endpoint in self._sessions:
            self._sessions[target_endpoint].last_activity_time = ts

    reset_activity = record_activity

    def set_active_session(
        self,
        endpoint: str,
        auth_user_index: int = 0,
        assignment: Optional[Assignment] = None,
        keepalive_daemon: Optional[Any] = None,
        high_ram: bool = False,
        kernel_id: Optional[str] = None,
        kernel_state: Optional[str] = None,
    ) -> RuntimeSessionRecord:
        """Attach or update the active runtime session."""
        now = time.time()
        existing = self._sessions.get(endpoint)
        if keepalive_daemon is not None:
            daemon = keepalive_daemon
        elif self._active_endpoint is None or self._active_endpoint == endpoint:
            daemon = self._keepalive_daemon
        else:
            daemon = None

        old_daemon = existing.keepalive_daemon if existing is not None else None
        if old_daemon is not None and old_daemon is not daemon:
            self._schedule_superseded_daemon_stop(old_daemon)
        if (
            self._active_endpoint == endpoint
            and self._keepalive_daemon is not None
            and self._keepalive_daemon is not daemon
            and self._keepalive_daemon is not old_daemon
        ):
            self._schedule_superseded_daemon_stop(self._keepalive_daemon)

        record = RuntimeSessionRecord(
            endpoint=endpoint,
            auth_user_index=auth_user_index,
            assignment=assignment,
            accelerator=(assignment.accelerator if assignment else "NONE"),
            variant=(assignment.variant or "DEFAULT") if assignment else "DEFAULT",
            high_ram=high_ram,
            kernel_id=kernel_id,
            kernel_state=kernel_state,
            created_at=existing.created_at if existing else now,
            last_activity_time=now,
            token_acquired_at=now,
            keepalive_daemon=daemon,
            status="CONNECTED",
        )
        self._sessions[endpoint] = record
        self._active_endpoint = endpoint
        self._last_activity_time = now
        self._session_status = "CONNECTED"
        self._keepalive_daemon = daemon
        self._track_daemon(daemon)
        return record

    def register_session(
        self,
        endpoint: str,
        auth_user_index: int = 0,
        last_activity: Optional[datetime | float | int] = None,
        assignment: Optional[Assignment] = None,
        keepalive_daemon: Optional[Any] = None,
        high_ram: bool = False,
        kernel_id: Optional[str] = None,
        kernel_state: Optional[str] = None,
    ) -> RuntimeSessionRecord:
        """Register a runtime session in the multi-session registry."""
        if isinstance(last_activity, datetime):
            dt = (
                last_activity
                if last_activity.tzinfo is not None
                else last_activity.replace(tzinfo=timezone.utc)
            )
            activity_ts = dt.timestamp()
        elif isinstance(last_activity, (int, float)):
            activity_ts = float(last_activity)
        else:
            activity_ts = time.time()

        now = time.time()
        existing = self._sessions.get(endpoint)
        old_daemon = existing.keepalive_daemon if existing is not None else None
        if old_daemon is not None and old_daemon is not keepalive_daemon:
            self._schedule_superseded_daemon_stop(old_daemon)
        if (
            self._active_endpoint == endpoint
            and self._keepalive_daemon is not None
            and self._keepalive_daemon is not keepalive_daemon
            and self._keepalive_daemon is not old_daemon
        ):
            self._schedule_superseded_daemon_stop(self._keepalive_daemon)

        record = RuntimeSessionRecord(
            endpoint=endpoint,
            auth_user_index=auth_user_index,
            assignment=assignment,
            accelerator=(assignment.accelerator if assignment else "NONE"),
            variant=(assignment.variant or "DEFAULT") if assignment else "DEFAULT",
            high_ram=high_ram,
            kernel_id=kernel_id,
            kernel_state=kernel_state,
            created_at=existing.created_at if existing else now,
            last_activity_time=activity_ts,
            token_acquired_at=now,
            keepalive_daemon=keepalive_daemon,
            status="CONNECTED",
        )
        self._sessions[endpoint] = record
        self._active_endpoint = endpoint
        self._last_activity_time = activity_ts
        self._session_status = "CONNECTED"
        self._keepalive_daemon = keepalive_daemon
        self._track_daemon(keepalive_daemon)
        return record

    def has_active_session(self, endpoint: str) -> bool:
        """Return True if ``endpoint`` is currently tracked as active."""
        return endpoint in self._sessions

    def get_session(self, endpoint: Optional[str] = None) -> Optional[RuntimeSessionRecord]:
        """Retrieve a session by endpoint or fall back to the active session."""
        if endpoint is not None:
            return self._sessions.get(endpoint)
        if self._active_endpoint and self._active_endpoint in self._sessions:
            return self._sessions[self._active_endpoint]
        if self._sessions:
            return next(iter(self._sessions.values()))
        return None

    def list_sessions(
        self, auth_user_index: Optional[int] = None
    ) -> List[RuntimeSessionRecord]:
        """List all tracked active sessions, optionally filtered by user index."""
        sessions = list(self._sessions.values())
        if auth_user_index is not None:
            sessions = [s for s in sessions if s.auth_user_index == auth_user_index]
        return sessions

    async def _stop_daemon_safely(self, daemon: Optional[Any]) -> None:
        if daemon is None:
            return
        try:
            if hasattr(daemon, "stop"):
                await daemon.stop()
        except Exception as exc:
            logger.warning("Error stopping keepalive daemon: %s", exc)
        finally:
            self._superseded_daemons.pop(id(daemon), None)
            still_referenced = (
                self._keepalive_daemon is daemon
                or any(rec.keepalive_daemon is daemon for rec in self._sessions.values())
            )
            if not still_referenced:
                self._tracked_daemons.pop(id(daemon), None)

    async def _unassign_endpoint_safely(
        self,
        endpoint: str,
        auth_user_index: int = 0,
        client: Optional[ColabTFEClient] = None,
    ) -> bool:
        try:
            if self.unassign_handler is not None:
                return bool(await self.unassign_handler(endpoint))
            if client is not None:
                return await client.unassign_runtime(endpoint)
            async with ColabTFEClient(auth_user_index=auth_user_index) as tfe_client:
                return await tfe_client.unassign_runtime(endpoint)
        except Exception as exc:
            logger.warning(
                "Watchdog/teardown failed to unassign runtime %s: %s",
                endpoint,
                exc,
            )
            return False

    async def check_watchdog(self) -> bool:
        """
        Evaluate inactivity against ``inactivity_timeout_seconds`` for the active session.

        If exceeded, stops the keepalive daemon, unassigns the VM, emits the
        stderr audit warning, and transitions status to ``IDLE_TIMEOUT_DISCONNECTED``.
        """
        endpoint = self._active_endpoint
        if endpoint is None:
            return False

        if self.idle_seconds < self.inactivity_timeout_seconds:
            return False

        rec = self._sessions.get(endpoint)
        if rec is not None and rec.kernel_state == "busy":
            rec.last_activity_time = time.time()
            self._last_activity_time = rec.last_activity_time
            return False

        async with self._get_endpoint_lock(endpoint):
            if self._active_endpoint != endpoint or self.idle_seconds < self.inactivity_timeout_seconds:
                return False

            record = self._sessions.get(endpoint)
            daemon = (record.keepalive_daemon if record else None) or self._keepalive_daemon
            auth_user_index = record.auth_user_index if record else 0

            await self._stop_daemon_safely(daemon)
            await self._unassign_endpoint_safely(endpoint, auth_user_index=auth_user_index)

            logger.warning(
                "[colab-mcp watchdog] Runtime %s was automatically deallocated after 30 minutes of inactivity to protect compute unit balance.",
                endpoint,
            )

            current = self._sessions.get(endpoint)
            if current is record:
                self._sessions.pop(endpoint, None)
                if self._active_endpoint == endpoint:
                    self._active_endpoint = None
                    if self._keepalive_daemon is daemon or (
                        record is not None and self._keepalive_daemon is record.keepalive_daemon
                    ):
                        self._keepalive_daemon = None
                    self._session_status = "IDLE_TIMEOUT_DISCONNECTED"
            return True

    _evaluate_inactivity = check_watchdog

    async def check_inactivity_watchdog(
        self, idle_threshold_seconds: float = DEFAULT_INACTIVITY_TIMEOUT_SECONDS
    ) -> int:
        """
        Evaluate all registered sessions against ``idle_threshold_seconds``.

        Returns the number of expired sessions deallocated.
        """
        now = time.time()
        expired = []
        for ep, rec in list(self._sessions.items()):
            if rec.kernel_state == "busy":
                rec.last_activity_time = now
                continue
            if (now - rec.last_activity_time) >= idle_threshold_seconds:
                expired.append((ep, rec))

        unassigned_count = 0
        for ep, rec in expired:
            async with self._get_endpoint_lock(ep):
                current = self._sessions.get(ep)
                if current is not rec:
                    continue
                if (time.time() - current.last_activity_time) < idle_threshold_seconds:
                    continue

                daemon = rec.keepalive_daemon or (
                    self._keepalive_daemon if self._active_endpoint == ep else None
                )
                await self._stop_daemon_safely(daemon)
                await self._unassign_endpoint_safely(ep, auth_user_index=rec.auth_user_index)

                # Re-check record identity after await: sync register_session() may have
                # replaced self._sessions[ep] while awaiting stop/unassign.
                if self._sessions.get(ep) is not rec:
                    continue

                logger.warning(
                    "[colab-mcp watchdog] Runtime %s was automatically deallocated after 30 minutes of inactivity to protect compute unit balance.",
                    ep,
                )

                self._sessions.pop(ep, None)
                if self._active_endpoint == ep:
                    self._active_endpoint = None
                    if (
                        self._keepalive_daemon is daemon
                        or self._keepalive_daemon is rec.keepalive_daemon
                    ):
                        self._keepalive_daemon = None
                    self._session_status = "IDLE_TIMEOUT_DISCONNECTED"
                unassigned_count += 1

        return unassigned_count

    async def start_watchdog_loop(self, check_interval_seconds: float = 60.0) -> None:
        """Start periodic background watchdog evaluation idempotently."""
        if self._watchdog_task is not None and not self._watchdog_task.done():
            return

        async def _loop() -> None:
            try:
                while True:
                    await asyncio.sleep(check_interval_seconds)
                    try:
                        await self.check_inactivity_watchdog(
                            idle_threshold_seconds=self.inactivity_timeout_seconds
                        )
                    except Exception as exc:
                        logger.warning("Inactivity watchdog tick failed: %s", exc)
            except asyncio.CancelledError:
                pass

        self._watchdog_task = asyncio.create_task(_loop())

    async def _discover_or_create_kernel(
        self,
        proxy_url: str,
        proxy_token: str,
        client_agent: str = "colab-mcp",
    ) -> str:
        """Query GET /api/kernels or create via POST /api/kernels on the runtime proxy."""
        base = proxy_url.rstrip("/")
        headers = {
            "X-Colab-Runtime-Proxy-Token": proxy_token,
            "X-Colab-Client-Agent": client_agent,
            "Accept": "application/json",
        }
        async with httpx.AsyncClient(timeout=15.0) as http:
            resp = await http.get(f"{base}/api/kernels", headers=headers)
            resp.raise_for_status()
            kernels = resp.json()
            if isinstance(kernels, list) and kernels:
                first = kernels[0]
                if isinstance(first, dict) and first.get("id"):
                    return str(first["id"])

            create_resp = await http.post(
                f"{base}/api/kernels",
                headers=headers,
                json={"name": "python3"},
            )
            create_resp.raise_for_status()
            created = create_resp.json()
            if isinstance(created, dict) and created.get("id"):
                return str(created["id"])

        raise RuntimeDisconnectedError("Failed to discover or initialize Jupyter kernel.")

    async def get_kernel_connection_params(
        self, runtime_id: Optional[str] = None
    ) -> KernelConnectionParams:
        """
        Retrieve active VM tunnel parameters and kernel binding (TECH_SPEC §2.4 Contract 2).

        Refreshes short-lived proxy tokens via GET /tun/m/assignments when near expiry.
        Raises RuntimeDisconnectedError if no matching VM session is active.
        """
        record = self.get_session(runtime_id)
        if record is None or record.assignment is None:
            raise RuntimeDisconnectedError(
                f"Runtime '{runtime_id or 'active'}' is disconnected or not found."
            )

        ep = record.endpoint
        async with self._get_endpoint_lock(ep):
            current = self._sessions.get(ep)
            if current is None or current.assignment is None:
                raise RuntimeDisconnectedError(
                    f"Runtime '{runtime_id or ep}' is disconnected or not found."
                )
            record = current
            assignment = record.assignment
            ttl = float(
                assignment.token_expires_in
                if assignment.token_expires_in is not None
                else 3600
            )
            elapsed = time.time() - record.token_acquired_at
            if (ttl - elapsed) <= PROXY_TOKEN_REFRESH_BUFFER_SECONDS:
                async with ColabTFEClient(auth_user_index=record.auth_user_index) as tfe_client:
                    active = await tfe_client.list_assignments()
                refreshed = next(
                    (a for a in active if a.endpoint == record.endpoint),
                    None,
                )
                if (
                    refreshed is None
                    or not refreshed.runtime_proxy_url
                    or not refreshed.runtime_proxy_token
                ):
                    daemon = record.keepalive_daemon or (
                        self._keepalive_daemon if self._active_endpoint == record.endpoint else None
                    )
                    await self._stop_daemon_safely(daemon)
                    if self._sessions.get(record.endpoint) is record:
                        self._sessions.pop(record.endpoint, None)
                        if self._active_endpoint == record.endpoint:
                            self._active_endpoint = None
                            if (
                                self._keepalive_daemon is daemon
                                or self._keepalive_daemon is record.keepalive_daemon
                            ):
                                self._keepalive_daemon = None
                            self._session_status = "DISCONNECTED"
                    raise RuntimeDisconnectedError(
                        f"Runtime '{record.endpoint}' has been terminated remotely."
                    )
                record.assignment = refreshed
                record.token_acquired_at = time.time()
                assignment = refreshed
                ttl = float(
                    assignment.token_expires_in
                    if assignment.token_expires_in is not None
                    else 3600
                )

            if not assignment.runtime_proxy_url or not assignment.runtime_proxy_token:
                raise RuntimeDisconnectedError(
                    f"Runtime '{record.endpoint}' is missing proxy connection credentials."
                )

            if not record.kernel_id:
                try:
                    record.kernel_id = await self._discover_or_create_kernel(
                        proxy_url=assignment.runtime_proxy_url,
                        proxy_token=assignment.runtime_proxy_token,
                    )
                except RuntimeDisconnectedError:
                    raise
                except Exception as exc:
                    raise RuntimeDisconnectedError(
                        f"Unable to reach kernel API for runtime '{record.endpoint}': {exc}"
                    ) from exc

            expires_at = datetime.fromtimestamp(
                record.token_acquired_at + ttl,
                tz=timezone.utc,
            )
            return KernelConnectionParams(
                endpoint=record.endpoint,
                proxy_url=assignment.runtime_proxy_url,
                proxy_token=assignment.runtime_proxy_token,
                token_expires_at=expires_at,
                kernel_id=record.kernel_id,
                client_agent="colab-mcp",
            )

    async def terminate_session(
        self,
        endpoint: Optional[str] = None,
        client: Optional[ColabTFEClient] = None,
    ) -> tuple[Optional[str], bool]:
        """Stop keepalive and unassign a single session; returns (endpoint, released)."""
        record = self.get_session(endpoint)
        if record is None:
            return (endpoint, False)

        ep = record.endpoint
        async with self._get_endpoint_lock(ep):
            current = self._sessions.get(ep)
            if current is None:
                return (ep, False)

            daemon = current.keepalive_daemon or (
                self._keepalive_daemon if self._active_endpoint == ep else None
            )
            await self._stop_daemon_safely(daemon)

            if self.unassign_handler is not None:
                released = bool(await self.unassign_handler(ep))
            elif client is not None:
                released = await client.unassign_runtime(ep)
            else:
                async with ColabTFEClient(auth_user_index=current.auth_user_index) as tfe_client:
                    released = await tfe_client.unassign_runtime(ep)

            if self._sessions.get(ep) is current:
                self._sessions.pop(ep, None)
                if self._active_endpoint == ep:
                    self._active_endpoint = None
                    if (
                        self._keepalive_daemon is daemon
                        or self._keepalive_daemon is current.keepalive_daemon
                    ):
                        self._keepalive_daemon = None
                    self._session_status = "DISCONNECTED"
            return (ep, released)

    async def terminate_all_sessions(self, auth_user_index: int = 0) -> int:
        """
        Terminate all active sessions for ``auth_user_index`` (used by colab_logout).

        Returns 0 immediately without network calls if no sessions exist for that user.
        """
        matching = [
            (ep, rec)
            for ep, rec in list(self._sessions.items())
            if rec.auth_user_index == auth_user_index
        ]
        if not matching:
            return 0

        terminated = 0
        for ep, rec in matching:
            async with self._get_endpoint_lock(ep):
                current = self._sessions.get(ep)
                if current is not rec:
                    continue

                daemon = current.keepalive_daemon or (
                    self._keepalive_daemon if self._active_endpoint == ep else None
                )
                await self._stop_daemon_safely(daemon)
                await self._unassign_endpoint_safely(ep, auth_user_index=auth_user_index)

                if self._sessions.get(ep) is current:
                    self._sessions.pop(ep, None)
                    if self._active_endpoint == ep:
                        self._active_endpoint = None
                        if (
                            self._keepalive_daemon is daemon
                            or self._keepalive_daemon is current.keepalive_daemon
                        ):
                            self._keepalive_daemon = None
                        self._session_status = "DISCONNECTED"
                    terminated += 1

        return terminated

    async def shutdown(self) -> None:
        """Cancel background watchdog and keepalive tasks cleanly on server exit."""
        if self._watchdog_task is not None and not self._watchdog_task.done():
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
            self._watchdog_task = None

        for lock in list(self._endpoint_locks.values()):
            async with lock:
                pass

        if self._cleanup_tasks:
            pending = list(self._cleanup_tasks)
            for task in pending:
                if not task.done():
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            self._cleanup_tasks.clear()

        seen_ids: set[int] = set()
        daemons_to_stop: List[Any] = []
        for daemon in (
            [rec.keepalive_daemon for rec in list(self._sessions.values())]
            + [self._keepalive_daemon]
            + list(self._superseded_daemons.values())
            + list(self._tracked_daemons.values())
        ):
            if daemon is not None and id(daemon) not in seen_ids:
                seen_ids.add(id(daemon))
                daemons_to_stop.append(daemon)

        for daemon in daemons_to_stop:
            await self._stop_daemon_safely(daemon)

        self._superseded_daemons.clear()
        self._tracked_daemons.clear()


RuntimeSessionManager = SessionManager

_SESSION_MANAGER: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Return the process-wide lazy SessionManager singleton."""
    global _SESSION_MANAGER
    if _SESSION_MANAGER is None:
        _SESSION_MANAGER = SessionManager()
    return _SESSION_MANAGER


def reset_session_manager() -> None:
    """Reset the singleton (primarily for test isolation)."""
    global _SESSION_MANAGER
    _SESSION_MANAGER = None


__all__ = [
    "DEFAULT_INACTIVITY_TIMEOUT_SECONDS",
    "RuntimeSessionManager",
    "RuntimeSessionRecord",
    "SessionManager",
    "get_session_manager",
    "reset_session_manager",
]
