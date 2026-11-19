"""
Background keepalive heartbeat daemon for active Google Colab runtimes.

Sends periodic GET requests to /{endpoint}/keep-alive/ every 35 seconds with
X-Colab-Tunnel: Google and treats httpx.ReadTimeout as a healthy heartbeat
per TECH_SPEC §4.7.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional
from urllib.parse import quote

import httpx

from colab_mcp.auth.models import ColabAuthError
from colab_mcp.auth.oauth import get_valid_auth_headers
from colab_mcp.utils.logging import get_logger

logger = get_logger(__name__)


class KeepAliveDaemon:
    """Periodic background heartbeat ping daemon for a Colab TFE runtime."""

    BASE_URL = "https://colab.research.google.com/tun/m"

    def __init__(
        self,
        endpoint: str,
        auth_token: Optional[str] = None,
        auth_user_index: int = 0,
        interval_seconds: float = 35.0,
        client_agent: str = "colab-mcp",
        client: Optional[httpx.AsyncClient] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.endpoint = endpoint
        self.auth_token = auth_token
        self.auth_user_index = auth_user_index
        self.interval_seconds = interval_seconds
        self.client_agent = client_agent
        self._transport = transport
        self._owns_client = client is None
        self.client: httpx.AsyncClient = client or httpx.AsyncClient(
            transport=self._transport,
            timeout=30.0,
        )
        self.is_healthy: bool = True
        self.consecutive_failures: int = 0
        self.is_running: bool = False
        self._task: Optional[asyncio.Task[Any]] = None

    @property
    def healthy(self) -> bool:
        """Alias for ``is_healthy``."""
        return self.is_healthy

    async def _get_headers(self) -> Dict[str, str]:
        if self.auth_token is not None:
            headers: Dict[str, str] = {
                "Authorization": f"Bearer {self.auth_token}",
            }
        else:
            headers = dict(await get_valid_auth_headers(self.auth_user_index))

        headers["X-Colab-Tunnel"] = "Google"
        headers["X-Colab-Client-Agent"] = self.client_agent
        headers["X-Goog-AuthUser"] = str(self.auth_user_index)
        return headers

    async def ping_once(self) -> bool:
        """
        Dispatch a single keep-alive heartbeat ping to TFE.

        ``httpx.ReadTimeout`` is caught and treated as healthy because Colab's
        gateway holds keep-alive connections open until socket timeout.
        """
        escaped_endpoint = quote(self.endpoint, safe="-_.")
        url = f"{self.BASE_URL}/{escaped_endpoint}/keep-alive/?authuser={self.auth_user_index}"

        try:
            headers = await self._get_headers()
            resp = await self.client.get(url, headers=headers)
            if resp.status_code == 200:
                self.is_healthy = True
                self.consecutive_failures = 0
                return True

            logger.warning(
                "Keepalive ping for %s returned HTTP %d",
                self.endpoint,
                resp.status_code,
            )
            self.is_healthy = False
            self.consecutive_failures += 1
            return False
        except httpx.ReadTimeout:
            logger.debug(
                "Keepalive ReadTimeout for %s (healthy per TECH_SPEC §4.7)",
                self.endpoint,
            )
            self.is_healthy = True
            self.consecutive_failures = 0
            return True
        except (ColabAuthError, httpx.HTTPError, Exception) as exc:
            logger.warning("Keepalive ping error for %s: %s", self.endpoint, exc)
            self.is_healthy = False
            self.consecutive_failures += 1
            return False

    _send_ping = ping_once

    async def _run_loop(self) -> None:
        try:
            while self.is_running:
                try:
                    await self.ping_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Unexpected exception in keepalive loop for %s: %s",
                        self.endpoint,
                        exc,
                    )
                if not self.is_running:
                    break
                await asyncio.sleep(self.interval_seconds)
        except asyncio.CancelledError:
            pass

    async def start(self) -> None:
        """Start the background keepalive task idempotently."""
        if self.is_running and self._task is not None and not self._task.done():
            return
        if self._owns_client and (self.client is None or self.client.is_closed):
            self.client = httpx.AsyncClient(
                transport=self._transport,
                timeout=30.0,
            )
        self.is_running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the background keepalive task cleanly and close owned HTTP client."""
        self.is_running = False
        if self._task is not None:
            if not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        if self._owns_client and self.client is not None and not self.client.is_closed:
            await self.client.aclose()


KeepaliveDaemon = KeepAliveDaemon

__all__ = ["KeepAliveDaemon", "KeepaliveDaemon"]
