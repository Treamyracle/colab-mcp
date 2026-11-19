"""
Google Colab Tunnel Frontend (TFE) REST Client.

Implements TFE runtime enumeration, CCU balance inspection, two-step assignment
handshake, session re-attachment, quota/capacity error disambiguation, and
two-step runtime teardown.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, NoReturn, Optional
from urllib.parse import quote, urlencode

import httpx

from colab_mcp.auth.oauth import get_valid_auth_headers
from colab_mcp.runtime.models import (
    Accelerator,
    Assignment,
    CCUInfo,
    CloudCapacityError,
    ColabRuntimeError,
    ComputeUnitsExhaustedError,
    FreeTierQuotaExceededError,
    HighRamNotAvailableError,
    RuntimeDisconnectedError,
    Shape,
    TooManyAssignmentsError,
    Variant,
    coerce_accelerator,
    coerce_shape,
    coerce_variant,
    resolve_variant,
)
from colab_mcp.utils.logging import get_logger
from colab_mcp.utils.nbh import generate_nbh

logger = get_logger(__name__)


def _strip_xssi_prefix(text: str) -> str:
    """Defensively strip optional Google XSSI framing prefix if present."""
    for prefix in (")]}\',\r\n", ")]}\',\n", ")]}\',", ")]}'\r\n", ")]}'\n"):
        if text.startswith(prefix):
            return text[len(prefix):]
    if text.startswith(")]}'"):
        return text[4:].lstrip("\r\n")
    return text


class ColabTFEClient:
    """Async REST client for Google Colab Tunnel Frontend (TFE) gateway."""

    BASE_URL = "https://colab.research.google.com/tun/m"

    def __init__(
        self,
        auth_token: Optional[str] = None,
        auth_user_index: int = 0,
        client_agent: str = "colab-mcp",
        transport: Optional[httpx.AsyncBaseTransport] = None,
        client: Optional[httpx.AsyncClient] = None,
        timeout: float = 30.0,
    ) -> None:
        self.auth_token = auth_token
        self.auth_user_index = auth_user_index
        self.client_agent = client_agent
        self._owns_client = client is None
        self.client: httpx.AsyncClient = client or httpx.AsyncClient(
            transport=transport,
            timeout=timeout,
        )

    async def __aenter__(self) -> "ColabTFEClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client if owned by this instance."""
        if self._owns_client and self.client is not None and not self.client.is_closed:
            await self.client.aclose()

    async def _get_headers(self) -> Dict[str, str]:
        """Build authenticated TFE headers via explicit token or stored credentials."""
        if self.auth_token is not None:
            return {
                "Authorization": f"Bearer {self.auth_token}",
                "Accept": "application/json",
                "X-Colab-Client-Agent": self.client_agent,
                "X-Goog-AuthUser": str(self.auth_user_index),
            }

        auth_headers = dict(await get_valid_auth_headers(self.auth_user_index))
        auth_headers["Accept"] = "application/json"
        auth_headers["X-Colab-Client-Agent"] = self.client_agent
        auth_headers["X-Goog-AuthUser"] = str(self.auth_user_index)
        return auth_headers

    def _parse_json(self, resp: httpx.Response) -> Any:
        """Parse JSON response body after stripping optional XSSI prefix."""
        cleaned = _strip_xssi_prefix(resp.text)
        return json.loads(cleaned)

    def _build_assign_url(
        self,
        nbh: str,
        variant: Variant,
        accelerator: Accelerator,
        shape: Shape,
    ) -> str:
        params: Dict[str, str] = {
            "authuser": str(self.auth_user_index),
            "nbh": nbh,
            "variant": variant.value,
            "accelerator": accelerator.value,
        }
        if shape == Shape.HIGH_RAM:
            params["shape"] = Shape.HIGH_RAM.value
        return f"{self.BASE_URL}/assign?{urlencode(params)}"

    def _parse_412_error(self, response_text: str) -> NoReturn:
        """Parse HTTP 412 Precondition Failed body into distinct quota exceptions."""
        codes: List[str] = []
        messages: List[str] = []

        cleaned = _strip_xssi_prefix(response_text or "")
        try:
            err_data = json.loads(cleaned)
        except Exception:
            err_data = None

        if isinstance(err_data, dict):
            err_field = err_data.get("error")
            if isinstance(err_field, dict):
                if err_field.get("code"):
                    codes.append(str(err_field["code"]))
                if err_field.get("message"):
                    messages.append(str(err_field["message"]))
            elif isinstance(err_field, str):
                codes.append(err_field)
                messages.append(err_field)

            if err_data.get("code"):
                codes.append(str(err_data["code"]))
            if err_data.get("message"):
                messages.append(str(err_data["message"]))

            details = err_data.get("details")
            if isinstance(details, list):
                for item in details:
                    if isinstance(item, dict):
                        if item.get("code"):
                            codes.append(str(item["code"]))
                        if item.get("message"):
                            messages.append(str(item["message"]))
                        if item.get("info"):
                            messages.append(str(item["info"]))
        elif response_text:
            messages.append(response_text)

        codes_upper = " ".join(codes).upper()
        msgs_lower = " ".join(messages).lower()
        primary_msg = "; ".join(m for m in messages if m) or response_text

        if "TOO_MANY_ASSIGNMENTS" in codes_upper or "too_many_assignments" in msgs_lower:
            raise TooManyAssignmentsError(
                "Maximum concurrent Colab runtimes reached for this account. "
                "Disconnect active sessions or upgrade your Colab tier."
            )
        if (
            "CCU_EXHAUSTED" in codes_upper
            or "ccu_exhausted" in msgs_lower
            or "compute units" in msgs_lower
        ):
            raise ComputeUnitsExhaustedError(
                "Colab Compute Unit (CCU) balance is exhausted. "
                "Please purchase additional compute units at colab.research.google.com/signup."
            )
        if (
            "FREE_TIER" in codes_upper
            or "free_tier" in msgs_lower
            or "quota" in msgs_lower
        ):
            raise FreeTierQuotaExceededError(
                "Colab Free Tier GPU usage quota reached. "
                "Google Colab limits free GPU access dynamically. Switch to CPU or retry later."
            )

        raise RuntimeError(f"Colab allocation rejected (HTTP 412): {primary_msg}")

    def _handle_assign_error_response(
        self,
        resp: httpx.Response,
        shape: Shape,
        accelerator: Accelerator,
    ) -> None:
        """Handle 400 High-RAM, 412 quota, 502/503/504 capacity, and generic HTTP errors."""
        if resp.status_code == 400 and shape == Shape.HIGH_RAM:
            raise HighRamNotAvailableError(
                "High-RAM shape is not available on Google Colab Free Tier. "
                "Please upgrade to Colab Pro/Pro+ or request standard RAM (high_ram=False)."
            )
        if resp.status_code == 412:
            self._parse_412_error(resp.text)
        if resp.status_code in (502, 503, 504):
            raise CloudCapacityError(
                f"Accelerator {accelerator.value} is currently out of capacity in Google Cloud "
                f"data centers (HTTP {resp.status_code})."
            )
        resp.raise_for_status()

    async def list_assignments(self) -> List[Assignment]:
        """Query active VM container assignments for the authenticated user index."""
        url = f"{self.BASE_URL}/assignments?authuser={self.auth_user_index}"
        headers = await self._get_headers()
        resp = await self.client.get(url, headers=headers)
        resp.raise_for_status()
        data = self._parse_json(resp)

        if isinstance(data, list):
            raw_items = data
        elif isinstance(data, dict):
            raw_items = data.get("assignments", [])
            if not isinstance(raw_items, list):
                raise ValueError("Invalid 'assignments' payload received from TFE.")
        else:
            raise ValueError("Unexpected response format from /tun/m/assignments.")

        return [Assignment.model_validate(item) for item in raw_items if isinstance(item, dict)]

    async def get_ccu_info(self) -> CCUInfo:
        """Query Compute Unit (CCU) balance, tier, and consumption rate."""
        url = f"{self.BASE_URL}/ccu-info?authuser={self.auth_user_index}"
        headers = await self._get_headers()
        resp = await self.client.get(url, headers=headers)
        resp.raise_for_status()
        data = self._parse_json(resp)
        return CCUInfo.model_validate(data)

    async def assign_runtime(
        self,
        accelerator: Accelerator | str = Accelerator.T4,
        high_ram: Optional[bool] = None,
        nbh: Optional[str] = None,
        reuse_existing: bool = True,
        variant: Optional[Variant | str] = None,
        shape: Optional[Shape | str] = None,
    ) -> Assignment:
        """
        Provision a new Colab runtime VM or re-attach to an existing matching VM.

        Supports two-step TFE XSRF handshake (GET /assign -> POST /assign),
        pre-assignment re-attachment, High-RAM Free Tier protection, HTTP 412
        quota disambiguation, and HTTP 502/503/504 cloud capacity handling.
        """
        acc = coerce_accelerator(accelerator)
        resolved_variant = (
            resolve_variant(acc) if variant is None else coerce_variant(variant)
        )
        if high_ram is True:
            resolved_shape = Shape.HIGH_RAM
        elif shape is not None:
            resolved_shape = coerce_shape(shape)
        else:
            resolved_shape = Shape.STANDARD

        resolved_nbh = nbh if nbh is not None else generate_nbh()

        # Step 0: Pre-assignment enumeration and re-attachment
        if reuse_existing:
            active_sessions = await self.list_assignments()
            for session in active_sessions:
                session_acc = (session.accelerator or "").upper()
                session_var = (session.variant or "").upper()
                matches = (session_acc == acc.value) or (
                    acc == Accelerator.NONE
                    and session_var == "DEFAULT"
                    and session_acc in ("", "NONE")
                )
                if (
                    matches
                    and session.endpoint
                    and session.runtime_proxy_url
                    and session.runtime_proxy_token
                ):
                    logger.info(
                        "Re-attaching to existing Colab %s runtime: %s",
                        acc.value,
                        session.endpoint,
                    )
                    return session

        url = self._build_assign_url(resolved_nbh, resolved_variant, acc, resolved_shape)
        headers = await self._get_headers()

        # Step 1: Pre-assignment check & XSRF token fetch
        resp = await self.client.get(url, headers=headers)
        self._handle_assign_error_response(resp, resolved_shape, acc)
        data = self._parse_json(resp)

        # Shortcut if Step 1 directly returned a full assignment
        if (
            isinstance(data, dict)
            and data.get("endpoint")
            and (
                data.get("runtimeProxyInfo")
                or (data.get("runtime_proxy_url") and data.get("runtime_proxy_token"))
            )
        ):
            return Assignment.model_validate(data)

        xsrf_token: Optional[str] = None
        if isinstance(data, dict):
            xsrf_token = data.get("token")
        if not xsrf_token:
            xsrf_token = resp.cookies.get("XSRF-TOKEN")
        if not xsrf_token:
            raise ValueError("Colab TFE failed to provide an assignment XSRF token.")

        # Step 2: Allocation claim with XSRF token
        post_headers = {**headers, "X-Goog-Colab-Token": str(xsrf_token)}
        alloc_resp = await self.client.post(url, headers=post_headers)
        self._handle_assign_error_response(alloc_resp, resolved_shape, acc)
        alloc_data = self._parse_json(alloc_resp)
        return Assignment.model_validate(alloc_data)

    async def unassign_runtime(self, endpoint: str) -> bool:
        """
        Execute two-step TFE teardown to release a Colab VM and halt CCU metering.

        Returns True idempotently when the backend reports HTTP 404 or already terminated.
        """
        if not endpoint or endpoint in (".", ".."):
            raise ValueError(f"Invalid endpoint identifier: {endpoint!r}")
        escaped_endpoint = quote(endpoint, safe="-_.")
        unassign_url = (
            f"{self.BASE_URL}/unassign/{escaped_endpoint}?authuser={self.auth_user_index}"
        )
        headers = await self._get_headers()

        # Step 1: Query unassign confirmation token
        get_resp = await self.client.get(unassign_url, headers=headers)
        if get_resp.status_code == 404:
            return True
        get_resp.raise_for_status()

        data = self._parse_json(get_resp)
        if isinstance(data, dict) and data.get("status") in ("terminated", "unassigned"):
            return True

        xsrf_token: Optional[str] = None
        if isinstance(data, dict):
            xsrf_token = data.get("token")
        if not xsrf_token:
            xsrf_token = get_resp.cookies.get("XSRF-TOKEN")
        if not xsrf_token:
            raise ValueError("Colab TFE failed to provide an unassign XSRF token.")

        # Step 2: Confirm deallocation
        post_headers = {**headers, "X-Goog-Colab-Token": str(xsrf_token)}
        post_resp = await self.client.post(unassign_url, headers=post_headers)
        if post_resp.status_code == 404:
            return True
        post_resp.raise_for_status()
        return 200 <= post_resp.status_code < 300


__all__ = [
    "Accelerator",
    "Assignment",
    "CCUInfo",
    "CloudCapacityError",
    "ColabRuntimeError",
    "ColabTFEClient",
    "ComputeUnitsExhaustedError",
    "FreeTierQuotaExceededError",
    "HighRamNotAvailableError",
    "RuntimeDisconnectedError",
    "Shape",
    "TooManyAssignmentsError",
    "Variant",
]
