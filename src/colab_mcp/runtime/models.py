"""
Pydantic models, enums, hardware catalog, and domain exceptions for Colab Runtime Lifecycle.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ============================================================================
# Hardware Accelerator, Variant, and RAM Shape Enums
# ============================================================================


class Variant(str, Enum):
    """Colab TFE runtime backend variant."""

    DEFAULT = "DEFAULT"
    GPU = "GPU"
    TPU = "TPU"


class Accelerator(str, Enum):
    """Supported Google Colab hardware accelerators."""

    NONE = "NONE"
    T4 = "T4"
    L4 = "L4"
    A100 = "A100"
    H100 = "H100"
    V5E1 = "V5E1"
    V6E1 = "V6E1"


class Shape(str, Enum):
    """Colab VM container memory shapes."""

    STANDARD = "standard"
    HIGH_RAM = "hm"


_ACCELERATOR_TO_VARIANT: Dict[Accelerator, Variant] = {
    Accelerator.NONE: Variant.DEFAULT,
    Accelerator.T4: Variant.GPU,
    Accelerator.L4: Variant.GPU,
    Accelerator.A100: Variant.GPU,
    Accelerator.H100: Variant.GPU,
    Accelerator.V5E1: Variant.TPU,
    Accelerator.V6E1: Variant.TPU,
}


def coerce_accelerator(accelerator: Accelerator | str) -> Accelerator:
    """Coerce an Accelerator enum or case-insensitive string to Accelerator."""
    if isinstance(accelerator, Accelerator):
        return accelerator
    if isinstance(accelerator, str):
        normalized = accelerator.strip().upper()
        try:
            return Accelerator(normalized)
        except ValueError as exc:
            valid = [a.value for a in Accelerator]
            raise ValueError(
                f"Unsupported accelerator '{accelerator}'. Valid options: {valid}"
            ) from exc
    raise ValueError(f"Invalid accelerator type: {type(accelerator).__name__}")


def coerce_variant(variant: Variant | str) -> Variant:
    """Coerce a Variant enum or case-insensitive string to Variant."""
    if isinstance(variant, Variant):
        return variant
    if isinstance(variant, str):
        normalized = variant.strip().upper()
        try:
            return Variant(normalized)
        except ValueError as exc:
            valid = [v.value for v in Variant]
            raise ValueError(f"Unsupported variant '{variant}'. Valid options: {valid}") from exc
    raise ValueError(f"Invalid variant type: {type(variant).__name__}")


def coerce_shape(shape: Shape | str) -> Shape:
    """Coerce a Shape enum or string ('standard', 'hm') to Shape."""
    if isinstance(shape, Shape):
        return shape
    if isinstance(shape, str):
        normalized = shape.strip().lower()
        try:
            return Shape(normalized)
        except ValueError as exc:
            valid = [s.value for s in Shape]
            raise ValueError(f"Unsupported shape '{shape}'. Valid options: {valid}") from exc
    raise ValueError(f"Invalid shape type: {type(shape).__name__}")


def resolve_variant(accelerator: Accelerator | str) -> Variant:
    """Map an accelerator to its canonical Colab TFE Variant."""
    acc = coerce_accelerator(accelerator)
    return _ACCELERATOR_TO_VARIANT[acc]


# ============================================================================
# Hardware Metadata Catalog (TECH_SPEC §4.3 & §7.3.2)
# ============================================================================


class AcceleratorSpec(BaseModel):
    """Hardware specification and CCU pricing metadata for a Colab runtime type."""

    accelerator: str
    variant: str
    display_name: str
    vram: Optional[str] = None
    system_ram: str
    high_ram_supported: bool = True
    free_tier_available: bool = False
    ccu_cost_per_hour: float
    available: bool = True


RuntimeTypeInfo = AcceleratorSpec


ACCELERATOR_CATALOG: Dict[Accelerator, AcceleratorSpec] = {
    Accelerator.NONE: AcceleratorSpec(
        accelerator="NONE",
        variant="DEFAULT",
        display_name="Standard CPU",
        vram=None,
        system_ram="12.7 GB (Standard) / 51.0 GB (High-RAM — Pro/Pro+ only)",
        high_ram_supported=True,
        free_tier_available=True,
        ccu_cost_per_hour=0.0,
        available=True,
    ),
    Accelerator.T4: AcceleratorSpec(
        accelerator="T4",
        variant="GPU",
        display_name="NVIDIA Tesla T4",
        vram="16 GB GDDR6",
        system_ram="12.7 GB (Standard) / 51.0 GB (High-RAM — Pro/Pro+ only)",
        high_ram_supported=True,
        free_tier_available=True,
        ccu_cost_per_hour=1.96,
        available=True,
    ),
    Accelerator.L4: AcceleratorSpec(
        accelerator="L4",
        variant="GPU",
        display_name="NVIDIA L4 Tensor Core",
        vram="24 GB GDDR6",
        system_ram="51.0 GB",
        high_ram_supported=True,
        free_tier_available=False,
        ccu_cost_per_hour=3.84,
        available=True,
    ),
    Accelerator.A100: AcceleratorSpec(
        accelerator="A100",
        variant="GPU",
        display_name="NVIDIA A100-SXM4",
        vram="40 GB / 80 GB HBM2",
        system_ram="83.5 GB",
        high_ram_supported=True,
        free_tier_available=False,
        ccu_cost_per_hour=5.42,
        available=True,
    ),
    Accelerator.H100: AcceleratorSpec(
        accelerator="H100",
        variant="GPU",
        display_name="NVIDIA H100-SXM5",
        vram="80 GB HBM3",
        system_ram="83.5 GB+",
        high_ram_supported=True,
        free_tier_available=False,
        ccu_cost_per_hour=13.50,
        available=True,
    ),
    Accelerator.V5E1: AcceleratorSpec(
        accelerator="V5E1",
        variant="TPU",
        display_name="Google Cloud TPU v5e",
        vram="16 GB HBM",
        system_ram="Single-chip pod slice",
        high_ram_supported=False,
        free_tier_available=False,
        ccu_cost_per_hour=1.96,
        available=True,
    ),
    Accelerator.V6E1: AcceleratorSpec(
        accelerator="V6E1",
        variant="TPU",
        display_name="Cloud TPU v6e",
        vram="32 GB HBM",
        system_ram="Single-chip pod slice",
        high_ram_supported=False,
        free_tier_available=False,
        ccu_cost_per_hour=4.50,
        available=True,
    ),
}


def get_accelerator_catalog(
    user_tier: Optional[str] = None,
    eligible_accelerators: Optional[set[str]] = None,
    ineligible_accelerators: Optional[set[str]] = None,
) -> List[RuntimeTypeInfo]:
    """Return the hardware matrix, adjusting availability by Colab tier or live eligibility."""
    tier = (user_tier or "").strip().upper()
    items: List[RuntimeTypeInfo] = []
    for acc, spec in ACCELERATOR_CATALOG.items():
        available = spec.available
        if eligible_accelerators is not None or ineligible_accelerators is not None:
            elig = {a.upper() for a in (eligible_accelerators or set())}
            inelig = {a.upper() for a in (ineligible_accelerators or set())}
            if acc == Accelerator.NONE:
                available = True
            elif acc.value in elig:
                available = True
            elif acc.value in inelig:
                available = False
            elif elig:
                available = False
        elif tier == "FREE":
            available = spec.free_tier_available
        elif tier == "PRO":
            available = acc not in (Accelerator.H100, Accelerator.V6E1)
        elif tier in ("PRO_PLUS", "PRO+"):
            available = True
        items.append(spec.model_copy(update={"available": available}))
    return items


# ============================================================================
# Domain Exceptions
# ============================================================================


class ColabRuntimeError(RuntimeError):
    """Base exception for Colab runtime lifecycle and TFE protocol errors."""


class HighRamNotAvailableError(ColabRuntimeError):
    """Raised when High-RAM (shape=hm) is rejected on Free Tier accounts."""


class TooManyAssignmentsError(ColabRuntimeError):
    """Raised when the user exceeds maximum allowed concurrent Colab sessions."""


class ComputeUnitsExhaustedError(ColabRuntimeError):
    """Raised when paid Colab Compute Units (CCU) balance is exhausted."""


class FreeTierQuotaExceededError(ColabRuntimeError):
    """Raised when dynamic Free Tier GPU allocation quota is exhausted."""


class CloudCapacityError(ColabRuntimeError):
    """Raised when Google Cloud GPU/TPU capacity is temporarily exhausted (HTTP 502/503/504)."""


class RuntimeDisconnectedError(ColabRuntimeError):
    """Raised when a target Colab runtime is disconnected or no longer exists."""


# ============================================================================
# TFE Wire Models & Kernel Connection Contract
# ============================================================================


class RuntimeProxyInfo(BaseModel):
    """Nested proxy info returned by Colab TFE /assign and /assignments."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    url: str
    token: str
    token_expires_in_seconds: int = Field(default=3600, alias="tokenExpiresInSeconds")

    @model_validator(mode="before")
    @classmethod
    def _normalize_expiry(cls, data: Any) -> Any:
        if isinstance(data, dict):
            normalized = dict(data)
            expires = next(
                (
                    normalized[key]
                    for key in (
                        "tokenExpiresInSeconds",
                        "token_expires_in_seconds",
                        "token_expires_in",
                    )
                    if key in normalized and normalized[key] is not None
                ),
                None,
            )
            for key in (
                "tokenExpiresInSeconds",
                "token_expires_in_seconds",
                "token_expires_in",
            ):
                if key in normalized and normalized[key] is None:
                    normalized[key] = 3600 if expires is None else expires
            if (
                "tokenExpiresInSeconds" not in normalized
                and "token_expires_in_seconds" not in normalized
                and "token_expires_in" in normalized
            ):
                normalized["token_expires_in_seconds"] = normalized["token_expires_in"]
            return normalized
        return data

    @property
    def token_expires_in(self) -> int:
        return self.token_expires_in_seconds

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style lookup compatibility for callers accessing runtimeProxyInfo fields."""
        mapping = {
            "url": self.url,
            "token": self.token,
            "tokenExpiresInSeconds": self.token_expires_in_seconds,
            "token_expires_in_seconds": self.token_expires_in_seconds,
            "token_expires_in": self.token_expires_in_seconds,
        }
        return mapping.get(key, default)

    def __getitem__(self, key: str) -> Any:
        sentinel = object()
        val = self.get(key, sentinel)
        if val is sentinel:
            raise KeyError(key)
        return val


_TFE_VARIANT_INT_MAP: Dict[int, str] = {
    0: "DEFAULT",
    1: "GPU",
    2: "TPU",
}

_TFE_SHAPE_INT_MAP: Dict[int, str] = {
    0: "STANDARD",
    1: "HIGH_RAM",
}


class Assignment(BaseModel):
    """
    Represents a Colab backend VM assignment.

    Supports both TFE camelCase nested ``runtimeProxyInfo`` payloads and
    flat snake_case ``runtime_proxy_url`` / ``runtime_proxy_token`` fields,
    as well as integer protobuf enum values returned by live Colab TFE.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    endpoint: Optional[str] = None
    accelerator: str = "NONE"
    variant: Optional[str] = None
    machine_shape: Optional[str] = Field(default=None, alias="machineShape")
    runtime_proxy_url: Optional[str] = None
    runtime_proxy_token: Optional[str] = None
    token_expires_in: int = 3600
    runtimeProxyInfo: Optional[RuntimeProxyInfo] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_proxy_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)

        raw_variant = normalized.get("variant")
        if isinstance(raw_variant, int) and not isinstance(raw_variant, bool):
            normalized["variant"] = _TFE_VARIANT_INT_MAP.get(raw_variant, str(raw_variant))
        elif isinstance(raw_variant, str) and raw_variant.strip().isdigit():
            normalized["variant"] = _TFE_VARIANT_INT_MAP.get(
                int(raw_variant.strip()), raw_variant
            )

        for shape_key in ("machineShape", "machine_shape"):
            raw_shape = normalized.get(shape_key)
            if isinstance(raw_shape, int) and not isinstance(raw_shape, bool):
                normalized[shape_key] = _TFE_SHAPE_INT_MAP.get(raw_shape, str(raw_shape))

        proxy_info = normalized.get("runtimeProxyInfo")
        if isinstance(proxy_info, RuntimeProxyInfo):
            url = proxy_info.url
            token = proxy_info.token
            expires = proxy_info.token_expires_in_seconds
            if not normalized.get("runtime_proxy_url") and url is not None:
                normalized["runtime_proxy_url"] = url
            if not normalized.get("runtime_proxy_token") and token is not None:
                normalized["runtime_proxy_token"] = token
            if (
                "token_expires_in" not in normalized
                or normalized.get("token_expires_in") is None
            ) and expires is not None:
                normalized["token_expires_in"] = expires
        elif isinstance(proxy_info, dict):
            url = proxy_info.get("url")
            token = proxy_info.get("token")
            expires = next(
                (
                    proxy_info[key]
                    for key in (
                        "tokenExpiresInSeconds",
                        "token_expires_in_seconds",
                        "token_expires_in",
                    )
                    if key in proxy_info and proxy_info[key] is not None
                ),
                3600,
            )
            if not normalized.get("runtime_proxy_url") and url is not None:
                normalized["runtime_proxy_url"] = url
            if not normalized.get("runtime_proxy_token") and token is not None:
                normalized["runtime_proxy_token"] = token
            if (
                "token_expires_in" not in normalized
                or normalized.get("token_expires_in") is None
            ) and expires is not None:
                normalized["token_expires_in"] = expires
        else:
            url = normalized.get("runtime_proxy_url")
            token = normalized.get("runtime_proxy_token")
            expires = normalized.get("token_expires_in")
            if expires is None:
                expires = 3600
                if "token_expires_in" in normalized:
                    normalized["token_expires_in"] = 3600
            if url is not None and token is not None and proxy_info is None:
                normalized["runtimeProxyInfo"] = {
                    "url": url,
                    "token": token,
                    "tokenExpiresInSeconds": expires,
                }

        if not normalized.get("variant") and normalized.get("accelerator"):
            try:
                normalized["variant"] = resolve_variant(normalized["accelerator"]).value
            except ValueError:
                pass

        return normalized


class CCUInfo(BaseModel):
    """Compute Unit (CCU) balance and burn-rate metrics from /tun/m/ccu-info."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    user_tier: str = "FREE"
    compute_units_remaining: float = 0.0
    consumption_rate_per_hour: float = 0.0
    active_assignments_count: int = 0
    auto_refill_enabled: bool = False
    eligible_gpus: List[str] = Field(default_factory=list)
    ineligible_gpus: List[str] = Field(default_factory=list)
    eligible_tpus: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_live_ccu_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if "compute_units_remaining" not in normalized and "currentBalance" in normalized:
            normalized["compute_units_remaining"] = normalized["currentBalance"]
        if "consumption_rate_per_hour" not in normalized and "consumptionRateHourly" in normalized:
            normalized["consumption_rate_per_hour"] = normalized["consumptionRateHourly"]
        if "active_assignments_count" not in normalized and "assignmentsCount" in normalized:
            normalized["active_assignments_count"] = normalized["assignmentsCount"]
        if "eligible_gpus" not in normalized and "eligibleGpus" in normalized:
            normalized["eligible_gpus"] = normalized["eligibleGpus"] or []
        if "ineligible_gpus" not in normalized and "ineligibleGpus" in normalized:
            normalized["ineligible_gpus"] = normalized["ineligibleGpus"] or []
        if "eligible_tpus" not in normalized and "eligibleTpus" in normalized:
            normalized["eligible_tpus"] = normalized["eligibleTpus"] or []

        if "user_tier" not in normalized or normalized.get("user_tier") is None:
            elig_gpus = {str(g).upper() for g in normalized.get("eligible_gpus", [])}
            elig_tpus = {str(t).upper() for t in normalized.get("eligible_tpus", [])}
            balance = float(normalized.get("compute_units_remaining") or 0.0)
            if "H100" in elig_gpus:
                normalized["user_tier"] = "PRO+"
            elif (
                elig_gpus & {"L4", "A100", "G4"}
                or bool(elig_tpus)
                or balance > 0.0
            ):
                normalized["user_tier"] = "PRO"

        return normalized


class KernelConnectionParams(BaseModel):
    """Contract 2 (`colab_mcp.runtime` -> `colab_mcp.kernel`) connection parameters."""

    endpoint: str
    proxy_url: str
    proxy_token: str
    token_expires_at: datetime
    kernel_id: str
    client_agent: str = "colab-mcp"


# ============================================================================
# MCP Tool Response Models (TECH_SPEC §7.3.2)
# ============================================================================


class ListRuntimeTypesResponse(BaseModel):
    """Response schema for colab_list_runtime_types."""

    account_tier: Optional[str] = None
    available_accelerators: List[RuntimeTypeInfo]


class ConnectRuntimeResponse(BaseModel):
    """Response schema for colab_connect_runtime."""

    runtime_id: str
    status: str = "connected"
    accelerator: str
    variant: str
    high_ram: bool = False
    reused_existing: bool = False
    resources: Optional[Dict[str, Any]] = None
    kernel_id: Optional[str] = None
    kernel_state: Optional[str] = None
    heartbeat_interval_seconds: int = 35


class RuntimeStatusResponse(BaseModel):
    """Response schema for colab_get_runtime_status."""

    runtime_id: Optional[str] = None
    status: str
    kernel_state: Optional[str] = None
    uptime_seconds: Optional[float] = None
    idle_seconds: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None
    keep_alive_healthy: Optional[bool] = None
    active_runtime: Optional[Dict[str, Any]] = None


class SessionSummary(BaseModel):
    """Summary of an active remote or locally tracked Colab runtime session."""

    runtime_id: str
    accelerator: str = "NONE"
    variant: Optional[str] = None
    tracked_locally: bool = False
    is_active_local: bool = False
    high_ram: Optional[bool] = None
    kernel_id: Optional[str] = None
    kernel_state: Optional[str] = None
    uptime_seconds: Optional[float] = None
    idle_seconds: Optional[float] = None
    keep_alive_healthy: Optional[bool] = None


class ListSessionsResponse(BaseModel):
    """Response schema for colab_list_sessions."""

    auth_user_index: int = 0
    total_sessions: int = 0
    active_runtime_id: Optional[str] = None
    sessions: List[SessionSummary] = Field(default_factory=list)
    remote_query_succeeded: bool = True
    note: Optional[str] = None


class DisconnectRuntimeResponse(BaseModel):
    """Response schema for colab_disconnect_runtime."""

    runtime_id: Optional[str] = None
    status: str
    compute_resources_released: bool = True
    already_terminated: bool = False
    terminated_runtime_ids: List[str] = Field(default_factory=list)
    terminated_count: int = 0
    message: str
