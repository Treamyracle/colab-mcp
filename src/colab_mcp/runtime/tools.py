"""
FastMCP Tool implementations for Google Colab Runtime Lifecycle Management.

Implements Tools 4-7 (TECH_SPEC §7.3.2):
- colab_list_runtime_types
- colab_connect_runtime
- colab_get_runtime_status
- colab_disconnect_runtime
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import httpx
from mcp.server.mcpserver import MCPServer

from colab_mcp.auth.models import AuthenticationRequiredError
from colab_mcp.runtime.client import ColabTFEClient
from colab_mcp.runtime.keepalive import KeepAliveDaemon
from colab_mcp.runtime.models import (
    ACCELERATOR_CATALOG,
    Accelerator,
    ColabRuntimeError,
    ConnectRuntimeResponse,
    DisconnectRuntimeResponse,
    GpuMetric,
    ListRuntimeTypesResponse,
    ListSessionsResponse,
    RuntimeStatusResponse,
    SessionSummary,
    SystemMetricsResponse,
    coerce_accelerator,
    get_accelerator_catalog,
    resolve_variant,
)
from colab_mcp.runtime.session_manager import get_session_manager
from colab_mcp.utils.logging import get_logger

logger = get_logger(__name__)


async def _probe_kernel_status(
    proxy_url: str,
    proxy_token: str,
    client_agent: str = "colab-mcp",
) -> tuple[Optional[str], Optional[str]]:
    """Best-effort query of GET /api/kernels on the runtime proxy."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            resp = await http.get(
                f"{proxy_url.rstrip('/')}/api/kernels",
                headers={
                    "X-Colab-Runtime-Proxy-Token": proxy_token,
                    "X-Colab-Client-Agent": client_agent,
                    "Accept": "application/json",
                },
            )
            if resp.status_code == 200:
                kernels = resp.json()
                if isinstance(kernels, list) and kernels:
                    first = kernels[0]
                    if isinstance(first, dict):
                        kid = first.get("id")
                        state = first.get("execution_state") or "idle"
                        return (str(kid) if kid else None, str(state))
    except Exception as exc:
        logger.debug("Kernel probe skipped/unavailable: %s", exc)
    return (None, None)


def _build_static_resources(accelerator: Accelerator, high_ram: bool) -> Dict[str, Any]:
    spec = ACCELERATOR_CATALOG[accelerator]
    resources: Dict[str, Any] = {
        "system_ram_gb": 51.0 if high_ram else 12.7,
    }
    if accelerator != Accelerator.NONE:
        resources["accelerator_name"] = spec.display_name
        if spec.vram:
            resources["vram"] = spec.vram
    return resources


async def list_runtime_types_impl(
    auth_user_index: int = 0,
) -> ListRuntimeTypesResponse:
    """Core logic for colab_list_runtime_types tool."""
    if isinstance(auth_user_index, bool) or not isinstance(auth_user_index, int):
        raise ValueError("auth_user_index must be an integer")
    if auth_user_index < 0:
        raise ValueError("auth_user_index must be >= 0")

    account_tier: Optional[str] = None
    eligible_accelerators: Optional[set[str]] = None
    ineligible_accelerators: Optional[set[str]] = None
    try:
        async with ColabTFEClient(auth_user_index=auth_user_index, timeout=5.0) as tfe_client:
            ccu = await tfe_client.get_ccu_info()
            account_tier = ccu.user_tier
            if ccu.eligible_gpus or ccu.eligible_tpus:
                eligible_accelerators = {
                    *(g.upper() for g in ccu.eligible_gpus),
                    *(t.upper() for t in ccu.eligible_tpus),
                }
            if ccu.ineligible_gpus:
                ineligible_accelerators = {g.upper() for g in ccu.ineligible_gpus}
    except Exception as exc:
        logger.debug(
            "Unable to fetch live CCU tier for user %d (returning static matrix): %s",
            auth_user_index,
            exc,
        )

    catalog = get_accelerator_catalog(
        account_tier,
        eligible_accelerators=eligible_accelerators,
        ineligible_accelerators=ineligible_accelerators,
    )
    return ListRuntimeTypesResponse(
        account_tier=account_tier,
        available_accelerators=catalog,
    )


async def connect_runtime_impl(
    accelerator: str = "NONE",
    high_ram: bool = False,
    reuse_existing: bool = True,
    auth_user_index: int = 0,
    timeout_seconds: int = 120,
) -> ConnectRuntimeResponse:
    """Core logic for colab_connect_runtime tool."""
    if isinstance(auth_user_index, bool) or not isinstance(auth_user_index, int):
        raise ValueError("auth_user_index must be an integer")
    if auth_user_index < 0:
        raise ValueError("auth_user_index must be >= 0")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
        raise ValueError("timeout_seconds must be an integer")
    if not (30 <= timeout_seconds <= 300):
        raise ValueError("timeout_seconds must be between 30 and 300")

    acc = coerce_accelerator(accelerator)
    variant = resolve_variant(acc)

    reused_existing = False
    async with ColabTFEClient(
        auth_user_index=auth_user_index,
        timeout=float(timeout_seconds),
    ) as tfe_client:
        assignment = None
        if reuse_existing:
            active_sessions = await tfe_client.list_assignments()
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
                    assignment = session
                    reused_existing = True
                    break

        if assignment is None:
            assignment = await tfe_client.assign_runtime(
                accelerator=acc,
                high_ram=high_ram,
                reuse_existing=False,
            )

    if not assignment.endpoint:
        raise ColabRuntimeError("Colab TFE assignment response did not include an endpoint.")

    mgr = get_session_manager()
    existing_rec = mgr.get_session(assignment.endpoint)
    if existing_rec is not None and existing_rec.keepalive_daemon is not None:
        await mgr._stop_daemon_safely(existing_rec.keepalive_daemon)

    kernel_id: Optional[str] = None
    kernel_state: Optional[str] = None
    if assignment.runtime_proxy_url and assignment.runtime_proxy_token:
        kernel_id, kernel_state = await _probe_kernel_status(
            assignment.runtime_proxy_url,
            assignment.runtime_proxy_token,
        )

    daemon = KeepAliveDaemon(
        endpoint=assignment.endpoint,
        auth_user_index=auth_user_index,
        interval_seconds=35.0,
    )
    mgr.register_session(
        endpoint=assignment.endpoint,
        auth_user_index=auth_user_index,
        assignment=assignment,
        keepalive_daemon=daemon,
        high_ram=high_ram,
        kernel_id=kernel_id,
        kernel_state=kernel_state,
    )
    await daemon.start()
    await mgr.start_watchdog_loop()

    return ConnectRuntimeResponse(
        runtime_id=assignment.endpoint,
        status="connected",
        accelerator=acc.value,
        variant=variant.value,
        high_ram=high_ram,
        reused_existing=reused_existing,
        resources=_build_static_resources(acc, high_ram),
        kernel_id=kernel_id,
        kernel_state=kernel_state,
        heartbeat_interval_seconds=35,
    )


async def get_runtime_status_impl(
    runtime_id: Optional[str] = None,
) -> RuntimeStatusResponse:
    """Core logic for colab_get_runtime_status tool."""
    if runtime_id is not None and not isinstance(runtime_id, str):
        raise ValueError("runtime_id must be a string or None")

    mgr = get_session_manager()
    record = mgr.get_session(runtime_id)
    if record is None or (runtime_id is not None and record.endpoint != runtime_id):
        return RuntimeStatusResponse(
            runtime_id=runtime_id,
            status="disconnected",
            kernel_state=None,
            uptime_seconds=None,
            idle_seconds=None,
            metrics=None,
            keep_alive_healthy=None,
            active_runtime=None,
        )

    now = time.time()
    uptime_seconds = max(0.0, now - record.created_at)
    idle_seconds = max(0.0, now - record.last_activity_time)

    kernel_state = record.kernel_state
    if (
        record.assignment is not None
        and record.assignment.runtime_proxy_url
        and record.assignment.runtime_proxy_token
    ):
        kid, live_state = await _probe_kernel_status(
            record.assignment.runtime_proxy_url,
            record.assignment.runtime_proxy_token,
        )
        if kid:
            record.kernel_id = kid
        if live_state:
            kernel_state = live_state
            record.kernel_state = live_state

    daemon = record.keepalive_daemon
    keep_alive_healthy = bool(daemon.is_healthy) if daemon is not None else None

    return RuntimeStatusResponse(
        runtime_id=record.endpoint,
        status="active",
        kernel_state=kernel_state or "unknown",
        uptime_seconds=round(uptime_seconds, 2),
        idle_seconds=round(idle_seconds, 2),
        metrics={
            "accelerator": record.accelerator,
            "variant": record.variant,
            "high_ram": record.high_ram,
        },
        keep_alive_healthy=keep_alive_healthy,
        active_runtime={
            "endpoint": record.endpoint,
            "accelerator": record.accelerator,
            "variant": record.variant,
            "auth_user_index": record.auth_user_index,
        },
    )


async def _build_session_summary(
    *,
    endpoint: str,
    accelerator: str,
    variant: Optional[str],
    rec: Optional[Any],
    proxy_url: Optional[str],
    proxy_token: Optional[str],
    active_endpoint: Optional[str],
    probe_kernels: bool,
    now: float,
) -> SessionSummary:
    """Construct a SessionSummary, probing live kernel status when requested."""
    kernel_id = rec.kernel_id if rec else None
    kernel_state = rec.kernel_state if rec else None
    if probe_kernels and proxy_url and proxy_token:
        probed_id, probed_state = await _probe_kernel_status(proxy_url, proxy_token)
        if probed_id:
            kernel_id = probed_id
            if rec is not None:
                rec.kernel_id = probed_id
        if probed_state:
            kernel_state = probed_state
            if rec is not None:
                rec.kernel_state = probed_state

    daemon = rec.keepalive_daemon if rec else None
    return SessionSummary(
        runtime_id=endpoint,
        accelerator=accelerator or "NONE",
        variant=variant,
        tracked_locally=rec is not None,
        is_active_local=(endpoint == active_endpoint),
        high_ram=rec.high_ram if rec else None,
        kernel_id=kernel_id,
        kernel_state=kernel_state,
        uptime_seconds=round(max(0.0, now - rec.created_at), 2) if rec else None,
        idle_seconds=round(max(0.0, now - rec.last_activity_time), 2) if rec else None,
        keep_alive_healthy=(bool(daemon.is_healthy) if daemon is not None else None),
    )


async def list_sessions_impl(
    auth_user_index: int = 0,
    probe_kernels: bool = True,
) -> ListSessionsResponse:
    """Core logic for colab_list_sessions tool."""
    if isinstance(auth_user_index, bool) or not isinstance(auth_user_index, int):
        raise ValueError("auth_user_index must be an integer")
    if auth_user_index < 0:
        raise ValueError("auth_user_index must be >= 0")
    if not isinstance(probe_kernels, bool):
        raise ValueError("probe_kernels must be a boolean")

    mgr = get_session_manager()
    local_records = {
        rec.endpoint: rec
        for rec in mgr.list_sessions()
        if rec.auth_user_index == auth_user_index
    }
    active_endpoint = mgr.active_endpoint

    remote_assignments = []
    remote_query_succeeded = True
    note: Optional[str] = None
    try:
        async with ColabTFEClient(auth_user_index=auth_user_index, timeout=10.0) as tfe_client:
            remote_assignments = await tfe_client.list_assignments()
    except AuthenticationRequiredError:
        if not local_records:
            raise
        remote_query_succeeded = False
        note = "Remote session lookup skipped: Google authentication required."

    now = time.time()
    summaries: list[SessionSummary] = []
    seen_endpoints: set[str] = set()

    for assignment in remote_assignments:
        ep = assignment.endpoint
        if not ep or ep in seen_endpoints:
            continue
        seen_endpoints.add(ep)
        rec = local_records.get(ep)
        proxy_url = (
            (rec.assignment.runtime_proxy_url if rec and rec.assignment else None)
            or assignment.runtime_proxy_url
        )
        proxy_token = (
            (rec.assignment.runtime_proxy_token if rec and rec.assignment else None)
            or assignment.runtime_proxy_token
        )
        summaries.append(
            await _build_session_summary(
                endpoint=ep,
                accelerator=(rec.accelerator if rec else assignment.accelerator) or "NONE",
                variant=(rec.variant if rec else assignment.variant),
                rec=rec,
                proxy_url=proxy_url,
                proxy_token=proxy_token,
                active_endpoint=active_endpoint,
                probe_kernels=probe_kernels,
                now=now,
            )
        )

    for ep, rec in local_records.items():
        if ep in seen_endpoints:
            continue
        seen_endpoints.add(ep)
        summaries.append(
            await _build_session_summary(
                endpoint=ep,
                accelerator=rec.accelerator,
                variant=rec.variant,
                rec=rec,
                proxy_url=rec.assignment.runtime_proxy_url if rec.assignment else None,
                proxy_token=rec.assignment.runtime_proxy_token if rec.assignment else None,
                active_endpoint=active_endpoint,
                probe_kernels=probe_kernels,
                now=now,
            )
        )

    return ListSessionsResponse(
        auth_user_index=auth_user_index,
        total_sessions=len(summaries),
        active_runtime_id=active_endpoint if active_endpoint in seen_endpoints else None,
        sessions=summaries,
        remote_query_succeeded=remote_query_succeeded,
        note=note,
    )


async def disconnect_runtime_impl(
    runtime_id: Optional[str] = None,
    disconnect_all: bool = False,
    auth_user_index: int = 0,
) -> DisconnectRuntimeResponse:
    """Core logic for colab_disconnect_runtime tool."""
    if runtime_id is not None and not isinstance(runtime_id, str):
        raise ValueError("runtime_id must be a string or None")
    if not isinstance(disconnect_all, bool):
        raise ValueError("disconnect_all must be a boolean")
    if isinstance(auth_user_index, bool) or not isinstance(auth_user_index, int):
        raise ValueError("auth_user_index must be an integer")
    if auth_user_index < 0:
        raise ValueError("auth_user_index must be >= 0")

    mgr = get_session_manager()

    if disconnect_all:
        terminated_ids: list[str] = []
        for rec in list(mgr.list_sessions()):
            if rec.auth_user_index != auth_user_index:
                continue
            endpoint, released = await mgr.terminate_session(rec.endpoint)
            if released and endpoint not in terminated_ids:
                terminated_ids.append(endpoint)

        try:
            async with ColabTFEClient(auth_user_index=auth_user_index, timeout=15.0) as tfe_client:
                remote_assignments = await tfe_client.list_assignments()
                for assignment in remote_assignments:
                    ep = assignment.endpoint
                    if ep and ep not in terminated_ids:
                        if await tfe_client.unassign_runtime(ep):
                            terminated_ids.append(ep)
        except AuthenticationRequiredError:
            pass

        if not terminated_ids:
            return DisconnectRuntimeResponse(
                runtime_id=None,
                status="disconnected",
                compute_resources_released=False,
                already_terminated=True,
                terminated_runtime_ids=[],
                terminated_count=0,
                message="No active Colab runtime sessions found.",
            )

        return DisconnectRuntimeResponse(
            runtime_id=terminated_ids[0] if len(terminated_ids) == 1 else None,
            status="terminated",
            compute_resources_released=True,
            already_terminated=False,
            terminated_runtime_ids=terminated_ids,
            terminated_count=len(terminated_ids),
            message=f"Terminated {len(terminated_ids)} Colab runtime session(s) and halted compute units.",
        )

    record = mgr.get_session(runtime_id)
    if record is not None and (runtime_id is None or record.endpoint == runtime_id):
        endpoint, released = await mgr.terminate_session(record.endpoint)
        return DisconnectRuntimeResponse(
            runtime_id=endpoint,
            status="terminated",
            compute_resources_released=released,
            already_terminated=False,
            terminated_runtime_ids=[endpoint] if released else [],
            terminated_count=1 if released else 0,
            message="Backend VM successfully deallocated and compute units halted.",
        )

    if runtime_id:
        try:
            async with ColabTFEClient(auth_user_index=auth_user_index, timeout=15.0) as tfe_client:
                remote_assignments = await tfe_client.list_assignments()
                remote_endpoints = {a.endpoint for a in remote_assignments if a.endpoint}
                if runtime_id in remote_endpoints:
                    released = await tfe_client.unassign_runtime(runtime_id)
                    return DisconnectRuntimeResponse(
                        runtime_id=runtime_id,
                        status="terminated",
                        compute_resources_released=released,
                        already_terminated=False,
                        terminated_runtime_ids=[runtime_id] if released else [],
                        terminated_count=1 if released else 0,
                        message=f"Remote Colab VM '{runtime_id}' deallocated and compute units halted.",
                    )
                return DisconnectRuntimeResponse(
                    runtime_id=runtime_id,
                    status="disconnected",
                    compute_resources_released=False,
                    already_terminated=True,
                    terminated_runtime_ids=[],
                    terminated_count=0,
                    message=f"Colab runtime '{runtime_id}' is not active or already terminated.",
                )
        except AuthenticationRequiredError:
            pass

    return DisconnectRuntimeResponse(
        runtime_id=runtime_id,
        status="disconnected",
        compute_resources_released=False,
        already_terminated=True,
        terminated_runtime_ids=[],
        terminated_count=0,
        message="No active Colab runtime session found.",
    )


async def colab_list_runtime_types(
    auth_user_index: int = 0,
) -> ListRuntimeTypesResponse:
    """Query available Google Colab hardware accelerators and CCU burn rates."""
    return await list_runtime_types_impl(auth_user_index=auth_user_index)


async def colab_connect_runtime(
    accelerator: str = "NONE",
    high_ram: bool = False,
    reuse_existing: bool = True,
    auth_user_index: int = 0,
    timeout_seconds: int = 120,
) -> ConnectRuntimeResponse:
    """Provision or re-attach to a Google Colab runtime VM container."""
    return await connect_runtime_impl(
        accelerator=accelerator,
        high_ram=high_ram,
        reuse_existing=reuse_existing,
        auth_user_index=auth_user_index,
        timeout_seconds=timeout_seconds,
    )


async def colab_get_runtime_status(
    runtime_id: Optional[str] = None,
) -> RuntimeStatusResponse:
    """Inspect active Colab VM health, idle timer, and kernel state."""
    return await get_runtime_status_impl(runtime_id=runtime_id)


async def colab_list_sessions(
    auth_user_index: int = 0,
    probe_kernels: bool = True,
) -> ListSessionsResponse:
    """List all active remote and locally tracked Google Colab runtime sessions."""
    return await list_sessions_impl(
        auth_user_index=auth_user_index,
        probe_kernels=probe_kernels,
    )


async def colab_disconnect_runtime(
    runtime_id: Optional[str] = None,
    disconnect_all: bool = False,
    auth_user_index: int = 0,
) -> DisconnectRuntimeResponse:
    """Release one or all allocated Colab container VMs via TFE /unassign."""
    return await disconnect_runtime_impl(
        runtime_id=runtime_id,
        disconnect_all=disconnect_all,
        auth_user_index=auth_user_index,
    )


async def get_system_metrics_impl(
    runtime_id: Optional[str] = None,
    *,
    _client_getter: Optional[Any] = None,
) -> SystemMetricsResponse:
    """Query GPU VRAM/utilization, system RAM, and disk usage on the active Colab VM."""
    import json as _json
    from colab_mcp.kernel.colab_shims import build_system_metrics_probe_code
    from colab_mcp.kernel.tools import get_active_kernel_client

    if runtime_id is not None and not isinstance(runtime_id, str):
        raise ValueError("runtime_id must be a string or None.")

    getter = _client_getter or get_active_kernel_client
    client = await getter(runtime_id)
    effective_runtime = getattr(client, "runtime_id", None) or getattr(client, "endpoint", None) or runtime_id
    get_session_manager().record_activity(endpoint=effective_runtime)

    probe_code = build_system_metrics_probe_code()
    if hasattr(client, "execute_raw"):
        raw = await client.execute_raw(
            code=probe_code,
            timeout_seconds=20,
            store_history=False,
            stop_on_error=True,
            silent=False,
            allow_stdin=False,
        )
        chunks: list[str] = []
        for msg in getattr(raw, "iopub_messages", []) or []:
            if not isinstance(msg, dict):
                continue
            hdr = msg.get("header") or {}
            if hdr.get("msg_type") == "stream":
                chunks.append(str((msg.get("content") or {}).get("text", "")))
        raw_text = "".join(chunks)
    else:
        blocks = await client.execute(
            code=probe_code,
            timeout_seconds=20,
            store_history=False,
        )
        raw_text = "\n".join(getattr(b, "text", "") for b in (blocks or []) if getattr(b, "type", None) == "text")

    parsed: Dict[str, Any] = {}
    for line in reversed(raw_text.strip().splitlines()):
        line_str = line.strip()
        if line_str.startswith("{") and line_str.endswith("}"):
            try:
                parsed = _json.loads(line_str)
                break
            except Exception:
                continue

    if not parsed:
        raise ColabRuntimeError(
            f"Unable to parse system metrics JSON output from runtime: {raw_text[:200]!r}"
        )

    gpus = [GpuMetric(**g) for g in (parsed.get("gpus") or []) if isinstance(g, dict)]
    return SystemMetricsResponse(
        runtime_id=effective_runtime,
        gpus=gpus,
        ram_total_mb=parsed.get("ram_total_mb"),
        ram_used_mb=parsed.get("ram_used_mb"),
        ram_free_mb=parsed.get("ram_free_mb"),
        ram_utilization_percent=parsed.get("ram_utilization_percent"),
        disk_path=str(parsed.get("disk_path") or "/content"),
        disk_total_gb=parsed.get("disk_total_gb"),
        disk_used_gb=parsed.get("disk_used_gb"),
        disk_free_gb=parsed.get("disk_free_gb"),
        disk_utilization_percent=parsed.get("disk_utilization_percent"),
    )


async def colab_get_system_metrics(
    runtime_id: Optional[str] = None,
) -> SystemMetricsResponse:
    """Measure GPU VRAM/utilization, system RAM, and disk space on an active Colab VM."""
    return await get_system_metrics_impl(runtime_id=runtime_id)


def register_runtime_tools(mcp: MCPServer) -> None:
    """Register all runtime lifecycle MCP tools on the MCPServer."""

    @mcp.tool(
        name="colab_list_runtime_types",
        description=(
            "Query available Google Colab hardware accelerators (CPU, T4, L4, A100, H100 GPUs, "
            "TPU v5e/v6e), high-RAM configurations, availability, and hourly CCU burn rates."
        ),
    )
    async def _tool_colab_list_runtime_types(
        auth_user_index: int = 0,
    ) -> ListRuntimeTypesResponse:
        return await colab_list_runtime_types(auth_user_index=auth_user_index)

    @mcp.tool(
        name="colab_list_sessions",
        description=(
            "List all active remote Google Colab VM sessions on the authenticated account "
            "(including browser or prior-run sessions) merged with locally tracked keep-alive state."
        ),
    )
    async def _tool_colab_list_sessions(
        auth_user_index: int = 0,
        probe_kernels: bool = True,
    ) -> ListSessionsResponse:
        return await colab_list_sessions(
            auth_user_index=auth_user_index,
            probe_kernels=probe_kernels,
        )

    @mcp.tool(
        name="colab_connect_runtime",
        description=(
            "Provision a new Colab container VM or re-attach to an existing active session "
            "with requested accelerator and RAM shape (Standard vs High-RAM)."
        ),
    )
    async def _tool_colab_connect_runtime(
        accelerator: str = "NONE",
        high_ram: bool = False,
        reuse_existing: bool = True,
        auth_user_index: int = 0,
        timeout_seconds: int = 120,
    ) -> ConnectRuntimeResponse:
        return await colab_connect_runtime(
            accelerator=accelerator,
            high_ram=high_ram,
            reuse_existing=reuse_existing,
            auth_user_index=auth_user_index,
            timeout_seconds=timeout_seconds,
        )

    @mcp.tool(
        name="colab_get_runtime_status",
        description=(
            "Inspect active VM health, kernel execution state, uptime, and remaining "
            "inactivity watchdog countdown."
        ),
    )
    async def _tool_colab_get_runtime_status(
        runtime_id: Optional[str] = None,
    ) -> RuntimeStatusResponse:
        return await colab_get_runtime_status(runtime_id=runtime_id)

    @mcp.tool(
        name="colab_disconnect_runtime",
        description=(
            "Release and terminate an allocated Colab container VM (local or remote by runtime_id, "
            "or all active sessions with disconnect_all=True) via TFE /unassign, halting compute unit burn."
        ),
    )
    async def _tool_colab_disconnect_runtime(
        runtime_id: Optional[str] = None,
        disconnect_all: bool = False,
        auth_user_index: int = 0,
    ) -> DisconnectRuntimeResponse:
        return await colab_disconnect_runtime(
            runtime_id=runtime_id,
            disconnect_all=disconnect_all,
            auth_user_index=auth_user_index,
        )

    @mcp.tool(
        name="colab_get_system_metrics",
        description=(
            "Inspect live GPU VRAM usage, GPU utilization %, system RAM, and /content disk usage "
            "on the connected Google Colab VM without perturbing execution history."
        ),
    )
    async def _tool_colab_get_system_metrics(
        runtime_id: Optional[str] = None,
    ) -> SystemMetricsResponse:
        return await colab_get_system_metrics(runtime_id=runtime_id)


__all__ = [
    "colab_connect_runtime",
    "colab_disconnect_runtime",
    "colab_get_runtime_status",
    "colab_get_system_metrics",
    "colab_list_runtime_types",
    "colab_list_sessions",
    "connect_runtime_impl",
    "disconnect_runtime_impl",
    "get_runtime_status_impl",
    "get_system_metrics_impl",
    "list_runtime_types_impl",
    "list_sessions_impl",
    "register_runtime_tools",
]
