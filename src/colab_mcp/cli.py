"""CLI entry point for colab-mcp."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Optional

import typer

from colab_mcp.utils.logging import StderrProxy, unprotect_stdout
from colab_mcp.utils.redaction import redact_credentials, redact_data

app = typer.Typer(
    name="colab-mcp",
    help="Google Colab Model Context Protocol (MCP) Server CLI",
    no_args_is_help=True,
)

_VALID_TRANSPORTS = ("stdio", "sse")
_VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _ensure_cli_stdout() -> None:
    """Restore interactive CLI stdout if a lazy import activated StderrProxy."""
    if isinstance(sys.stdout, StderrProxy):
        unprotect_stdout()


def _echo_json(payload: object) -> None:
    _ensure_cli_stdout()
    safe_payload = redact_data(payload)
    typer.echo(json.dumps(safe_payload, indent=2))


def _exit_with_error(exc: Exception, code: int = 1) -> None:
    _ensure_cli_stdout()
    typer.echo(f"Error: {redact_credentials(str(exc))}", err=True)
    raise typer.Exit(code=code)


@app.command("version")
def version() -> None:
    """Print the colab-mcp version."""
    import colab_mcp

    _ensure_cli_stdout()
    typer.echo(f"colab-mcp version: {colab_mcp.__version__}")


@app.command("run")
def run(
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="MCP transport protocol ('stdio' or 'sse').",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Bind host address for SSE transport.",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Bind port (1-65535) for SSE transport.",
    ),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    ),
) -> None:
    """Start the Google Colab MCP server over stdio or SSE transport."""
    normalized_transport = transport.strip().lower()
    if normalized_transport not in _VALID_TRANSPORTS:
        raise typer.BadParameter(
            f"Unsupported transport '{transport}'. Choose from: {', '.join(_VALID_TRANSPORTS)}.",
            param_hint="--transport",
        )
    if not (1 <= port <= 65535):
        raise typer.BadParameter(
            f"Port must be between 1 and 65535, got {port}.",
            param_hint="--port",
        )

    effective_log_level = "INFO"
    if log_level is not None:
        normalized_level = log_level.strip().upper()
        if normalized_level not in _VALID_LOG_LEVELS:
            raise typer.BadParameter(
                f"Unsupported log level '{log_level}'. Choose from: {', '.join(_VALID_LOG_LEVELS)}.",
                param_hint="--log-level",
            )
        effective_log_level = normalized_level

    from colab_mcp.server import run_server

    try:
        run_server(
            transport=normalized_transport,
            host=host,
            port=port,
            log_level=effective_log_level,
        )
    except Exception as exc:
        typer.echo(f"Error starting server: {redact_credentials(str(exc))}", err=True)
        raise typer.Exit(code=1) from None


@app.command("login")
def login(
    method: str = typer.Option(
        "oauth_remote",
        "--method",
        "-m",
        help="Authentication method: oauth_remote, oauth_loopback, oauth_adc, cdp_browser, cookie_import.",
    ),
    auth_user_index: int = typer.Option(
        0,
        "--auth-user-index",
        "-u",
        help="Google multi-account user index (>= 0).",
    ),
    auth_code: Optional[str] = typer.Option(
        None,
        "--auth-code",
        help="Authorization code for oauth_remote flow.",
    ),
    timeout_seconds: int = typer.Option(
        180,
        "--timeout-seconds",
        "--timeout",
        help="Timeout in seconds (30-600) for interactive login flows.",
    ),
) -> None:
    """Authenticate with Google Colab and persist credentials securely."""
    if auth_user_index < 0:
        raise typer.BadParameter("auth_user_index must be >= 0.", param_hint="--auth-user-index")
    if not (30 <= timeout_seconds <= 600):
        raise typer.BadParameter(
            "timeout_seconds must be between 30 and 600.",
            param_hint="--timeout-seconds",
        )

    from colab_mcp.auth.tools import colab_login_impl

    try:
        resp = asyncio.run(
            colab_login_impl(
                method=method,
                auth_user_index=auth_user_index,
                auth_code=auth_code,
                timeout_seconds=timeout_seconds,
            )
        )
        _ensure_cli_stdout()
        if (
            resp.status == "authorization_required"
            and method == "oauth_remote"
            and not auth_code
            and resp.auth_url
        ):
            typer.echo(f"Authorization URL:\n{redact_credentials(resp.auth_url)}")
            entered_code = typer.prompt("Enter authorization code").strip()
            resp = asyncio.run(
                colab_login_impl(
                    method=method,
                    auth_user_index=auth_user_index,
                    auth_code=entered_code,
                    timeout_seconds=timeout_seconds,
                )
            )
        _echo_json(resp.model_dump(mode="json"))
    except typer.Exit:
        raise
    except Exception as exc:
        _exit_with_error(exc)


@app.command("status")
def status(
    auth_user_index: int = typer.Option(
        0,
        "--auth-user-index",
        "-u",
        help="Google multi-account user index (>= 0).",
    ),
    force_refresh: bool = typer.Option(
        False,
        "--force-refresh",
        help="Perform an active token refresh and Colab TFE CCU check.",
    ),
) -> None:
    """Display authentication status, account tier, CCU balance, and runtime state."""
    if auth_user_index < 0:
        raise typer.BadParameter("auth_user_index must be >= 0.", param_hint="--auth-user-index")

    from colab_mcp.auth.tools import colab_get_auth_status_impl
    from colab_mcp.runtime.tools import get_runtime_status_impl

    try:
        auth_resp = asyncio.run(
            colab_get_auth_status_impl(
                auth_user_index=auth_user_index,
                force_refresh=force_refresh,
            )
        )
        payload = auth_resp.model_dump(mode="json")
        try:
            rt_resp = asyncio.run(get_runtime_status_impl())
            if rt_resp.status == "active":
                payload["active_runtime"] = rt_resp.model_dump(mode="json")
        except Exception:
            pass
        _echo_json(payload)
    except typer.Exit:
        raise
    except Exception as exc:
        _exit_with_error(exc)


@app.command("list-runtimes")
def list_runtimes(
    auth_user_index: int = typer.Option(
        0,
        "--auth-user-index",
        "-u",
        help="Google multi-account user index (>= 0).",
    ),
) -> None:
    """List available Colab hardware accelerators, RAM shapes, and hourly CCU rates."""
    if auth_user_index < 0:
        raise typer.BadParameter("auth_user_index must be >= 0.", param_hint="--auth-user-index")

    from colab_mcp.runtime.tools import list_runtime_types_impl

    try:
        resp = asyncio.run(list_runtime_types_impl(auth_user_index=auth_user_index))
        _ensure_cli_stdout()
        tier_label = resp.account_tier or "UNKNOWN"
        typer.echo(redact_credentials(f"Account Tier: {tier_label}"))
        for item in resp.available_accelerators:
            vram = item.vram or "None"
            avail = "available" if item.available else "unavailable"
            line = (
                f"- {item.accelerator} ({item.variant}): {item.display_name} | "
                f"VRAM: {vram} | RAM: {item.system_ram} | "
                f"{item.ccu_cost_per_hour:.2f} CCU/hr | {avail}"
            )
            typer.echo(redact_credentials(line))
    except typer.Exit:
        raise
    except Exception as exc:
        _exit_with_error(exc)


@app.command("list-sessions")
def list_sessions(
    auth_user_index: int = typer.Option(
        0,
        "--auth-user-index",
        "-u",
        help="Google multi-account user index (>= 0).",
    ),
    probe_kernels: bool = typer.Option(
        True,
        "--probe-kernels/--no-probe-kernels",
        help="Probe active runtime proxies for live Jupyter kernel state.",
    ),
) -> None:
    """List all active remote and locally tracked Google Colab sessions."""
    if auth_user_index < 0:
        raise typer.BadParameter("auth_user_index must be >= 0.", param_hint="--auth-user-index")

    from colab_mcp.runtime.tools import list_sessions_impl

    try:
        resp = asyncio.run(
            list_sessions_impl(
                auth_user_index=auth_user_index,
                probe_kernels=probe_kernels,
            )
        )
        _echo_json(resp.model_dump(mode="json"))
    except typer.Exit:
        raise
    except Exception as exc:
        _exit_with_error(exc)


@app.command("disconnect")
def disconnect(
    runtime_id: Optional[str] = typer.Option(
        None,
        "--runtime-id",
        "-r",
        help="Target Colab runtime endpoint ID to terminate.",
    ),
    disconnect_all: bool = typer.Option(
        False,
        "--all",
        help="Terminate all active local and remote Colab sessions.",
    ),
    auth_user_index: int = typer.Option(
        0,
        "--auth-user-index",
        "-u",
        help="Google multi-account user index (>= 0).",
    ),
) -> None:
    """Terminate one or all active Google Colab runtime sessions."""
    if auth_user_index < 0:
        raise typer.BadParameter("auth_user_index must be >= 0.", param_hint="--auth-user-index")

    from colab_mcp.runtime.tools import disconnect_runtime_impl

    try:
        resp = asyncio.run(
            disconnect_runtime_impl(
                runtime_id=runtime_id,
                disconnect_all=disconnect_all,
                auth_user_index=auth_user_index,
            )
        )
        _echo_json(resp.model_dump(mode="json"))
    except typer.Exit:
        raise
    except Exception as exc:
        _exit_with_error(exc)


@app.command("logout")
def logout(
    auth_user_index: int = typer.Option(
        0,
        "--auth-user-index",
        "-u",
        help="Google multi-account user index (>= 0).",
    ),
    terminate_active_runtimes: bool = typer.Option(
        True,
        "--terminate-active-runtimes/--no-terminate-active-runtimes",
        help="Terminate active Colab runtimes before purging credentials.",
    ),
) -> None:
    """Purge stored credentials and optionally terminate active Colab runtimes."""
    if auth_user_index < 0:
        raise typer.BadParameter("auth_user_index must be >= 0.", param_hint="--auth-user-index")

    from colab_mcp.auth.tools import colab_logout_impl

    try:
        resp = asyncio.run(
            colab_logout_impl(
                auth_user_index=auth_user_index,
                terminate_active_runtimes=terminate_active_runtimes,
            )
        )
        _echo_json(resp.model_dump(mode="json"))
    except typer.Exit:
        raise
    except Exception as exc:
        _exit_with_error(exc)


if __name__ == "__main__":
    app()
