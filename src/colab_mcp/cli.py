"""CLI entry point for ``colab-mcp`` providing full parity with the MCP server."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import typer

from colab_mcp.cli_job import (
    build_job_payload,
    build_runtime_browser_url,
    extract_job_result,
)

from colab_mcp.cli_output import (
    CLIExitCode,
    emit_error,
    emit_json,
    ensure_cli_stdout,
    render_auth_status,
    render_drive_files,
    render_execution_blocks,
    render_key_values,
    render_notebook_summary,
    render_remote_files,
    render_runtimes,
    render_secrets,
    render_sessions,
    render_system_metrics,
    save_image_blocks,
)
from colab_mcp.cli_session import (
    CLISessionState,
    cleanup_cli_runtime_resources,
    clear_cli_session,
    ensure_cli_runtime_attached,
    load_cli_session,
    save_cli_session,
)
from colab_mcp.utils.redaction import redact_credentials

app = typer.Typer(
    name="colab-mcp",
    help="Google Colab CLI & Model Context Protocol (MCP) Server",
    no_args_is_help=True,
)

_VALID_TRANSPORTS = ("stdio", "sse")
_VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _validate_user_index(auth_user_index: int) -> None:
    if auth_user_index < 0:
        raise typer.BadParameter("auth_user_index must be >= 0.", param_hint="--auth-user-index")


def _parse_env_pairs(env_items: Optional[List[str]]) -> Optional[Dict[str, str]]:
    if not env_items:
        return None
    parsed: Dict[str, str] = {}
    for item in env_items:
        if "=" not in item:
            raise ValueError(f"Invalid --env entry '{item}'. Expected KEY=VALUE format.")
        key, val = item.split("=", 1)
        if not key.strip():
            raise ValueError(f"Invalid --env entry '{item}': key must be non-empty.")
        parsed[key.strip()] = val
    return parsed


def _resolve_code_input(code: Optional[str], file_path: Optional[Path]) -> str:
    if code is not None and file_path is not None:
        raise ValueError("Provide either positional CODE or --file PATH, not both.")
    if file_path is not None:
        if not file_path.is_file():
            raise FileNotFoundError(f"Python file not found: '{file_path}'.")
        return file_path.read_text(encoding="utf-8")
    if code is None:
        raise ValueError("Provide CODE argument, '-' to read from stdin, or --file PATH.")
    if code == "-":
        ensure_cli_stdout()
        return sys.stdin.read()
    return code


def _run_with_json(fn_coro: Any, *, as_json: bool) -> Any:
    try:
        return asyncio.run(fn_coro)
    except typer.Exit:
        raise
    except Exception as exc:
        emit_error(exc, as_json=as_json)


def _make_cli_stdin_callback() -> Callable[[str, bool, Dict[str, Any]], Any]:
    """Return an async interactive stdin prompt callback with ``$SECRET:`` interpolation."""
    from colab_mcp.auth.secrets import resolve_secret_interpolation

    async def _interactive_stdin_cb(
        prompt: str,
        password: bool,
        _header: Dict[str, Any],
    ) -> Optional[str]:
        def _ask() -> str:
            ensure_cli_stdout()
            label = prompt.rstrip() or ("Secret input" if password else "Input")
            raw_val = typer.prompt(label, hide_input=password)
            return resolve_secret_interpolation(str(raw_val))

        return await asyncio.to_thread(_ask)

    return _interactive_stdin_cb


def _emit_execution_response(
    blocks: Any,
    has_error: bool,
    *,
    save_images_dir: Optional[Path],
    json_output: bool,
    exit_code_override: Optional[int] = None,
) -> None:
    """Save optional plot images, render execution blocks or JSON, and apply exit code."""
    saved_paths: List[Path] = []
    if save_images_dir is not None and isinstance(blocks, list):
        try:
            saved_paths = save_image_blocks(blocks, save_images_dir)
        except Exception as exc:
            emit_error(exc, as_json=json_output)

    code_to_exit = (
        exit_code_override
        if exit_code_override is not None
        else (int(CLIExitCode.EXECUTION_ERROR) if has_error else 0)
    )

    if json_output:
        serialized_blocks = [
            b.model_dump(mode="json") if hasattr(b, "model_dump") else b
            for b in (blocks if isinstance(blocks, list) else [blocks])
        ]
        payload: Dict[str, Any] = {
            "has_error": bool(has_error),
            "blocks": serialized_blocks,
            "saved_images": [str(p) for p in saved_paths],
        }
        if exit_code_override is not None:
            payload["exit_code"] = code_to_exit
        emit_json(payload)
    else:
        render_execution_blocks(blocks if isinstance(blocks, list) else [], saved_paths)

    if code_to_exit != 0:
        raise typer.Exit(code=code_to_exit)


# ============================================================================
# 1. Server, Version & Host Setup Commands
# ============================================================================


@app.command("version")
def version(
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Print the colab-mcp version."""
    import colab_mcp

    ensure_cli_stdout()
    if json_output:
        emit_json({"version": colab_mcp.__version__})
        return
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
        raise typer.Exit(code=int(CLIExitCode.ERROR)) from None


@app.command("setup-mcp")
def setup_mcp(
    client: str = typer.Option(
        "auto",
        "--client",
        "-c",
        help="Target host: auto, claude-desktop, claude-code, cursor, antigravity, vscode.",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config-path",
        help="Override target MCP JSON config file path.",
    ),
    server_name: str = typer.Option(
        "colab",
        "--server-name",
        help="Server key name inside mcpServers/servers.",
    ),
    command_mode: str = typer.Option(
        "auto",
        "--command",
        help="Launcher mode: auto, uvx, or python.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview changes without writing config files.",
    ),
    remove: bool = typer.Option(
        False,
        "--remove",
        help="Remove the colab entry from the target MCP config.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Auto-configure MCP hosts (Claude Desktop, Claude Code, Cursor, Antigravity, VS Code)."""
    from colab_mcp.cli_setup import configure_mcp_clients

    try:
        resp = configure_mcp_clients(
            client=client,
            config_path=config_path,
            server_name=server_name,
            command_mode=command_mode,
            dry_run=dry_run,
            remove=remove,
        )
    except Exception as exc:
        emit_error(exc, as_json=json_output)

    if json_output:
        emit_json(resp.model_dump(mode="json"))
        return

    ensure_cli_stdout()
    if not resp.results:
        typer.echo(
            "No supported MCP host directories detected. Specify --client (e.g. cursor, claude-desktop) or --config-path."
        )
        return

    rows = [
        (f"{r.client} ({r.action}{', dry-run' if r.dry_run else ''})", r.config_path)
        for r in resp.results
    ]
    render_key_values(f"MCP Host Setup ({resp.status})", rows)


# ============================================================================
# 2. Authentication & Secret Vault Commands
# ============================================================================


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
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Authenticate with Google Colab and persist credentials securely."""
    _validate_user_index(auth_user_index)
    if not (30 <= timeout_seconds <= 600):
        raise typer.BadParameter(
            "timeout_seconds must be between 30 and 600.",
            param_hint="--timeout-seconds",
        )

    from colab_mcp.auth.tools import colab_login_impl

    async def _do_login():
        resp = await colab_login_impl(
            method=method,
            auth_user_index=auth_user_index,
            auth_code=auth_code,
            timeout_seconds=timeout_seconds,
        )
        ensure_cli_stdout()
        if (
            resp.status == "authorization_required"
            and method == "oauth_remote"
            and not auth_code
            and resp.auth_url
        ):
            typer.echo(f"Authorization URL:\n{redact_credentials(resp.auth_url)}")
            entered_code = typer.prompt("Enter authorization code").strip()
            resp = await colab_login_impl(
                method=method,
                auth_user_index=auth_user_index,
                auth_code=entered_code,
                timeout_seconds=timeout_seconds,
            )
        return resp

    resp = _run_with_json(_do_login(), as_json=json_output)
    if json_output:
        emit_json(resp.model_dump(mode="json"))
        return

    render_key_values(
        "Google Colab Login",
        [
            ("Status", resp.status),
            ("Method", resp.auth_method),
            ("User Index", resp.auth_user_index),
            ("Email", resp.user_email or "-"),
            ("Message", resp.message),
        ],
    )


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
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Display authentication status, account tier, CCU balance, and runtime state."""
    _validate_user_index(auth_user_index)

    from colab_mcp.auth.tools import colab_get_auth_status_impl
    from colab_mcp.runtime.tools import get_runtime_status_impl

    async def _gather():
        try:
            auth_resp = await colab_get_auth_status_impl(
                auth_user_index=auth_user_index,
                force_refresh=force_refresh,
            )
            payload = auth_resp.model_dump(mode="json")
            saved = load_cli_session()
            if saved is not None and auth_resp.is_authenticated:
                try:
                    await ensure_cli_runtime_attached(
                        runtime_id=saved.runtime_id,
                        auth_user_index=auth_user_index,
                    )
                except Exception:
                    pass
            try:
                rt_resp = await get_runtime_status_impl()
                if rt_resp.status == "active":
                    payload["active_runtime"] = rt_resp.model_dump(mode="json")
            except Exception:
                pass
            return payload
        finally:
            await cleanup_cli_runtime_resources()

    payload = _run_with_json(_gather(), as_json=json_output)
    if json_output:
        emit_json(payload)
        return
    render_auth_status(payload)


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
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Purge stored credentials and optionally terminate active Colab runtimes."""
    _validate_user_index(auth_user_index)

    from colab_mcp.auth.tools import colab_logout_impl

    async def _do_logout():
        try:
            resp = await colab_logout_impl(
                auth_user_index=auth_user_index,
                terminate_active_runtimes=terminate_active_runtimes,
            )
            clear_cli_session()
            return resp
        finally:
            await cleanup_cli_runtime_resources()

    resp = _run_with_json(_do_logout(), as_json=json_output)
    if json_output:
        emit_json(resp.model_dump(mode="json"))
        return

    render_key_values(
        "Logged Out",
        [
            ("Status", resp.status),
            ("User Index", resp.auth_user_index),
            ("Keystore Purged", resp.keystore_purged),
            ("Terminated Runtimes", resp.terminated_runtimes_count),
        ],
    )


@app.command("set-secret")
def set_secret_cmd(
    name: str = typer.Argument(..., help="Secret identifier (e.g. HF_TOKEN, WANDB_API_KEY)."),
    value: Optional[str] = typer.Option(
        None,
        "--value",
        "-v",
        help="Secret value. Omit to enter securely via hidden prompt.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Encrypt and store a secret/token in the colab-mcp vault."""
    from colab_mcp.auth.secrets import set_secret

    try:
        secret_value = value
        if secret_value is None:
            ensure_cli_stdout()
            secret_value = typer.prompt(f"Enter value for {name}", hide_input=True)
        meta = set_secret(name=name, value=secret_value)
    except typer.Exit:
        raise
    except Exception as exc:
        emit_error(exc, as_json=json_output)

    payload = {
        "status": "stored",
        "name": meta.name,
        "masked_preview": meta.masked_preview,
        "length": meta.length,
        "updated_at": meta.updated_at.isoformat(),
    }
    if json_output:
        emit_json(payload)
        return

    render_key_values(
        "Secret Stored",
        [
            ("Name", meta.name),
            ("Preview", meta.masked_preview),
            ("Length", meta.length),
            ("Updated", meta.updated_at.isoformat()),
        ],
    )


@app.command("list-secrets")
def list_secrets_cmd(
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """List stored secret names and masked previews from the colab-mcp vault."""
    from colab_mcp.auth.secrets import list_secrets

    try:
        items = list_secrets()
    except Exception as exc:
        emit_error(exc, as_json=json_output)

    if json_output:
        emit_json({"count": len(items), "secrets": [m.model_dump(mode="json") for m in items]})
        return
    render_secrets(items)


@app.command("delete-secret")
def delete_secret_cmd(
    name: str = typer.Argument(..., help="Secret name to remove from the vault."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Delete a stored secret/token from the colab-mcp vault."""
    from colab_mcp.auth.secrets import delete_secret

    try:
        existed = delete_secret(name=name)
    except Exception as exc:
        emit_error(exc, as_json=json_output)

    clean_name = name.strip()
    if json_output:
        emit_json({"status": "deleted", "name": clean_name, "existed": existed})
        return

    ensure_cli_stdout()
    if existed:
        typer.echo(f"Deleted secret '{clean_name}' from vault.")
    else:
        typer.echo(f"Secret '{clean_name}' was not present in vault.")


# ============================================================================
# 3. Runtime Lifecycle & Hardware Telemetry Commands
# ============================================================================


@app.command("list-runtimes")
def list_runtimes(
    auth_user_index: int = typer.Option(
        0,
        "--auth-user-index",
        "-u",
        help="Google multi-account user index (>= 0).",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """List available Colab hardware accelerators, RAM shapes, and hourly CCU rates."""
    _validate_user_index(auth_user_index)

    from colab_mcp.runtime.tools import list_runtime_types_impl

    resp = _run_with_json(
        list_runtime_types_impl(auth_user_index=auth_user_index),
        as_json=json_output,
    )
    if json_output:
        emit_json(resp.model_dump(mode="json"))
        return
    render_runtimes(resp.account_tier, resp.available_accelerators)


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
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """List all active remote and locally tracked Google Colab sessions."""
    _validate_user_index(auth_user_index)

    from colab_mcp.runtime.tools import list_sessions_impl

    async def _list():
        try:
            return await list_sessions_impl(
                auth_user_index=auth_user_index,
                probe_kernels=probe_kernels,
            )
        finally:
            await cleanup_cli_runtime_resources()

    resp = _run_with_json(_list(), as_json=json_output)
    saved = load_cli_session()
    active_id = resp.active_runtime_id or (saved.runtime_id if saved else None)
    if json_output:
        payload = resp.model_dump(mode="json")
        if payload.get("active_runtime_id") is None and active_id is not None:
            payload["active_runtime_id"] = active_id
        emit_json(payload)
        return
    render_sessions(resp.sessions, active_id)


@app.command("connect-runtime")
def connect_runtime(
    accelerator: str = typer.Option(
        "NONE",
        "--accelerator",
        "-a",
        help="Hardware accelerator: NONE, T4, L4, G4, A100, H100, V5E1, V6E1.",
    ),
    high_ram: bool = typer.Option(
        False,
        "--high-ram",
        help="Request High-RAM VM shape (Pro/Pro+ only).",
    ),
    reuse_existing: bool = typer.Option(
        True,
        "--reuse-existing/--no-reuse-existing",
        help="Reuse an existing matching active session if available.",
    ),
    auth_user_index: int = typer.Option(
        0,
        "--auth-user-index",
        "-u",
        help="Google multi-account user index (>= 0).",
    ),
    timeout_seconds: int = typer.Option(
        120,
        "--timeout-seconds",
        "--timeout",
        help="Provisioning timeout in seconds (30-300).",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Provision or re-attach a Colab runtime and persist the active CLI session."""
    _validate_user_index(auth_user_index)

    from colab_mcp.runtime.tools import connect_runtime_impl

    async def _connect():
        try:
            resp = await connect_runtime_impl(
                accelerator=accelerator,
                high_ram=high_ram,
                reuse_existing=reuse_existing,
                auth_user_index=auth_user_index,
                timeout_seconds=timeout_seconds,
            )
            save_cli_session(
                CLISessionState(
                    runtime_id=resp.runtime_id,
                    kernel_id=resp.kernel_id,
                    accelerator=resp.accelerator,
                    variant=resp.variant,
                    high_ram=resp.high_ram,
                    auth_user_index=auth_user_index,
                )
            )
            return resp
        finally:
            await cleanup_cli_runtime_resources()

    resp = _run_with_json(_connect(), as_json=json_output)
    if json_output:
        emit_json(resp.model_dump(mode="json"))
        return

    render_key_values(
        "Connected Colab Runtime",
        [
            ("Runtime ID", resp.runtime_id),
            ("Accelerator", f"{resp.accelerator} ({resp.variant})"),
            ("High RAM", resp.high_ram),
            ("Reused Existing", resp.reused_existing),
            ("Kernel ID", resp.kernel_id or "-"),
            ("Kernel State", resp.kernel_state or "-"),
        ],
    )


@app.command("runtime-status")
def runtime_status(
    runtime_id: Optional[str] = typer.Option(
        None,
        "--runtime-id",
        "-r",
        help="Target Colab runtime endpoint ID (defaults to active CLI session).",
    ),
    auth_user_index: Optional[int] = typer.Option(
        None,
        "--auth-user-index",
        "-u",
        help="Google multi-account user index override.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Inspect active VM health, kernel execution state, and uptime."""
    if auth_user_index is not None:
        _validate_user_index(auth_user_index)

    from colab_mcp.runtime.tools import get_runtime_status_impl

    async def _inspect():
        try:
            attached_id, _ = await ensure_cli_runtime_attached(
                runtime_id=runtime_id,
                auth_user_index=auth_user_index,
            )
            return await get_runtime_status_impl(runtime_id=attached_id)
        finally:
            await cleanup_cli_runtime_resources()

    resp = _run_with_json(_inspect(), as_json=json_output)
    if json_output:
        emit_json(resp.model_dump(mode="json"))
        return

    render_key_values(
        "Colab Runtime Status",
        [
            ("Runtime ID", resp.runtime_id or "-"),
            ("Status", resp.status),
            ("Kernel State", resp.kernel_state or "-"),
            ("Accelerator", (resp.metrics or {}).get("accelerator", "-")),
            ("High RAM", (resp.metrics or {}).get("high_ram", "-")),
        ],
    )


@app.command("runtime-url")
def runtime_url(
    runtime_id: Optional[str] = typer.Option(
        None,
        "--runtime-id",
        "-r",
        help="Target Colab runtime endpoint ID (defaults to active CLI session).",
    ),
    auth_user_index: Optional[int] = typer.Option(
        None,
        "--auth-user-index",
        "-u",
        help="Google multi-account user index override.",
    ),
    host: str = typer.Option(
        "https://colab.research.google.com",
        "--host",
        help="Colab web UI host origin.",
    ),
    open_browser: bool = typer.Option(
        False,
        "--open",
        help="Open the generated attach URL in the default web browser.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Print (or open) a Colab browser URL attached to an active CLI/MCP runtime."""
    if auth_user_index is not None:
        _validate_user_index(auth_user_index)

    from colab_mcp.runtime.session_manager import get_session_manager

    async def _resolve_url() -> Dict[str, str]:
        try:
            attached_id, _ = await ensure_cli_runtime_attached(
                runtime_id=runtime_id,
                auth_user_index=auth_user_index,
            )
            ep, _, _, _ = await get_session_manager().get_runtime_proxy_params(
                runtime_id=attached_id
            )
            return {
                "runtime_id": ep,
                "url": build_runtime_browser_url(ep, host=host),
            }
        finally:
            await cleanup_cli_runtime_resources()

    data = _run_with_json(_resolve_url(), as_json=json_output)
    ensure_cli_stdout()
    if json_output:
        emit_json(data)
    else:
        typer.echo(data["url"])

    if open_browser:
        webbrowser.open(data["url"])


@app.command("system-metrics")
def system_metrics(
    runtime_id: Optional[str] = typer.Option(
        None,
        "--runtime-id",
        "-r",
        help="Target Colab runtime endpoint ID (defaults to active CLI session).",
    ),
    auth_user_index: Optional[int] = typer.Option(
        None,
        "--auth-user-index",
        "-u",
        help="Google multi-account user index override.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Inspect live GPU VRAM, GPU utilization %, system RAM, and /content disk usage."""
    if auth_user_index is not None:
        _validate_user_index(auth_user_index)

    from colab_mcp.runtime.tools import get_system_metrics_impl

    async def _metrics():
        try:
            attached_id, _ = await ensure_cli_runtime_attached(
                runtime_id=runtime_id,
                auth_user_index=auth_user_index,
            )
            return await get_system_metrics_impl(runtime_id=attached_id)
        finally:
            await cleanup_cli_runtime_resources()

    resp = _run_with_json(_metrics(), as_json=json_output)
    if json_output:
        emit_json(resp.model_dump(mode="json"))
        return
    render_system_metrics(resp)


@app.command("disconnect")
def disconnect(
    runtime_id: Optional[str] = typer.Option(
        None,
        "--runtime-id",
        "-r",
        help="Target Colab runtime endpoint ID to terminate (defaults to active CLI session).",
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
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Terminate one or all active Google Colab runtime sessions."""
    _validate_user_index(auth_user_index)

    from colab_mcp.runtime.tools import disconnect_runtime_impl

    async def _disconnect():
        try:
            saved = load_cli_session()
            target_id = runtime_id
            effective_user = auth_user_index
            if target_id is None and not disconnect_all and saved is not None:
                target_id = saved.runtime_id
                if auth_user_index == 0:
                    effective_user = saved.auth_user_index
            resp = await disconnect_runtime_impl(
                runtime_id=target_id,
                disconnect_all=disconnect_all,
                auth_user_index=effective_user,
            )
            if disconnect_all or target_id is None:
                clear_cli_session()
            else:
                clear_cli_session(runtime_id=target_id)
            return resp
        finally:
            await cleanup_cli_runtime_resources()

    resp = _run_with_json(_disconnect(), as_json=json_output)
    if json_output:
        emit_json(resp.model_dump(mode="json"))
        return

    render_key_values(
        "Disconnected Runtime",
        [
            ("Status", resp.status),
            ("Runtime ID", resp.runtime_id or "-"),
            ("Terminated Count", resp.terminated_count),
            ("Terminated IDs", ", ".join(resp.terminated_runtime_ids) or "-"),
            ("Message", resp.message),
        ],
    )


# ============================================================================
# 4. Code & Notebook Execution Commands
# ============================================================================


@app.command("execute-code")
def execute_code(
    code: Optional[str] = typer.Argument(
        None,
        help="Python code string, or '-' to read from stdin (omit when passing --file).",
    ),
    file: Optional[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="Path to a local Python script to execute on Colab.",
    ),
    runtime_id: Optional[str] = typer.Option(
        None,
        "--runtime-id",
        "-r",
        help="Target Colab runtime endpoint ID (defaults to active CLI session).",
    ),
    auth_user_index: Optional[int] = typer.Option(
        None,
        "--auth-user-index",
        "-u",
        help="Google multi-account user index override.",
    ),
    timeout_seconds: int = typer.Option(
        300,
        "--timeout-seconds",
        "--timeout",
        help="Execution timeout in seconds (5-3600).",
    ),
    stdin_input: Optional[List[str]] = typer.Option(
        None,
        "--stdin-input",
        help="Pre-supplied input() reply (repeatable; supports '$SECRET:<NAME>').",
    ),
    env: Optional[List[str]] = typer.Option(
        None,
        "--env",
        "-e",
        help="Environment variable override in KEY=VALUE format (repeatable).",
    ),
    save_images_dir: Optional[Path] = typer.Option(
        None,
        "--save-images-dir",
        help="Directory to write Matplotlib/display images as PNG/JPEG files.",
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Fail immediately (exit 13) if code prompts via input()/getpass() without --stdin-input.",
    ),
    max_text_lines: int = typer.Option(1000, "--max-text-lines", help="Max output text lines."),
    max_images: int = typer.Option(5, "--max-images", help="Max plot images to collect."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Execute Python code or IPython magic on the active Colab runtime."""
    if auth_user_index is not None:
        _validate_user_index(auth_user_index)

    from colab_mcp.kernel.tools import execute_code_impl

    try:
        source_code = _resolve_code_input(code, file)
        env_map = _parse_env_pairs(env)
    except Exception as exc:
        emit_error(exc, as_json=json_output)

    stdin_cb = None if non_interactive else _make_cli_stdin_callback()

    async def _exec():
        try:
            attached_id, _ = await ensure_cli_runtime_attached(
                runtime_id=runtime_id,
                auth_user_index=auth_user_index,
            )
            return await execute_code_impl(
                code=source_code,
                runtime_id=attached_id,
                timeout_seconds=timeout_seconds,
                wait_for_completion=True,
                store_history=True,
                max_text_lines=max_text_lines,
                max_images=max_images,
                stdin_inputs=list(stdin_input) if stdin_input else None,
                env=env_map,
                _on_input_request=stdin_cb,
            )
        finally:
            await cleanup_cli_runtime_resources()

    blocks, has_error = _run_with_json(_exec(), as_json=json_output)
    _emit_execution_response(
        blocks,
        has_error,
        save_images_dir=save_images_dir,
        json_output=json_output,
    )


@app.command("job", context_settings={"allow_interspersed_args": False})
def job(
    script: Path = typer.Argument(..., help="Local Python script to run on a fresh Colab VM."),
    script_args: Optional[List[str]] = typer.Argument(
        None,
        metavar="[ARGS...]",
        help="Arguments forwarded to SCRIPT as sys.argv[1:].",
    ),
    accelerator: str = typer.Option(
        "NONE",
        "--accelerator",
        "-a",
        help="Hardware accelerator: NONE, T4, L4, G4, A100, H100, V5E1, V6E1.",
    ),
    high_ram: bool = typer.Option(
        False,
        "--high-ram",
        help="Request High-RAM VM shape (Pro/Pro+ only).",
    ),
    auth_user_index: int = typer.Option(
        0,
        "--auth-user-index",
        "-u",
        help="Google multi-account user index (>= 0).",
    ),
    connect_timeout: int = typer.Option(
        120,
        "--connect-timeout",
        help="VM provisioning timeout in seconds (30-300).",
    ),
    timeout_seconds: int = typer.Option(
        300,
        "--timeout-seconds",
        "--timeout",
        help="Script execution timeout in seconds (5-3600).",
    ),
    stdin_input: Optional[List[str]] = typer.Option(
        None,
        "--stdin-input",
        help="Pre-supplied input() reply (repeatable; supports '$SECRET:<NAME>').",
    ),
    env: Optional[List[str]] = typer.Option(
        None,
        "--env",
        "-e",
        help="Environment variable override in KEY=VALUE format (repeatable).",
    ),
    save_images_dir: Optional[Path] = typer.Option(
        None,
        "--save-images-dir",
        help="Directory to write Matplotlib/display images as PNG/JPEG files.",
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Fail immediately (exit 13) if script prompts via input()/getpass() without --stdin-input.",
    ),
    max_text_lines: int = typer.Option(1000, "--max-text-lines", help="Max output text lines."),
    max_images: int = typer.Option(5, "--max-images", help="Max plot images to collect."),
    keep: bool = typer.Option(
        False,
        "--keep",
        help="Keep the provisioned Colab VM alive after script completion.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Provision an ephemeral Colab VM, execute SCRIPT [ARGS...], and auto-disconnect."""
    _validate_user_index(auth_user_index)

    if not script.is_file():
        emit_error(FileNotFoundError(f"Python script not found: '{script}'."), as_json=json_output)

    from colab_mcp.kernel.tools import execute_code_impl
    from colab_mcp.runtime.tools import connect_runtime_impl, disconnect_runtime_impl

    try:
        env_map = _parse_env_pairs(env)
        source_payload = build_job_payload(script, script_args or [])
    except Exception as exc:
        emit_error(exc, as_json=json_output)

    stdin_cb = None if non_interactive else _make_cli_stdin_callback()

    async def _run_job() -> tuple[List[Any], int, bool]:
        provisioned_id: Optional[str] = None
        try:
            if not json_output:
                typer.echo(
                    f"[colab-mcp] Provisioning ephemeral {accelerator.upper()} runtime...",
                    err=True,
                )
            conn = await connect_runtime_impl(
                accelerator=accelerator,
                high_ram=high_ram,
                reuse_existing=False,
                auth_user_index=auth_user_index,
                timeout_seconds=connect_timeout,
            )
            provisioned_id = conn.runtime_id
            if not json_output:
                typer.echo(
                    f"[colab-mcp] Executing '{script.name}' on {provisioned_id}...",
                    err=True,
                )
            blocks, has_error = await execute_code_impl(
                code=source_payload,
                runtime_id=provisioned_id,
                timeout_seconds=timeout_seconds,
                wait_for_completion=True,
                store_history=False,
                max_text_lines=max_text_lines,
                max_images=max_images,
                stdin_inputs=list(stdin_input) if stdin_input else None,
                env=env_map,
                _on_input_request=stdin_cb,
            )
            cleaned_blocks, exit_code = extract_job_result(
                blocks if isinstance(blocks, list) else [],
                has_error=has_error,
            )
            if keep:
                save_cli_session(
                    CLISessionState(
                        runtime_id=conn.runtime_id,
                        kernel_id=conn.kernel_id,
                        accelerator=conn.accelerator,
                        variant=conn.variant,
                        high_ram=conn.high_ram,
                        auth_user_index=auth_user_index,
                    )
                )
            return cleaned_blocks, exit_code, has_error
        finally:
            if provisioned_id is not None and not keep:
                if not json_output:
                    typer.echo(
                        f"[colab-mcp] Releasing ephemeral runtime {provisioned_id}...",
                        err=True,
                    )
                try:
                    await disconnect_runtime_impl(
                        runtime_id=provisioned_id,
                        disconnect_all=False,
                        auth_user_index=auth_user_index,
                    )
                except Exception as teardown_exc:
                    typer.echo(
                        f"[colab-mcp] Warning: failed to release '{provisioned_id}': "
                        f"{redact_credentials(str(teardown_exc))}",
                        err=True,
                    )
                clear_cli_session(runtime_id=provisioned_id)
            await cleanup_cli_runtime_resources()

    cleaned_blocks, exit_code, has_error = _run_with_json(_run_job(), as_json=json_output)
    _emit_execution_response(
        cleaned_blocks,
        has_error=(exit_code != 0) or has_error,
        save_images_dir=save_images_dir,
        json_output=json_output,
        exit_code_override=exit_code,
    )


@app.command("execute-notebook")
def execute_notebook(
    notebook_path: Path = typer.Argument(..., help="Path to local .ipynb notebook file."),
    runtime_id: Optional[str] = typer.Option(
        None,
        "--runtime-id",
        "-r",
        help="Target Colab runtime endpoint ID (defaults to active CLI session).",
    ),
    auth_user_index: Optional[int] = typer.Option(
        None,
        "--auth-user-index",
        "-u",
        help="Google multi-account user index override.",
    ),
    reset_kernel: bool = typer.Option(
        True,
        "--reset-kernel/--no-reset-kernel",
        help="Restart the Jupyter kernel before executing cells.",
    ),
    cell_index: Optional[List[int]] = typer.Option(
        None,
        "--cell-index",
        help="0-based cell index to execute (repeatable).",
    ),
    stop_on_error: bool = typer.Option(
        True,
        "--stop-on-error/--continue-on-error",
        help="Halt execution on the first failing cell.",
    ),
    timeout_per_cell: int = typer.Option(
        300,
        "--timeout-per-cell",
        help="Timeout per cell in seconds (5-14400).",
    ),
    output_save_path: Optional[str] = typer.Option(
        None,
        "--output-save-path",
        help="Optional output path to serialize the executed .ipynb file.",
    ),
    stdin_input: Optional[List[str]] = typer.Option(
        None,
        "--stdin-input",
        help="Pre-supplied input() replies consumed sequentially across cells.",
    ),
    env: Optional[List[str]] = typer.Option(
        None,
        "--env",
        "-e",
        help="Environment variable override in KEY=VALUE format (repeatable).",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Execute a Jupyter notebook (.ipynb) cell-by-cell on the connected Colab kernel."""
    if auth_user_index is not None:
        _validate_user_index(auth_user_index)

    from colab_mcp.notebook.runner import execute_notebook_impl

    try:
        env_map = _parse_env_pairs(env)
    except Exception as exc:
        emit_error(exc, as_json=json_output)

    async def _exec_nb():
        try:
            attached_id, _ = await ensure_cli_runtime_attached(
                runtime_id=runtime_id,
                auth_user_index=auth_user_index,
            )
            return await execute_notebook_impl(
                notebook_path=str(notebook_path),
                reset_kernel_before_run=reset_kernel,
                cell_indices=list(cell_index) if cell_index else None,
                stop_on_error=stop_on_error,
                timeout_per_cell_seconds=timeout_per_cell,
                output_save_path=output_save_path,
                runtime_id=attached_id,
                stdin_inputs=list(stdin_input) if stdin_input else None,
                env=env_map,
            )
        finally:
            await cleanup_cli_runtime_resources()

    resp = _run_with_json(_exec_nb(), as_json=json_output)
    if json_output:
        emit_json(resp.model_dump(mode="json"))
    else:
        render_notebook_summary(resp)

    if resp.failed_cells > 0:
        raise typer.Exit(code=int(CLIExitCode.EXECUTION_ERROR))


@app.command("interrupt")
def interrupt(
    runtime_id: Optional[str] = typer.Option(
        None,
        "--runtime-id",
        "-r",
        help="Target Colab runtime endpoint ID (defaults to active CLI session).",
    ),
    auth_user_index: Optional[int] = typer.Option(
        None,
        "--auth-user-index",
        "-u",
        help="Google multi-account user index override.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Send SIGINT (escalating to kernel restart if hung) to the active Colab kernel."""
    if auth_user_index is not None:
        _validate_user_index(auth_user_index)

    from colab_mcp.kernel.tools import interrupt_execution_impl

    async def _interrupt():
        try:
            attached_id, _ = await ensure_cli_runtime_attached(
                runtime_id=runtime_id,
                auth_user_index=auth_user_index,
            )
            return await interrupt_execution_impl(runtime_id=attached_id)
        finally:
            await cleanup_cli_runtime_resources()

    data = _run_with_json(_interrupt(), as_json=json_output)
    if json_output:
        emit_json(data)
        return

    render_key_values(
        "Kernel Interrupt Result",
        [
            ("Runtime ID", data.get("runtime_id") or "-"),
            ("Status", data.get("status")),
            ("Kernel State", data.get("kernel_state")),
            ("Kernel Restarted", data.get("kernel_restarted")),
            ("Message", data.get("message")),
        ],
    )


# ============================================================================
# 5. VM File Transfer & Google Drive Commands
# ============================================================================


@app.command("list-files")
def list_files(
    remote_path: str = typer.Argument(
        "/content",
        help="Directory or file path on the Colab VM.",
    ),
    runtime_id: Optional[str] = typer.Option(
        None,
        "--runtime-id",
        "-r",
        help="Target Colab runtime endpoint ID (defaults to active CLI session).",
    ),
    auth_user_index: Optional[int] = typer.Option(
        None,
        "--auth-user-index",
        "-u",
        help="Google multi-account user index override.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """List files and directories on the connected Colab VM."""
    if auth_user_index is not None:
        _validate_user_index(auth_user_index)

    from colab_mcp.files.tools import list_files_impl

    async def _ls():
        try:
            attached_id, _ = await ensure_cli_runtime_attached(
                runtime_id=runtime_id,
                auth_user_index=auth_user_index,
            )
            return await list_files_impl(remote_path=remote_path, runtime_id=attached_id)
        finally:
            await cleanup_cli_runtime_resources()

    resp = _run_with_json(_ls(), as_json=json_output)
    if json_output:
        emit_json(resp.model_dump(mode="json"))
        return
    render_remote_files(resp)


@app.command("upload-file")
def upload_file(
    local_path: str = typer.Argument(..., help="Local file path to upload."),
    remote_path: str = typer.Argument(..., help="Destination path on Colab VM (e.g. /content/data.zip)."),
    runtime_id: Optional[str] = typer.Option(
        None,
        "--runtime-id",
        "-r",
        help="Target Colab runtime endpoint ID (defaults to active CLI session).",
    ),
    auth_user_index: Optional[int] = typer.Option(
        None,
        "--auth-user-index",
        "-u",
        help="Google multi-account user index override.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Upload a local file to the connected Colab VM."""
    if auth_user_index is not None:
        _validate_user_index(auth_user_index)

    from colab_mcp.files.tools import upload_file_impl

    async def _up():
        try:
            attached_id, _ = await ensure_cli_runtime_attached(
                runtime_id=runtime_id,
                auth_user_index=auth_user_index,
            )
            return await upload_file_impl(
                local_path=local_path,
                remote_path=remote_path,
                runtime_id=attached_id,
            )
        finally:
            await cleanup_cli_runtime_resources()

    resp = _run_with_json(_up(), as_json=json_output)
    if json_output:
        emit_json(resp.model_dump(mode="json"))
        return

    render_key_values(
        "File Uploaded to Colab VM",
        [
            ("Runtime ID", resp.runtime_id),
            ("Local Path", resp.local_path),
            ("Remote Path", resp.remote_path),
            ("Size (bytes)", resp.size_bytes),
            ("SHA-256", resp.sha256),
            ("Transfer Method", resp.transfer_method),
        ],
    )


@app.command("download-file")
def download_file(
    remote_path: str = typer.Argument(..., help="Source file path on Colab VM."),
    local_path: str = typer.Argument(..., help="Destination file path on local disk."),
    runtime_id: Optional[str] = typer.Option(
        None,
        "--runtime-id",
        "-r",
        help="Target Colab runtime endpoint ID (defaults to active CLI session).",
    ),
    auth_user_index: Optional[int] = typer.Option(
        None,
        "--auth-user-index",
        "-u",
        help="Google multi-account user index override.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Download a file from the connected Colab VM to local disk."""
    if auth_user_index is not None:
        _validate_user_index(auth_user_index)

    from colab_mcp.files.tools import download_file_impl

    async def _dl():
        try:
            attached_id, _ = await ensure_cli_runtime_attached(
                runtime_id=runtime_id,
                auth_user_index=auth_user_index,
            )
            return await download_file_impl(
                remote_path=remote_path,
                local_path=local_path,
                runtime_id=attached_id,
            )
        finally:
            await cleanup_cli_runtime_resources()

    resp = _run_with_json(_dl(), as_json=json_output)
    if json_output:
        emit_json(resp.model_dump(mode="json"))
        return

    render_key_values(
        "File Downloaded from Colab VM",
        [
            ("Runtime ID", resp.runtime_id),
            ("Remote Path", resp.remote_path),
            ("Local Path", resp.local_path),
            ("Size (bytes)", resp.size_bytes),
            ("SHA-256", resp.sha256),
            ("Transfer Method", resp.transfer_method),
        ],
    )


@app.command("delete-file")
def delete_file(
    remote_path: str = typer.Argument(..., help="Remote file or directory path on Colab VM."),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        help="Recursively remove non-empty directories.",
    ),
    runtime_id: Optional[str] = typer.Option(
        None,
        "--runtime-id",
        "-r",
        help="Target Colab runtime endpoint ID (defaults to active CLI session).",
    ),
    auth_user_index: Optional[int] = typer.Option(
        None,
        "--auth-user-index",
        "-u",
        help="Google multi-account user index override.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Delete a file or directory on the connected Colab VM."""
    if auth_user_index is not None:
        _validate_user_index(auth_user_index)

    from colab_mcp.files.tools import delete_file_impl

    async def _rm():
        try:
            attached_id, _ = await ensure_cli_runtime_attached(
                runtime_id=runtime_id,
                auth_user_index=auth_user_index,
            )
            return await delete_file_impl(
                remote_path=remote_path,
                recursive=recursive,
                runtime_id=attached_id,
            )
        finally:
            await cleanup_cli_runtime_resources()

    resp = _run_with_json(_rm(), as_json=json_output)
    if json_output:
        emit_json(resp.model_dump(mode="json"))
        return

    render_key_values(
        "Deleted Remote File",
        [
            ("Runtime ID", resp.runtime_id),
            ("Remote Path", resp.remote_path),
            ("Recursive", resp.recursive),
            ("Method", resp.delete_method),
        ],
    )


@app.command("drive-list")
def drive_list(
    folder: str = typer.Option(
        "MyDrive",
        "--folder",
        "-f",
        help="Google Drive folder path or folder ID.",
    ),
    query: Optional[str] = typer.Option(
        None,
        "--query",
        "-q",
        help="Optional substring filter on file/folder name.",
    ),
    max_results: int = typer.Option(50, "--max-results", help="Maximum results (1-200)."),
    auth_user_index: int = typer.Option(
        0,
        "--auth-user-index",
        "-u",
        help="Google multi-account user index (>= 0).",
    ),
    quota_project_id: Optional[str] = typer.Option(
        None,
        "--quota-project-id",
        help="Optional GCP project ID for X-Goog-User-Project quota attribution.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """List files and folders in Google Drive via Drive API v3."""
    _validate_user_index(auth_user_index)

    from colab_mcp.files.tools import drive_list_impl

    resp = _run_with_json(
        drive_list_impl(
            folder=folder,
            query=query,
            max_results=max_results,
            auth_user_index=auth_user_index,
            quota_project_id=quota_project_id,
        ),
        as_json=json_output,
    )
    if json_output:
        emit_json(resp.model_dump(mode="json"))
        return
    render_drive_files(resp)


@app.command("drive-download")
def drive_download(
    file_id_or_path: str = typer.Argument(
        ...,
        help="Google Drive file ID or path (e.g. 'MyDrive/data/dataset.zip').",
    ),
    local_path: Optional[str] = typer.Option(
        None,
        "--local-path",
        "-l",
        help="Destination path on local disk.",
    ),
    runtime_path: Optional[str] = typer.Option(
        None,
        "--runtime-path",
        help="Destination path directly on the connected Colab VM.",
    ),
    runtime_id: Optional[str] = typer.Option(
        None,
        "--runtime-id",
        "-r",
        help="Target Colab runtime endpoint ID when --runtime-path is set.",
    ),
    auth_user_index: int = typer.Option(
        0,
        "--auth-user-index",
        "-u",
        help="Google multi-account user index (>= 0).",
    ),
    quota_project_id: Optional[str] = typer.Option(
        None,
        "--quota-project-id",
        help="Optional GCP quota project ID.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Download a file from Google Drive to local disk and/or the connected Colab VM."""
    _validate_user_index(auth_user_index)

    from colab_mcp.files.tools import drive_download_impl

    async def _ddl():
        try:
            target_rt = runtime_id
            if runtime_path is not None:
                target_rt, _ = await ensure_cli_runtime_attached(
                    runtime_id=runtime_id,
                    auth_user_index=auth_user_index,
                )
            return await drive_download_impl(
                file_id_or_path=file_id_or_path,
                local_path=local_path,
                runtime_path=runtime_path,
                auth_user_index=auth_user_index,
                runtime_id=target_rt,
                quota_project_id=quota_project_id,
            )
        finally:
            await cleanup_cli_runtime_resources()

    resp = _run_with_json(_ddl(), as_json=json_output)
    if json_output:
        emit_json(resp.model_dump(mode="json"))
        return

    render_key_values(
        "Google Drive Download Complete",
        [
            ("File Name", resp.file_name),
            ("File ID", resp.file_id),
            ("Size (bytes)", resp.size_bytes),
            ("SHA-256", resp.sha256 or "-"),
            ("Local Path", resp.local_path or "-"),
            ("Runtime Path", resp.runtime_path or "-"),
        ],
    )


@app.command("drive-upload")
def drive_upload(
    local_path: Optional[str] = typer.Option(
        None,
        "--local-path",
        "-l",
        help="Source file path on local disk.",
    ),
    runtime_path: Optional[str] = typer.Option(
        None,
        "--runtime-path",
        help="Source file path on the connected Colab VM.",
    ),
    drive_folder: str = typer.Option(
        "MyDrive",
        "--drive-folder",
        "-f",
        help="Destination folder path or ID in Google Drive.",
    ),
    file_name: Optional[str] = typer.Option(
        None,
        "--file-name",
        help="Override uploaded file name in Google Drive.",
    ),
    runtime_id: Optional[str] = typer.Option(
        None,
        "--runtime-id",
        "-r",
        help="Target Colab runtime endpoint ID when --runtime-path is set.",
    ),
    auth_user_index: int = typer.Option(
        0,
        "--auth-user-index",
        "-u",
        help="Google multi-account user index (>= 0).",
    ),
    quota_project_id: Optional[str] = typer.Option(
        None,
        "--quota-project-id",
        help="Optional GCP quota project ID.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Upload a local or Colab VM file into Google Drive."""
    _validate_user_index(auth_user_index)

    from colab_mcp.files.tools import drive_upload_impl

    async def _dup():
        try:
            target_rt = runtime_id
            if runtime_path is not None:
                target_rt, _ = await ensure_cli_runtime_attached(
                    runtime_id=runtime_id,
                    auth_user_index=auth_user_index,
                )
            return await drive_upload_impl(
                local_path=local_path,
                runtime_path=runtime_path,
                drive_folder=drive_folder,
                file_name=file_name,
                auth_user_index=auth_user_index,
                runtime_id=target_rt,
                quota_project_id=quota_project_id,
            )
        finally:
            await cleanup_cli_runtime_resources()

    resp = _run_with_json(_dup(), as_json=json_output)
    if json_output:
        emit_json(resp.model_dump(mode="json"))
        return

    render_key_values(
        "Google Drive Upload Complete",
        [
            ("File Name", resp.file_name),
            ("File ID", resp.file_id),
            ("Size (bytes)", resp.size_bytes),
            ("SHA-256", resp.sha256 or "-"),
            ("Drive Folder", drive_folder),
            ("Local Source", resp.local_path or "-"),
            ("Runtime Source", resp.runtime_path or "-"),
        ],
    )


# ============================================================================
# 6. Remote SSH Access Command
# ============================================================================


async def _disconnect_ssh_runtime(runtime_id: str, auth_user_index: int) -> None:
    """Best-effort VM teardown on SSH exit for ephemeral ``--rm`` sessions."""
    from colab_mcp.runtime.tools import disconnect_runtime_impl

    try:
        await disconnect_runtime_impl(
            runtime_id=runtime_id,
            disconnect_all=False,
            auth_user_index=auth_user_index,
        )
    except Exception as exc:
        typer.echo(
            f"[colab-mcp] Warning: failed to stop runtime '{runtime_id}': "
            f"{redact_credentials(str(exc))}",
            err=True,
        )
    finally:
        clear_cli_session(runtime_id=runtime_id)
        await cleanup_cli_runtime_resources()


async def _prepare_ssh_runtime(
    *,
    runtime_id: Optional[str],
    auth_user_index: Optional[int],
    accelerator: Optional[str],
    high_ram: bool,
    allow_create: bool,
) -> tuple[str, int, bool]:
    """Attach to an active runtime or auto-create one when allowed."""
    from colab_mcp.runtime.models import RuntimeDisconnectedError
    from colab_mcp.runtime.tools import connect_runtime_impl

    try:
        attached_id, effective_user = await ensure_cli_runtime_attached(
            runtime_id=runtime_id,
            auth_user_index=auth_user_index,
        )
        return attached_id, effective_user, False
    except RuntimeDisconnectedError:
        if not allow_create or runtime_id is not None:
            raise

    effective_user = auth_user_index if auth_user_index is not None else 0
    requested_acc = accelerator or "NONE"
    typer.echo(
        f"[colab-mcp] Creating {requested_acc.upper()} runtime for SSH session...",
        err=True,
    )
    conn = await connect_runtime_impl(
        accelerator=requested_acc,
        high_ram=high_ram,
        reuse_existing=True,
        auth_user_index=effective_user,
        timeout_seconds=120,
    )
    save_cli_session(
        CLISessionState(
            runtime_id=conn.runtime_id,
            kernel_id=conn.kernel_id,
            accelerator=conn.accelerator,
            variant=conn.variant,
            high_ram=conn.high_ram,
            auth_user_index=effective_user,
        )
    )
    return conn.runtime_id, effective_user, True


async def _kill_stale_sshd(runtime_id: str) -> None:
    """Terminate stale SSH session processes on the Colab VM via Jupyter kernel."""
    from colab_mcp.kernel.tools import execute_code_impl

    typer.echo("[colab-mcp] --force: clearing stale SSH sessions on runtime...", err=True)
    try:
        await execute_code_impl(
            code="import os; os.system('pkill -P $(pgrep -o -x sshd) 2>/dev/null || true')",
            runtime_id=runtime_id,
            timeout_seconds=15,
            wait_for_completion=True,
            store_history=False,
        )
    except Exception as exc:
        typer.echo(
            f"[colab-mcp] Warning: --force cleanup failed: {redact_credentials(str(exc))}",
            err=True,
        )


@app.command("ssh", context_settings={"allow_interspersed_args": False})
def ssh(
    cmd: Optional[List[str]] = typer.Argument(
        None,
        metavar="[-- CMD [ARGS...]]",
        help="Optional remote command to execute in /content without launching an interactive shell.",
    ),
    runtime_id: Optional[str] = typer.Option(
        None,
        "--runtime-id",
        "-r",
        help="Target Colab runtime endpoint ID (defaults to active CLI session).",
    ),
    identity: Optional[str] = typer.Option(
        None,
        "--identity",
        "-i",
        help="SSH public (.pub) or private key path (defaults to ~/.ssh/id_ed25519.pub or id_ecdsa.pub).",
    ),
    accelerator: Optional[str] = typer.Option(
        None,
        "--accelerator",
        "-a",
        help="Accelerator if auto-creating a runtime: NONE, T4, L4, G4, A100, H100, V5E1, V6E1.",
    ),
    high_ram: bool = typer.Option(
        False,
        "--high-ram",
        help="Request High-RAM VM shape when auto-creating a runtime.",
    ),
    auth_user_index: Optional[int] = typer.Option(
        None,
        "--auth-user-index",
        "-u",
        help="Google multi-account user index override.",
    ),
    proxy_mode: bool = typer.Option(
        False,
        "--proxy-mode",
        help="Run as an OpenSSH ProxyCommand WebSocket-to-stdio bridge.",
    ),
    ssh_mode: str = typer.Option(
        "proxy",
        "--ssh-mode",
        hidden=True,
        help="Internal SSH client agent mode tag.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Terminate stale VM sshd child sessions before connecting (useful on HTTP 429).",
    ),
    rm: bool = typer.Option(
        False,
        "--rm",
        help="Disconnect an auto-created runtime (or bridged --proxy-mode runtime) on exit.",
    ),
    ssh_config: bool = typer.Option(
        False,
        "--ssh-config",
        help="Print a ready-to-paste ~/.ssh/config entry for VS Code / Cursor Remote-SSH and exit.",
    ),
) -> None:
    """Connect to a Colab VM over WebSocket SSH (interactive shell, ProxyCommand, or -- CMD)."""
    if auth_user_index is not None:
        _validate_user_index(auth_user_index)

    from colab_mcp.runtime.session_manager import get_session_manager
    from colab_mcp.runtime.ssh import (
        bridge_ws_stdio,
        build_proxy_command,
        build_ssh_subprocess_args,
        connect_ssh_websocket,
        install_rm_signal_handlers,
        render_ssh_config_snippet,
        resolve_ssh_pubkey,
    )

    if ssh_config:
        ensure_cli_stdout()
        typer.echo(
            render_ssh_config_snippet(
                runtime_id=runtime_id,
                identity=identity,
                accelerator=accelerator or "T4",
                rm=rm,
            )
        )
        return

    if proxy_mode and cmd:
        emit_error(
            ValueError("--proxy-mode cannot be combined with positional CMD arguments."),
            as_json=False,
        )

    try:
        pubkey = resolve_ssh_pubkey(identity)
    except Exception as exc:
        emit_error(exc, as_json=False)

    if proxy_mode:
        target_info: Dict[str, Any] = {"endpoint": None, "user": auth_user_index or 0}
        restore_signals = install_rm_signal_handlers(lambda: None) if rm else None

        async def _bridge() -> int:
            try:
                ep, eff_user, created = await _prepare_ssh_runtime(
                    runtime_id=runtime_id,
                    auth_user_index=auth_user_index,
                    accelerator=accelerator,
                    high_ram=high_ram,
                    allow_create=(ssh_mode != "cmd"),
                )
                target_info["endpoint"] = ep
                target_info["user"] = eff_user
                if (accelerator or high_ram) and not created:
                    typer.echo(
                        "[colab-mcp] --accelerator/--high-ram ignored: reusing existing runtime.",
                        err=True,
                    )
                if force:
                    await _kill_stale_sshd(ep)
                _, proxy_url, proxy_token, _ = await get_session_manager().get_runtime_proxy_params(
                    runtime_id=ep
                )
                ws = await connect_ssh_websocket(
                    proxy_url=proxy_url,
                    proxy_token=proxy_token,
                    pubkey=pubkey,
                    mode=ssh_mode,
                )
                return await bridge_ws_stdio(ws)
            finally:
                await cleanup_cli_runtime_resources()

        rc = 0
        try:
            rc = _run_with_json(_bridge(), as_json=False)
        finally:
            if restore_signals is not None:
                restore_signals()
            ep_to_stop = target_info.get("endpoint")
            if rm and ep_to_stop:
                try:
                    asyncio.run(
                        _disconnect_ssh_runtime(ep_to_stop, int(target_info.get("user", 0)))
                    )
                except Exception:
                    pass
        raise typer.Exit(code=int(rc))

    # Interactive SSH shell or one-shot `-- CMD`
    is_cmd_mode = bool(cmd)
    from colab_mcp.runtime.ssh import ColabSSHError
    import shutil as _shutil

    if not _shutil.which("ssh"):
        emit_error(
            ColabSSHError(
                "OpenSSH client (`ssh`) was not found on PATH. Install OpenSSH or use `--proxy-mode`."
            ),
            as_json=False,
        )

    async def _prep() -> tuple[str, int, bool]:
        try:
            return await _prepare_ssh_runtime(
                runtime_id=runtime_id,
                auth_user_index=auth_user_index,
                accelerator=accelerator,
                high_ram=high_ram,
                allow_create=not is_cmd_mode,
            )
        finally:
            await cleanup_cli_runtime_resources()

    ep, eff_user, created_here = _run_with_json(_prep(), as_json=False)

    if (accelerator or high_ram) and not created_here:
        typer.echo(
            "[colab-mcp] --accelerator/--high-ram ignored: reusing existing runtime.",
            err=True,
        )
    if rm and (is_cmd_mode or not created_here):
        typer.echo(
            "[colab-mcp] --rm ignored: only runtimes auto-created by `colab-mcp ssh` are removed on exit.",
            err=True,
        )

    effective_mode = "cmd" if is_cmd_mode else "interactive"
    proxy_cmd = build_proxy_command(
        runtime_id=ep,
        identity=identity,
        auth_user_index=eff_user,
        mode=effective_mode,
        rm=False,
    )
    force_tty = (sys.stdin.isatty() and sys.stdout.isatty()) if is_cmd_mode else True
    try:
        ssh_args = build_ssh_subprocess_args(
            proxy_cmd,
            identity=identity,
            remote_command=cmd if is_cmd_mode else None,
            force_tty=force_tty,
        )
    except Exception as exc:
        emit_error(exc, as_json=False)

    ensure_cli_stdout()
    try:
        if force:
            _run_with_json(_kill_stale_sshd_standalone(ep, eff_user), as_json=False)
        rc = subprocess.call(ssh_args)
    finally:
        if created_here and rm and not is_cmd_mode:
            try:
                asyncio.run(_disconnect_ssh_runtime(ep, eff_user))
            except Exception:
                pass
    raise typer.Exit(code=int(rc))


async def _kill_stale_sshd_standalone(runtime_id: str, auth_user_index: int) -> None:
    try:
        await ensure_cli_runtime_attached(
            runtime_id=runtime_id,
            auth_user_index=auth_user_index,
        )
        await _kill_stale_sshd(runtime_id)
    finally:
        await cleanup_cli_runtime_resources()


if __name__ == "__main__":
    app()
