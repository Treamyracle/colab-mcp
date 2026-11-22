"""
MCP Tool implementations for Colab Authentication.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from mcp.server.mcpserver import MCPServer

from colab_mcp.auth import cdp_browser, keystore, oauth
from colab_mcp.auth.secrets import (
    DeleteSecretResponse,
    ListSecretsResponse,
    SetSecretResponse,
    delete_secret,
    list_secrets,
    set_secret,
)
from colab_mcp.auth.models import (
    AuthenticationRequiredError,
    AuthMethod,
    ColabAuthStatusResponse,
    ColabLoginResponse,
    ColabLogoutResponse,
    GoogleCredentials,
)
from colab_mcp.utils.logging import get_logger

logger = get_logger(__name__)


async def colab_login_impl(
    method: str = "oauth_remote",
    auth_user_index: int = 0,
    auth_code: Optional[str] = None,
    cookies: Optional[Dict[str, str]] = None,
    timeout_seconds: int = 180,
) -> ColabLoginResponse:
    """Core logic for colab_login tool."""
    if auth_user_index < 0:
        raise ValueError("auth_user_index must be >= 0")
    if not (30 <= timeout_seconds <= 600):
        raise ValueError("timeout_seconds must be between 30 and 600")

    valid_methods = [m.value for m in AuthMethod]
    if method not in valid_methods:
        raise ValueError(f"Unsupported authentication method: '{method}'. Valid options: {valid_methods}")

    if method == AuthMethod.OAUTH_REMOTE.value:
        if not auth_code:
            consent_url = oauth.generate_remote_consent_url(auth_user_index=auth_user_index)
            return ColabLoginResponse(
                status="authorization_required",
                auth_method=method,
                auth_user_index=auth_user_index,
                user_email=None,
                expires_at=None,
                has_refresh_token=False,
                auth_url=consent_url,
                message=(
                    "Authorization required. Please open the URL in 'auth_url' in your browser, "
                    "authorize Colab MCP, and call colab_login again with method='oauth_remote' "
                    "and auth_code='<CODE>'."
                ),
            )
        else:
            creds = await oauth.exchange_remote_auth_code(auth_code=auth_code, auth_user_index=auth_user_index)
            keystore.store_credentials(auth_user_index, creds)
            return ColabLoginResponse(
                status="authenticated",
                auth_method=method,
                auth_user_index=auth_user_index,
                user_email=creds.user_email,
                expires_at=creds.expiry.isoformat() if creds.expiry else None,
                has_refresh_token=bool(creds.refresh_token),
                auth_url=None,
                message="Successfully authenticated with Google Colab via OAuth 2.0 remote flow.",
            )

    elif method == AuthMethod.OAUTH_LOOPBACK.value:
        creds = await oauth.run_loopback_flow(auth_user_index=auth_user_index, timeout_seconds=timeout_seconds)
        keystore.store_credentials(auth_user_index, creds)
        return ColabLoginResponse(
            status="authenticated",
            auth_method=method,
            auth_user_index=auth_user_index,
            user_email=creds.user_email,
            expires_at=creds.expiry.isoformat() if creds.expiry else None,
            has_refresh_token=bool(creds.refresh_token),
            auth_url=None,
            message="Successfully authenticated with Google Colab via OAuth 2.0 loopback flow.",
        )

    elif method == AuthMethod.OAUTH_ADC.value:
        creds = oauth.ingest_adc_credentials(auth_user_index=auth_user_index)
        keystore.store_credentials(auth_user_index, creds)
        return ColabLoginResponse(
            status="authenticated",
            auth_method=method,
            auth_user_index=auth_user_index,
            user_email=creds.user_email,
            expires_at=creds.expiry.isoformat() if creds.expiry else None,
            has_refresh_token=bool(creds.refresh_token),
            auth_url=None,
            message="Successfully authenticated via ambient Application Default Credentials (ADC).",
        )

    elif method == AuthMethod.COOKIE_IMPORT.value:
        if not cookies or "SAPISID" not in cookies or "SID" not in cookies:
            raise ValueError("Cookie import requires a dictionary containing at least 'SAPISID' and 'SID'.")
        creds = GoogleCredentials(
            cookies=cookies,
            auth_method=AuthMethod.COOKIE_IMPORT,
            auth_user_index=auth_user_index,
        )
        keystore.store_credentials(auth_user_index, creds)
        return ColabLoginResponse(
            status="authenticated",
            auth_method=method,
            auth_user_index=auth_user_index,
            user_email=None,
            expires_at=None,
            has_refresh_token=False,
            auth_url=None,
            message="Successfully authenticated via session cookie import.",
        )

    elif method == AuthMethod.CDP_BROWSER.value:
        creds = await cdp_browser.launch_and_extract_cdp_cookies(
            auth_user_index=auth_user_index,
            timeout_seconds=timeout_seconds,
        )
        keystore.store_credentials(auth_user_index, creds)
        return ColabLoginResponse(
            status="authenticated",
            auth_method=method,
            auth_user_index=auth_user_index,
            user_email=creds.user_email,
            expires_at=None,
            has_refresh_token=False,
            auth_url=None,
            message="Successfully authenticated via CDP browser session capture.",
        )

    raise ValueError(f"Unhandled method: {method}")


async def colab_get_auth_status_impl(
    auth_user_index: int = 0,
    force_refresh: bool = False,
) -> ColabAuthStatusResponse:
    """Core logic for colab_get_auth_status tool."""
    if auth_user_index < 0:
        raise ValueError("auth_user_index must be >= 0")

    creds = keystore.get_credentials(auth_user_index)
    if not creds:
        return ColabAuthStatusResponse(
            is_authenticated=False,
            auth_method=None,
            auth_user_index=auth_user_index,
            user_email=None,
            token_valid=False,
            expires_in_seconds=None,
            account_tier=None,
            compute_units_remaining=None,
            active_sessions_count=0,
        )

    expires_in: Optional[int] = None
    token_valid = True

    if creds.expiry:
        now = datetime.now(timezone.utc)
        exp = creds.expiry if creds.expiry.tzinfo else creds.expiry.replace(tzinfo=timezone.utc)
        expires_in = int((exp - now).total_seconds())
        if expires_in <= 0:
            token_valid = False

    # Proactive refresh if expiring within 300s or force_refresh requested
    if creds.refresh_token and ((expires_in is not None and expires_in < 300) or force_refresh or not token_valid):
        try:
            creds = await oauth.refresh_credentials(creds)
            keystore.store_credentials(auth_user_index, creds)
            token_valid = True
            if creds.expiry:
                now = datetime.now(timezone.utc)
                exp = creds.expiry if creds.expiry.tzinfo else creds.expiry.replace(tzinfo=timezone.utc)
                expires_in = max(0, int((exp - now).total_seconds()))
        except AuthenticationRequiredError as exc:
            logger.warning("Authentication revoked or expired: %s; purging keystore for user %d", exc, auth_user_index)
            keystore.purge_credentials(auth_user_index)
            return ColabAuthStatusResponse(
                is_authenticated=False,
                auth_method=creds.auth_method.value if creds.auth_method else None,
                auth_user_index=auth_user_index,
                user_email=creds.user_email,
                token_valid=False,
                expires_in_seconds=0,
                account_tier=None,
                compute_units_remaining=None,
                active_sessions_count=0,
            )
        except Exception as exc:
            logger.warning("Transient error during token refresh: %s (preserving credentials)", exc)
            still_valid = expires_in is not None and expires_in > 0 and not force_refresh
            return ColabAuthStatusResponse(
                is_authenticated=still_valid,
                auth_method=creds.auth_method.value if creds.auth_method else None,
                auth_user_index=auth_user_index,
                user_email=creds.user_email,
                token_valid=still_valid,
                expires_in_seconds=max(0, expires_in) if expires_in is not None else None,
                account_tier=None,
                compute_units_remaining=None,
                active_sessions_count=0,
            )

    account_tier: Optional[str] = None
    compute_units_remaining: Optional[float] = None
    active_sessions_count: int = 0

    try:
        from colab_mcp.runtime.session_manager import get_session_manager

        mgr = get_session_manager()
        active_sessions_count = len(mgr.list_sessions(auth_user_index=auth_user_index))
    except Exception:
        pass

    if token_valid:
        try:
            import httpx
            from colab_mcp.runtime.client import ColabTFEClient

            async with ColabTFEClient(auth_user_index=auth_user_index, timeout=3.0) as tfe_client:
                ccu = await tfe_client.get_ccu_info()
                account_tier = ccu.user_tier
                compute_units_remaining = ccu.compute_units_remaining
                if ccu.active_assignments_count > active_sessions_count:
                    active_sessions_count = ccu.active_assignments_count
        except AuthenticationRequiredError:
            if force_refresh:
                token_valid = False
        except httpx.HTTPStatusError as exc:
            if force_refresh and exc.response is not None and exc.response.status_code == 401:
                token_valid = False
        except Exception:
            pass

    return ColabAuthStatusResponse(
        is_authenticated=token_valid,
        auth_method=creds.auth_method.value if creds.auth_method else None,
        auth_user_index=auth_user_index,
        user_email=creds.user_email,
        token_valid=token_valid,
        expires_in_seconds=max(0, expires_in) if expires_in is not None else None,
        account_tier=account_tier,
        compute_units_remaining=compute_units_remaining,
        active_sessions_count=active_sessions_count,
    )


async def colab_logout_impl(
    auth_user_index: int = 0,
    terminate_active_runtimes: bool = True,
) -> ColabLogoutResponse:
    """Core logic for colab_logout tool."""
    if auth_user_index < 0:
        raise ValueError("auth_user_index must be >= 0")

    terminated_count = 0
    purged = False
    try:
        if terminate_active_runtimes:
            try:
                from colab_mcp.runtime.session_manager import get_session_manager
                mgr = get_session_manager()
                terminated_count = await mgr.terminate_all_sessions(auth_user_index=auth_user_index)
            except (ImportError, AttributeError):
                terminated_count = 0
    finally:
        purged = keystore.purge_credentials(auth_user_index)

    return ColabLogoutResponse(
        status="logged_out",
        auth_user_index=auth_user_index,
        keystore_purged=purged,
        terminated_runtimes_count=terminated_count,
        message="Authentication credentials securely wiped and active runtimes deallocated.",
    )


async def _sync_secret_with_active_kernel(
    *,
    name: str,
    value: Optional[str] = None,
    removed: bool = False,
    runtime_id: Optional[str] = None,
) -> bool:
    """Best-effort injection or removal of a secret on any currently connected kernel."""
    try:
        from colab_mcp.kernel.client import KernelWebSocketClient
        from colab_mcp.kernel.tools import _KERNEL_CLIENTS
        from colab_mcp.runtime.session_manager import get_session_manager

        mgr = get_session_manager()
        target_id = runtime_id
        if target_id is None:
            session = mgr.get_session(None)
            target_id = session.endpoint if session is not None else None
        if target_id is None:
            return False
        client = _KERNEL_CLIENTS.get(target_id)
        if isinstance(client, KernelWebSocketClient) and client.is_connected:
            await client.bootstrap_colab_environment(
                secrets={name: value} if (not removed and value is not None) else None,
                removed_secrets=[name] if removed else None,
                force=True,
            )
            return True
    except Exception as exc:
        logger.debug("Skipping active kernel secret sync for %s: %s", name, exc)
    return False


async def colab_set_secret_impl(
    name: str,
    value: str,
    runtime_id: Optional[str] = None,
) -> SetSecretResponse:
    """Encrypt and store a named secret/token and push it to any connected kernel."""
    meta = set_secret(name=name, value=value)
    injected = await _sync_secret_with_active_kernel(
        name=meta.name,
        value=value,
        removed=False,
        runtime_id=runtime_id,
    )
    return SetSecretResponse(
        status="stored",
        name=meta.name,
        masked_preview=meta.masked_preview,
        length=meta.length,
        injected_into_active_kernel=injected,
        message=(
            f"Secret '{meta.name}' encrypted and stored"
            + (" (and injected into active Colab kernel)." if injected else ". Will auto-inject on kernel execution.")
        ),
    )


async def colab_list_secrets_impl() -> ListSecretsResponse:
    """List metadata and masked previews for stored secrets."""
    items = list_secrets()
    return ListSecretsResponse(count=len(items), secrets=items)


async def colab_delete_secret_impl(
    name: str,
    runtime_id: Optional[str] = None,
) -> DeleteSecretResponse:
    """Delete a stored secret and unset it on any connected kernel."""
    existed = delete_secret(name=name)
    clean_name = name.strip() if isinstance(name, str) else str(name)
    removed = await _sync_secret_with_active_kernel(
        name=clean_name,
        removed=True,
        runtime_id=runtime_id,
    )
    return DeleteSecretResponse(
        status="deleted",
        name=clean_name,
        existed=existed,
        removed_from_active_kernel=removed,
        message=(
            f"Secret '{clean_name}' removed from encrypted vault."
            if existed
            else f"Secret '{clean_name}' was not present in vault."
        ),
    )


def register_auth_tools(mcp: MCPServer) -> None:
    """Register all authentication tools on the MCPServer instance."""

    @mcp.tool(
        name="colab_login",
        description=(
            "Authenticate with Google Colab. Supports Google OAuth 2.0 (remote copy-paste "
            "authorization URL, local loopback browser listener, or ambient Application Default Credentials) "
            "and browser-assisted cookie extraction via Chrome DevTools Protocol (CDP)."
        ),
    )
    async def colab_login(
        method: str = "oauth_remote",
        auth_user_index: int = 0,
        auth_code: Optional[str] = None,
        cookies: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 180,
    ) -> ColabLoginResponse:
        return await colab_login_impl(
            method=method,
            auth_user_index=auth_user_index,
            auth_code=auth_code,
            cookies=cookies,
            timeout_seconds=timeout_seconds,
        )

    @mcp.tool(
        name="colab_get_auth_status",
        description=(
            "Query the current authentication status, active account identity, "
            "token/cookie expiration times, and available Colab Compute Units (CCU)."
        ),
    )
    async def colab_get_auth_status(
        auth_user_index: int = 0,
        force_refresh: bool = False,
    ) -> ColabAuthStatusResponse:
        return await colab_get_auth_status_impl(
            auth_user_index=auth_user_index,
            force_refresh=force_refresh,
        )

    @mcp.tool(
        name="colab_logout",
        description=(
            "Purge cached Google authentication credentials from OS keystore and local encrypted cache, "
            "and optionally terminate all active Colab cloud runtimes."
        ),
    )
    async def colab_logout(
        auth_user_index: int = 0,
        terminate_active_runtimes: bool = True,
    ) -> ColabLogoutResponse:
        return await colab_logout_impl(
            auth_user_index=auth_user_index,
            terminate_active_runtimes=terminate_active_runtimes,
        )

    @mcp.tool(
        name="colab_set_secret",
        description=(
            "Encrypt and store an API token or secret (e.g. HF_TOKEN, WANDB_API_KEY, OPENAI_API_KEY) "
            "in the OS keyring vault. Stored secrets are automatically redacted on egress and injected into "
            "os.environ and google.colab.userdata.get() on Colab kernels without browser popups."
        ),
    )
    async def colab_set_secret(
        name: str,
        value: str,
        runtime_id: Optional[str] = None,
    ) -> SetSecretResponse:
        return await colab_set_secret_impl(
            name=name,
            value=value,
            runtime_id=runtime_id,
        )

    @mcp.tool(
        name="colab_list_secrets",
        description=(
            "List names and masked previews of all API tokens/secrets stored in the colab-mcp vault "
            "(never returns plaintext values)."
        ),
    )
    async def colab_list_secrets() -> ListSecretsResponse:
        return await colab_list_secrets_impl()

    @mcp.tool(
        name="colab_delete_secret",
        description=(
            "Delete a named secret/token from the colab-mcp encrypted vault and unset it from any active kernel."
        ),
    )
    async def colab_delete_secret(
        name: str,
        runtime_id: Optional[str] = None,
    ) -> DeleteSecretResponse:
        return await colab_delete_secret_impl(
            name=name,
            runtime_id=runtime_id,
        )
