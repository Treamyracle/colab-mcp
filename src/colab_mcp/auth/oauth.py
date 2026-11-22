"""
Google OAuth 2.0 authentication flows, token lifecycle, and header generation.
"""

import asyncio
import json
import os
import stat
import sys
import threading
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import parse_qs, urlparse

import google.auth
import google.auth.transport.requests
import google.oauth2.credentials
import httpx
from google.auth.exceptions import RefreshError
from google_auth_oauthlib.flow import InstalledAppFlow

from colab_mcp.auth.models import (
    AuthenticationRequiredError,
    AuthMethod,
    ColabAuthError,
    GoogleCredentials,
    TokenRefreshError,
)
from colab_mcp.auth.sapisid import get_sapisid_auth_headers
from colab_mcp.utils.logging import get_logger

logger = get_logger(__name__)

# Authoritative Google Cloud SDK Public Client ID and Secret
DEFAULT_CLIENT_ID = "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com"
DEFAULT_CLIENT_SECRET = "d-FL95Q19q7MQmFpd7hHD0Ty"
REMOTE_REDIRECT_URI = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"

COLAB_OAUTH_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/colaboratory",
    "https://www.googleapis.com/auth/drive.file",
]
COLAB_CUSTOM_DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"

# In-memory registry to preserve InstalledAppFlow instances between remote URL generation and code exchange
_ACTIVE_REMOTE_FLOWS: Dict[int, InstalledAppFlow] = {}

_SCOPE_RELAX_LOCK = threading.Lock()
_SCOPE_RELAX_DEPTH = 0
_SCOPE_RELAX_PREV: Optional[str] = None


@contextmanager
def _relax_oauthlib_token_scope() -> Iterator[None]:
    """
    Temporarily set OAUTHLIB_RELAX_TOKEN_SCOPE=1 during OAuth code exchange.
    Google incremental/unbundled consent (and ColabTwin) may return a subset of
    requested scopes, which is valid per RFC 6749 §3.3/§5.1.
    """
    global _SCOPE_RELAX_DEPTH, _SCOPE_RELAX_PREV
    with _SCOPE_RELAX_LOCK:
        if _SCOPE_RELAX_DEPTH == 0:
            _SCOPE_RELAX_PREV = os.environ.get("OAUTHLIB_RELAX_TOKEN_SCOPE")
            os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
        _SCOPE_RELAX_DEPTH += 1
    try:
        yield
    finally:
        with _SCOPE_RELAX_LOCK:
            _SCOPE_RELAX_DEPTH -= 1
            if _SCOPE_RELAX_DEPTH == 0:
                if _SCOPE_RELAX_PREV is None:
                    os.environ.pop("OAUTHLIB_RELAX_TOKEN_SCOPE", None)
                else:
                    os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = _SCOPE_RELAX_PREV
                _SCOPE_RELAX_PREV = None


def get_oauth_client_config() -> Dict[str, Any]:
    """
    Returns the OAuth 2.0 client configuration dictionary.
    Checks ~/.colabmcp/config.json for custom client credentials; otherwise uses defaults.
    """
    config_file = os.path.expanduser("~/.colabmcp/config.json")
    if os.path.isfile(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "installed" in data:
                    return data
                if "client_id" in data and "client_secret" in data:
                    return {
                        "installed": {
                            "client_id": data["client_id"],
                            "client_secret": data["client_secret"],
                            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                            "token_uri": "https://oauth2.googleapis.com/token",
                            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                            "redirect_uris": [
                                REMOTE_REDIRECT_URI,
                                "http://localhost",
                                "http://127.0.0.1",
                            ],
                        }
                    }
        except Exception as e:
            logger.warning("Failed to parse custom config %s: %s. Using default client ID.", config_file, e)

    return {
        "installed": {
            "client_id": DEFAULT_CLIENT_ID,
            "client_secret": DEFAULT_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "redirect_uris": [
                REMOTE_REDIRECT_URI,
                "http://localhost",
                "http://127.0.0.1",
            ],
        }
    }


def fetch_user_email_from_token(access_token: str) -> Optional[str]:
    """Queries Google userinfo endpoint to resolve account email."""
    if not access_token:
        return None
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("email")
    except Exception as e:
        logger.debug("Failed to resolve user email from userinfo endpoint: %s", e)
    return None


def _resolve_oauth_scopes(client_config: Dict[str, Any]) -> List[str]:
    """
    Resolve OAuth scopes cleanly: use Google's verified public client scopes for
    ``DEFAULT_CLIENT_ID`` (avoiding restricted ``auth/drive`` blocks on consumer
    accounts), while honoring custom ``scopes`` or allowing full Drive scope for
    user-supplied OAuth client configs.
    """
    explicit_scopes = client_config.get("scopes")
    if isinstance(explicit_scopes, list) and explicit_scopes:
        return [str(s) for s in explicit_scopes]

    installed = client_config.get("installed") if isinstance(client_config, dict) else None
    client_id = installed.get("client_id") if isinstance(installed, dict) else None
    if isinstance(client_id, str) and client_id and client_id != DEFAULT_CLIENT_ID:
        return [*COLAB_OAUTH_SCOPES, COLAB_CUSTOM_DRIVE_SCOPE]
    return list(COLAB_OAUTH_SCOPES)


def _normalize_auth_code(raw_code: str) -> str:
    """Accept either a bare authorization code or a full pasted redirect URL."""
    stripped = raw_code.strip()
    if stripped.startswith(("http://", "https://")) and "code=" in stripped:
        parsed = urlparse(stripped)
        codes = parse_qs(parsed.query).get("code")
        if codes and codes[0].strip():
            return codes[0].strip()
    return stripped


def _pending_pkce_file(auth_user_index: int) -> Path:
    from colab_mcp.auth import keystore

    return keystore.get_credentials_dir() / f"pending_oauth_{auth_user_index}.json"


def _save_pending_pkce(auth_user_index: int, code_verifier: Any) -> None:
    """Persist PKCE code_verifier across separate CLI invocations or MCP restarts."""
    if not isinstance(code_verifier, str) or not code_verifier.strip():
        return
    from colab_mcp.auth import keystore

    path = _pending_pkce_file(auth_user_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    payload = json.dumps({"code_verifier": code_verifier})
    if sys.platform != "win32":
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
        with open(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
    else:
        tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)
    keystore.enforce_file_permissions(path)


def _pop_pending_pkce(auth_user_index: int) -> Optional[str]:
    """Consume persisted PKCE code_verifier if present on disk."""
    path = _pending_pkce_file(auth_user_index)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        verifier = data.get("code_verifier")
        return str(verifier) if isinstance(verifier, str) and verifier.strip() else None
    except Exception:
        return None
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Flow 1: Remote Out-of-band flow
# ---------------------------------------------------------------------------

def generate_remote_consent_url(auth_user_index: int = 0) -> str:
    """
    Creates an InstalledAppFlow configured for remote copy-paste landing page.
    Caches the flow instance (and persists its PKCE verifier on disk) for
    subsequent code exchange across CLI or MCP processes.
    """
    client_config = get_oauth_client_config()
    scopes = _resolve_oauth_scopes(client_config)
    flow = InstalledAppFlow.from_client_config(
        client_config,
        scopes=scopes,
        redirect_uri=REMOTE_REDIRECT_URI,
    )
    auth_url, _ = flow.authorization_url(
        prompt="consent",
        token_usage="remote",
        access_type="offline",
    )
    _ACTIVE_REMOTE_FLOWS[auth_user_index] = flow
    _save_pending_pkce(auth_user_index, getattr(flow, "code_verifier", None))
    return auth_url


initiate_remote_oauth_flow = generate_remote_consent_url


async def exchange_remote_auth_code(auth_code: str, auth_user_index: int = 0) -> GoogleCredentials:
    """
    Exchanges the authorization code obtained from the remote landing page for tokens.
    """
    if not auth_code or not auth_code.strip():
        raise ColabAuthError("Authorization code cannot be empty.")

    cleaned_code = _normalize_auth_code(auth_code)
    flow = _ACTIVE_REMOTE_FLOWS.pop(auth_user_index, None)
    saved_verifier = _pop_pending_pkce(auth_user_index)
    client_config = get_oauth_client_config()
    scopes = _resolve_oauth_scopes(client_config)
    if flow is None:
        flow = InstalledAppFlow.from_client_config(
            client_config,
            scopes=scopes,
            redirect_uri=REMOTE_REDIRECT_URI,
        )
        if saved_verifier:
            flow.code_verifier = saved_verifier

    try:
        with _relax_oauthlib_token_scope():
            await asyncio.to_thread(flow.fetch_token, code=cleaned_code)
    except Exception as e:
        raise ColabAuthError(f"OAuth code exchange failed: {e}") from e

    creds = flow.credentials
    user_email = await asyncio.to_thread(fetch_user_email_from_token, creds.token)

    return GoogleCredentials(
        auth_method=AuthMethod.OAUTH_REMOTE,
        auth_user_index=auth_user_index,
        user_email=user_email,
        access_token=creds.token,
        refresh_token=creds.refresh_token,
        token_expiry=creds.expiry,
        token_uri=creds.token_uri,
        client_id=creds.client_id,
        client_secret=creds.client_secret,
        scopes=list(creds.scopes) if creds.scopes else scopes,
    )


# ---------------------------------------------------------------------------
# Flow 2: Local loopback server flow
# ---------------------------------------------------------------------------

async def run_loopback_flow(auth_user_index: int = 0, timeout_seconds: int = 180) -> GoogleCredentials:
    """
    Executes interactive loopback OAuth flow on 127.0.0.1:0.
    Opens system browser and blocks until callback arrives or timeout.
    """
    client_config = get_oauth_client_config()
    scopes = _resolve_oauth_scopes(client_config)
    flow = InstalledAppFlow.from_client_config(
        client_config,
        scopes=scopes,
    )

    def _run_local_server() -> Any:
        with _relax_oauthlib_token_scope():
            return flow.run_local_server(
                host="127.0.0.1",
                port=0,
                authorization_prompt_message=None,
                success_message="Authentication complete! You may close this window and return to colab-mcp.",
                open_browser=True,
                timeout_seconds=timeout_seconds,
                prompt="consent",
                access_type="offline",
            )

    try:
        creds = await asyncio.to_thread(_run_local_server)
    except Exception as e:
        raise ColabAuthError(f"Local loopback OAuth flow failed: {e}") from e

    user_email = await asyncio.to_thread(fetch_user_email_from_token, creds.token)

    return GoogleCredentials(
        auth_method=AuthMethod.OAUTH_LOOPBACK,
        auth_user_index=auth_user_index,
        user_email=user_email,
        access_token=creds.token,
        refresh_token=creds.refresh_token,
        token_expiry=creds.expiry,
        token_uri=creds.token_uri,
        client_id=creds.client_id,
        client_secret=creds.client_secret,
        scopes=list(creds.scopes) if creds.scopes else scopes,
    )


run_loopback_oauth_flow = run_loopback_flow


# ---------------------------------------------------------------------------
# Flow 3: Ambient ADC loader
# ---------------------------------------------------------------------------

def ingest_adc_credentials(auth_user_index: int = 0) -> GoogleCredentials:
    """
    Discovers ambient Application Default Credentials (e.g. from gcloud CLI).
    """
    try:
        from unittest.mock import patch as _patch
        from google.auth import _cloud_sdk

        with (
            warnings.catch_warnings(),
            _patch.object(_cloud_sdk, "get_project_id", return_value=None),
        ):
            warnings.filterwarnings(
                "ignore",
                message=r"Your application has authenticated using end user credentials.*",
                category=UserWarning,
            )
            creds, _ = google.auth.default(scopes=COLAB_OAUTH_SCOPES)

        request = google.auth.transport.requests.Request()
        if not creds.valid:
            creds.refresh(request)

        user_email = fetch_user_email_from_token(creds.token)
        refresh_token = getattr(creds, "refresh_token", None)
        raw_client_id = getattr(creds, "client_id", None)
        client_id = raw_client_id if isinstance(raw_client_id, str) and raw_client_id else DEFAULT_CLIENT_ID

        raw_client_secret = getattr(creds, "client_secret", None)
        client_secret = raw_client_secret if isinstance(raw_client_secret, str) and raw_client_secret else DEFAULT_CLIENT_SECRET

        return GoogleCredentials(
            auth_method=AuthMethod.OAUTH_ADC,
            auth_user_index=auth_user_index,
            user_email=user_email,
            access_token=creds.token,
            refresh_token=refresh_token,
            token_expiry=getattr(creds, "expiry", None),
            token_uri=getattr(creds, "token_uri", "https://oauth2.googleapis.com/token"),
            client_id=client_id,
            client_secret=client_secret,
            scopes=COLAB_OAUTH_SCOPES,
            is_adc=True,
        )
    except Exception as e:
        logger.debug("Ambient ADC ingestion failed: %s", e)
        raise ColabAuthError(
            f"Ambient ADC ingestion failed: {e}. Run: "
            "gcloud auth application-default login "
            "--scopes=openid,https://www.googleapis.com/auth/cloud-platform,"
            "https://www.googleapis.com/auth/userinfo.email,"
            "https://www.googleapis.com/auth/colaboratory"
        ) from e


load_ambient_adc = ingest_adc_credentials


# ---------------------------------------------------------------------------
# Token Lifecycle & Proactive Refresh
# ---------------------------------------------------------------------------

async def refresh_credentials(creds: GoogleCredentials) -> GoogleCredentials:
    """
    Refreshes expired or expiring OAuth credentials.
    On RefreshError (revocation or invalid grant), purges keystore and raises AuthenticationRequiredError.
    """
    from colab_mcp.auth.keystore import purge_credentials, store_credentials

    if not creds.refresh_token:
        raise AuthenticationRequiredError(
            "Access token expired and no refresh token is available. Please re-authenticate."
        )

    google_creds = google.oauth2.credentials.Credentials(
        token=creds.access_token,
        refresh_token=creds.refresh_token,
        token_uri=creds.token_uri,
        client_id=creds.client_id or DEFAULT_CLIENT_ID,
        client_secret=creds.client_secret or DEFAULT_CLIENT_SECRET,
        scopes=creds.scopes or COLAB_OAUTH_SCOPES,
    )

    request = google.auth.transport.requests.Request()
    try:
        await asyncio.to_thread(google_creds.refresh, request)
    except RefreshError as e:
        logger.error("OAuth token refresh failed (revoked or expired): %s. Purging keystore.", e)
        purge_credentials(creds.auth_user_index)
        raise AuthenticationRequiredError(
            f"Google OAuth credentials expired or were revoked: {e}. "
            "Please re-authenticate using colab_login."
        ) from e
    except Exception as e:
        raise TokenRefreshError(f"Transient error refreshing OAuth token: {e}") from e

    # Update creds object with refreshed values
    creds.access_token = google_creds.token
    if google_creds.expiry:
        creds.token_expiry = (
            google_creds.expiry.replace(tzinfo=timezone.utc)
            if google_creds.expiry.tzinfo is None
            else google_creds.expiry
        )
    else:
        creds.token_expiry = None
    if not creds.user_email:
        creds.user_email = await asyncio.to_thread(fetch_user_email_from_token, creds.access_token)

    # Persist updated credentials back into keystore
    store_credentials(creds.auth_user_index, creds)
    return creds


async def get_valid_auth_headers(auth_user_index: int = 0) -> Dict[str, str]:
    """
    Retrieves stored credentials for auth_user_index, proactively refreshes
    if within 5 minutes of expiration, and returns valid HTTP headers.
    """
    from colab_mcp.auth.keystore import get_credentials

    creds = get_credentials(auth_user_index)
    if creds is None:
        raise AuthenticationRequiredError(
            f"No credentials found for auth_user_index={auth_user_index}. "
            "Please authenticate using colab_login."
        )

    if creds.is_cookie_based and not creds.is_oauth:
        # Google 1P Cookie-based request signing
        return get_sapisid_auth_headers(
            cookies=creds.cookies,
            auth_user_index=auth_user_index,
        )

    # OAuth token-based path
    if creds.is_expired(buffer_seconds=300):
        logger.info("Credentials for user %d expiring within 5m or expired. Proactively refreshing...", auth_user_index)
        creds = await refresh_credentials(creds)

    headers = {
        "Authorization": f"Bearer {creds.access_token}",
        "X-Goog-AuthUser": str(auth_user_index),
        "X-Colab-Client-Agent": "colab-mcp",
    }
    return headers
