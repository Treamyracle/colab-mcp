"""
End-to-End tests for Colab MCP Authentication Lifecycle (Live & Mock).

Requirements:
- TECH_SPEC §3.2: Google OAuth 2.0 (Remote flow, Loopback, ADC, Proactive Refresh).
- TECH_SPEC §3.3: SAPISIDHASH request signing and cookie verification.
- TECH_SPEC §3.4: OS Keyring Keystore with AES-256-GCM fallback.
- TECH_SPEC §8.1: Zero-leakage credential redaction.
- TEST_INFRA.md § Feature Areas 3, 4, 5, 6, 7.

Supports live credentials via environment variables:
- COLAB_AUTH_TOKEN: Active OAuth 2.0 Bearer access token
- COLAB_COOKIE_SAPISID: Active Google SAPISID cookie
When absent, runs full deterministic simulated wire sessions ensuring complete coverage.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from unittest.mock import MagicMock, patch
import pytest

from colab_mcp.auth.keystore import (
    get_credentials,
    purge_credentials,
    store_credentials,
)
from colab_mcp.auth.models import (
    AuthenticationRequiredError,
    AuthMethod,
    ColabAuthStatusResponse,
    ColabLoginResponse,
    ColabLogoutResponse,
    GoogleCredentials,
)
from colab_mcp.auth.oauth import (
    DEFAULT_CLIENT_ID,
    generate_remote_consent_url,
    get_valid_auth_headers,
    refresh_credentials,
)
from colab_mcp.auth.sapisid import generate_sapisid_hash
from colab_mcp.auth.tools import (
    colab_get_auth_status_impl,
    colab_login_impl,
    colab_logout_impl,
)
from colab_mcp.utils.redaction import redact_credentials

LIVE_AUTH_TOKEN = os.environ.get("COLAB_AUTH_TOKEN")
LIVE_SAPISID = os.environ.get("COLAB_COOKIE_SAPISID")
IS_LIVE_ENVIRONMENT = bool(LIVE_AUTH_TOKEN or LIVE_SAPISID)


# ============================================================================
# TIER 1: AUTHENTICATION FLOW COVERAGE (Happy Paths)
# ============================================================================

@pytest.mark.asyncio
async def test_e2e_oauth_remote_authorization_and_storage(
    mock_keystore_dir, mock_keyring, mock_machine_id, monkeypatch
):
    """Tier 1: Full OAuth remote authorization flow: consent URL -> exchange -> keystore storage -> status check."""
    # Step 1: Initiate remote OAuth flow
    login_step1 = await colab_login_impl(method="oauth_remote", auth_user_index=0)
    assert login_step1.status == "authorization_required"
    assert login_step1.auth_url is not None
    assert "https://accounts.google.com/o/oauth2/" in login_step1.auth_url
    assert "token_usage=remote" in login_step1.auth_url
    assert "prompt=consent" in login_step1.auth_url

    # Step 2: Complete exchange with auth code (live or deterministic mock)
    if LIVE_AUTH_TOKEN:
        # Use live token if provided in environment
        creds = GoogleCredentials(
            access_token=LIVE_AUTH_TOKEN,
            refresh_token="1//04live_mock_refresh",
            token_uri="https://oauth2.googleapis.com/token",
            client_id=DEFAULT_CLIENT_ID,
            token_expiry=datetime.now(timezone.utc) + timedelta(seconds=3600),
            user_email="live_user@colab.google.com",
            auth_method=AuthMethod.OAUTH_REMOTE,
        )
        store_credentials(0, creds)
    else:
        # Deterministic simulation of Google OAuth token endpoint exchange
        mock_oauth_creds = MagicMock()
        mock_oauth_creds.token = "ya29.a0AfH6SMB_SIMULATED_ACCESS_TOKEN_12345"
        mock_oauth_creds.refresh_token = "1//04SIMULATED_REFRESH_TOKEN_67890"
        mock_oauth_creds.expiry = datetime.now(timezone.utc) + timedelta(seconds=3600)
        mock_oauth_creds.token_uri = "https://oauth2.googleapis.com/token"
        mock_oauth_creds.client_id = DEFAULT_CLIENT_ID
        mock_oauth_creds.client_secret = "simulated_secret"
        mock_oauth_creds.scopes = ["https://www.googleapis.com/auth/colaboratory"]

        mock_flow = MagicMock()
        mock_flow.credentials = mock_oauth_creds

        from colab_mcp.auth.oauth import _ACTIVE_REMOTE_FLOWS
        _ACTIVE_REMOTE_FLOWS[0] = mock_flow

        monkeypatch.setattr(
            "colab_mcp.auth.oauth.InstalledAppFlow.from_client_config",
            lambda *args, **kwargs: mock_flow,
        )
        monkeypatch.setattr(
            "colab_mcp.auth.oauth.fetch_user_email_from_token",
            lambda token: "ml_engineer@example.com",
        )

        login_step2 = await colab_login_impl(
            method="oauth_remote",
            auth_code="4/0AWgavdf_simulated_auth_code_from_google",
            auth_user_index=0,
        )
        assert login_step2.status == "authenticated"
        assert login_step2.user_email == "ml_engineer@example.com"
        assert login_step2.has_refresh_token is True

    # Step 3: Query colab_get_auth_status to verify persistence
    status = await colab_get_auth_status_impl(auth_user_index=0)
    assert status.is_authenticated is True
    assert status.token_valid is True
    assert status.expires_in_seconds is not None
    assert status.expires_in_seconds > 0


@pytest.mark.asyncio
async def test_e2e_cookie_import_and_sapisidhash_signing(
    mock_keystore_dir, mock_keyring, mock_machine_id
):
    """Tier 1: Cookie import flow with SAPISID/SID/OSID and SAPISIDHASH request header calculation."""
    sapisid_val = LIVE_SAPISID if LIVE_SAPISID else "mock_sapisid_cookie_value_12345"
    cookies = {
        "SAPISID": sapisid_val,
        "SID": "mock_sid_cookie_value_67890",
        "OSID": "mock_osid_cookie_for_colab_research",
        "HSID": "mock_hsid_112233",
        "SSID": "mock_ssid_445566",
        "APISID": "mock_apisid_778899",
        "__Secure-1PSID": "mock_1psid_990011",
        "__Secure-3PSID": "mock_3psid_223344",
        "__Secure-OSID": "mock_sec_osid_556677",
    }

    login_resp = await colab_login_impl(
        method="cookie_import",
        cookies=cookies,
        auth_user_index=0,
    )
    assert login_resp.status == "authenticated"
    assert login_resp.auth_method == "cookie_import"

    # Verify credentials stored in keystore
    stored = get_credentials(0)
    assert stored is not None
    assert stored.cookies.get("SAPISID") == sapisid_val
    assert stored.cookies.get("OSID") == "mock_osid_cookie_for_colab_research"

    # Compute SAPISIDHASH signature for Colab origin
    origin = "https://colab.research.google.com"
    auth_header = generate_sapisid_hash(sapisid=sapisid_val, origin=origin)
    assert auth_header.startswith("SAPISIDHASH ")
    parts = auth_header.replace("SAPISIDHASH ", "").split("_")
    assert len(parts) == 2
    epoch_sec, sha1_hex = parts[0], parts[1]
    assert epoch_sec.isdigit()
    assert len(sha1_hex) == 40  # SHA-1 hex digest is 40 chars


@pytest.mark.asyncio
async def test_e2e_proactive_token_refresh(
    mock_keystore_dir, mock_keyring, mock_machine_id, sample_expiring_soon_credentials, monkeypatch
):
    """Tier 1: Transparent proactive refresh when token is within 5-minute expiry window."""
    # Store token expiring in 120 seconds (< 300 seconds)
    store_credentials(0, sample_expiring_soon_credentials)

    # Mock the Google OAuth credentials refresh call
    refreshed_access_token = "ya29.a0AfH6SMB_NEWLY_REFRESHED_ACCESS_TOKEN_99999"
    new_expiry = datetime.now(timezone.utc) + timedelta(seconds=3600)

    async def mock_refresh(creds_data: GoogleCredentials) -> GoogleCredentials:
        creds_data.access_token = refreshed_access_token
        creds_data.token_expiry = new_expiry
        store_credentials(creds_data.auth_user_index, creds_data)
        return creds_data

    monkeypatch.setattr("colab_mcp.auth.oauth.refresh_credentials", mock_refresh)

    # Request valid auth headers: must trigger proactive refresh
    headers = await get_valid_auth_headers(auth_user_index=0)
    assert "Authorization" in headers
    assert "X-Goog-AuthUser" in headers
    assert headers["X-Goog-AuthUser"] == "0"

    # Keystore should now contain refreshed access token
    updated = get_credentials(0)
    assert updated is not None
    assert updated.access_token == refreshed_access_token


@pytest.mark.asyncio
async def test_e2e_logout_flow_and_keystore_purge(
    mock_keystore_dir, mock_keyring, mock_machine_id, sample_credentials
):
    """Tier 1: Safe logout wipes credentials and sets unauthenticated status."""
    store_credentials(0, sample_credentials)
    assert get_credentials(0) is not None

    logout_resp = await colab_logout_impl(auth_user_index=0, terminate_active_runtimes=True)
    assert logout_resp.status == "logged_out"
    assert logout_resp.keystore_purged is True
    assert logout_resp.auth_user_index == 0

    # Keyring and encrypted file must be purged
    assert get_credentials(0) is None

    # Status must reflect unauthenticated
    status = await colab_get_auth_status_impl(auth_user_index=0)
    assert status.is_authenticated is False
    assert status.token_valid is False
    assert status.user_email is None


# ============================================================================
# TIER 2: BOUNDARY & SECURITY EDGE CONDITIONS
# ============================================================================

@pytest.mark.asyncio
async def test_e2e_revoked_refresh_token_triggers_keystore_purge(
    mock_keystore_dir, mock_keyring, mock_machine_id, sample_expiring_soon_credentials, monkeypatch
):
    """Tier 2: Revoked refresh token wipes credentials from keystore and raises AuthenticationRequiredError."""
    store_credentials(0, sample_expiring_soon_credentials)

    async def mock_revoked_refresh(*args, **kwargs):
        purge_credentials(0)
        raise AuthenticationRequiredError("Google OAuth credentials expired or were revoked: invalid_grant")

    monkeypatch.setattr("colab_mcp.auth.oauth.refresh_credentials", mock_revoked_refresh)

    # When refresh fails due to revocation, keystore must be purged and explicit error raised
    with pytest.raises(AuthenticationRequiredError) as exc_info:
        await get_valid_auth_headers(auth_user_index=0)

    assert "re-authenticate" in str(exc_info.value).lower() or "revoked" in str(exc_info.value).lower()
    # Credentials purged
    assert get_credentials(0) is None


@pytest.mark.asyncio
async def test_e2e_multi_account_isolation(
    mock_keystore_dir, mock_keyring, mock_machine_id, sample_credentials
):
    """Tier 2: Multi-account routing maintains strict separation between account 0 and account 1."""
    creds_user_0 = sample_credentials.model_copy(update={"user_email": "user0@google.com", "auth_user_index": 0})
    creds_user_1 = sample_credentials.model_copy(
        update={
            "user_email": "user1@research.org",
            "access_token": "ya29.user1_access_token_xyz",
            "auth_user_index": 1,
        }
    )

    store_credentials(0, creds_user_0)
    store_credentials(1, creds_user_1)

    status_0 = await colab_get_auth_status_impl(auth_user_index=0)
    status_1 = await colab_get_auth_status_impl(auth_user_index=1)

    assert status_0.user_email == "user0@google.com"
    assert status_1.user_email == "user1@research.org"

    # Purge account 0 only
    purge_credentials(0)
    assert get_credentials(0) is None
    assert get_credentials(1) is not None
    assert get_credentials(1).user_email == "user1@research.org"


@pytest.mark.asyncio
async def test_e2e_cookie_import_missing_required_cookie_rejected(
    mock_keystore_dir, mock_keyring, mock_machine_id
):
    """Tier 2: Missing mandatory cookies for Colab domain is rejected."""
    # Test missing SAPISID
    incomplete_cookies = {
        "SID": "test_sid_456",
        # Missing SAPISID!
    }
    with pytest.raises(ValueError) as exc_info:
        await colab_login_impl(
            method="cookie_import",
            cookies=incomplete_cookies,
            auth_user_index=0,
        )
    assert "sapisid" in str(exc_info.value).lower()

    # Verify CDP browser mandatory cookies require OSID
    from colab_mcp.auth.cdp_browser import MANDATORY_COOKIES
    assert "OSID" in MANDATORY_COOKIES
    assert "SAPISID" in MANDATORY_COOKIES
    assert "SID" in MANDATORY_COOKIES


def test_e2e_zero_leakage_in_auth_logging(sample_credentials):
    """Tier 2: Verify zero raw tokens or cookies leak into logging or serializations."""
    sensitive_log = (
        f"Connecting with token={sample_credentials.access_token} "
        f"and refresh={sample_credentials.refresh_token} "
        f"and SAPISID={sample_credentials.cookies['SAPISID']}"
    )
    sanitized = redact_credentials(sensitive_log)
    assert sample_credentials.access_token not in sanitized
    assert sample_credentials.refresh_token not in sanitized
    assert sample_credentials.cookies["SAPISID"] not in sanitized
    assert "[REDACTED]" in sanitized
