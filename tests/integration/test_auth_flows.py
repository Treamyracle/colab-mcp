"""
Integration tests for FastMCP authentication tools and end-to-end auth flows.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
import json
import pytest
import google.oauth2.credentials
from google.auth.exceptions import RefreshError
from google_auth_oauthlib.flow import InstalledAppFlow

from colab_mcp.auth.keystore import get_credentials, store_credentials
from colab_mcp.auth.models import (
    AuthMethod,
    ColabAuthStatusResponse,
    ColabLoginResponse,
    ColabLogoutResponse,
    GoogleCredentials,
)
from colab_mcp.auth.oauth import DEFAULT_CLIENT_ID
from colab_mcp.auth.tools import (
    colab_get_auth_status_impl,
    colab_login_impl,
    colab_logout_impl,
)


@pytest.mark.asyncio
async def test_fastmcp_colab_login_oauth_remote_step1(mock_keystore_dir, mock_keyring, mock_machine_id):
    resp = await colab_login_impl(method="oauth_remote", auth_user_index=0)
    assert resp.status == "authorization_required"
    assert resp.auth_method == "oauth_remote"
    assert resp.auth_url is not None
    assert "https://accounts.google.com/o/oauth2/" in resp.auth_url
    assert "prompt=consent" in resp.auth_url
    assert "token_usage=remote" in resp.auth_url


@pytest.mark.asyncio
async def test_fastmcp_colab_login_oauth_remote_step2(mock_keystore_dir, mock_keyring, mock_machine_id, monkeypatch):
    mock_creds = MagicMock()
    mock_creds.token = "ya29.remote_step2_token"
    mock_creds.refresh_token = "1//04remote_step2_refresh"
    mock_creds.expiry = datetime.now(timezone.utc) + timedelta(seconds=3600)
    mock_creds.token_uri = "https://oauth2.googleapis.com/token"
    mock_creds.client_id = DEFAULT_CLIENT_ID
    mock_creds.client_secret = "secret"
    mock_creds.scopes = ["https://www.googleapis.com/auth/colaboratory"]

    mock_flow = MagicMock(spec=InstalledAppFlow)
    mock_flow.credentials = mock_creds

    monkeypatch.setattr("colab_mcp.auth.oauth.InstalledAppFlow.from_client_config", lambda *args, **kwargs: mock_flow)
    monkeypatch.setattr("colab_mcp.auth.oauth.fetch_user_email_from_token", lambda token: "researcher@colab.org")

    resp = await colab_login_impl(method="oauth_remote", auth_code="valid_google_code_123", auth_user_index=0)
    assert resp.status == "authenticated"
    assert resp.auth_method == "oauth_remote"
    assert resp.user_email == "researcher@colab.org"
    assert resp.has_refresh_token is True
    assert resp.expires_at is not None

    stored = get_credentials(0)
    assert stored is not None
    assert stored.access_token == "ya29.remote_step2_token"


@pytest.mark.asyncio
async def test_fastmcp_colab_get_auth_status_authenticated(mock_keystore_dir, mock_keyring, mock_machine_id, sample_credentials):
    store_credentials(0, sample_credentials)

    status = await colab_get_auth_status_impl(auth_user_index=0)
    assert status.is_authenticated is True
    assert status.token_valid is True
    assert status.user_email == "researcher@example.com"
    assert status.expires_in_seconds is not None
    assert status.expires_in_seconds > 0


@pytest.mark.asyncio
async def test_fastmcp_colab_get_auth_status_unauthenticated(mock_keystore_dir, mock_keyring, mock_machine_id):
    status = await colab_get_auth_status_impl(auth_user_index=0)
    assert status.is_authenticated is False
    assert status.token_valid is False
    assert status.user_email is None
    assert status.expires_in_seconds is None


@pytest.mark.asyncio
async def test_fastmcp_colab_logout(mock_keystore_dir, mock_keyring, mock_machine_id, sample_credentials):
    store_credentials(0, sample_credentials)
    assert get_credentials(0) is not None

    logout_resp = await colab_logout_impl(auth_user_index=0, terminate_active_runtimes=True)
    assert logout_resp.status == "logged_out"
    assert logout_resp.keystore_purged is True

    status = await colab_get_auth_status_impl(auth_user_index=0)
    assert status.is_authenticated is False


@pytest.mark.asyncio
async def test_fastmcp_colab_login_cookie_import(mock_keystore_dir, mock_keyring, mock_machine_id):
    cookies = {
        "SAPISID": "cookie_sapisid_val",
        "SID": "cookie_sid_val",
        "OSID": "cookie_osid_val",
    }
    resp = await colab_login_impl(method="cookie_import", cookies=cookies, auth_user_index=0)
    assert resp.status == "authenticated"
    assert resp.auth_method == "cookie_import"

    stored = get_credentials(0)
    assert stored is not None
    assert stored.is_cookie_based is True
    assert stored.cookies["SAPISID"] == "cookie_sapisid_val"


@pytest.mark.asyncio
async def test_fastmcp_colab_login_cookie_import_missing_keys_raises(mock_keystore_dir, mock_keyring, mock_machine_id):
    with pytest.raises(ValueError, match="SAPISID"):
        await colab_login_impl(method="cookie_import", cookies={"SID": "only_sid"}, auth_user_index=0)


@pytest.mark.asyncio
async def test_fastmcp_colab_get_auth_status_force_refresh_revoked(mock_keystore_dir, mock_keyring, mock_machine_id, sample_expiring_soon_credentials, monkeypatch):
    store_credentials(0, sample_expiring_soon_credentials)

    def mock_refresh_fail(self, request):
        raise RefreshError("invalid_grant: Token has been revoked.")

    monkeypatch.setattr(google.oauth2.credentials.Credentials, "refresh", mock_refresh_fail)

    status = await colab_get_auth_status_impl(auth_user_index=0, force_refresh=True)
    assert status.is_authenticated is False
    assert status.token_valid is False
    assert get_credentials(0) is None


@pytest.mark.asyncio
async def test_fastmcp_multi_account_routing(mock_keystore_dir, mock_keyring, mock_machine_id, sample_credentials):
    user0_creds = sample_credentials.model_copy(update={"user_email": "user0@gmail.com"})
    user1_creds = sample_credentials.model_copy(update={"user_email": "user1@gmail.com", "access_token": "token1"})

    store_credentials(0, user0_creds)
    store_credentials(1, user1_creds)

    status0 = await colab_get_auth_status_impl(auth_user_index=0)
    status1 = await colab_get_auth_status_impl(auth_user_index=1)

    assert status0.user_email == "user0@gmail.com"
    assert status1.user_email == "user1@gmail.com"

    # Logout user 0 only
    await colab_logout_impl(auth_user_index=0)
    assert (await colab_get_auth_status_impl(auth_user_index=0)).is_authenticated is False
    assert (await colab_get_auth_status_impl(auth_user_index=1)).is_authenticated is True


@pytest.mark.asyncio
async def test_fastmcp_colab_login_invalid_method_raises():
    with pytest.raises(ValueError, match="Unsupported authentication method"):
        await colab_login_impl(method="invalid_magic_method", auth_user_index=0)


@pytest.mark.asyncio
async def test_fastmcp_tool_listing(fastmcp_test_server):
    tools = await fastmcp_test_server.list_tools()
    tool_names = {t.name for t in tools}
    assert "colab_login" in tool_names
    assert "colab_get_auth_status" in tool_names
    assert "colab_logout" in tool_names


@pytest.mark.asyncio
async def test_fastmcp_call_tool_via_server(fastmcp_test_server, mock_keystore_dir, mock_keyring, mock_machine_id):
    # Call colab_login step 1 through fastmcp.call_tool
    result = await fastmcp_test_server.call_tool("colab_login", {"method": "oauth_remote", "auth_user_index": 0})
    # result is a tuple of (content, context) or list of Content blocks
    text_content = result[0][0].text if isinstance(result, tuple) else result[0].text
    data = json.loads(text_content)
    assert data["status"] == "authorization_required"
    assert "https://accounts.google.com/o/oauth2/" in data["auth_url"]


@pytest.mark.asyncio
async def test_colab_get_auth_status_offset_naive_preserves_keystore(mock_keystore_dir, mock_keyring, mock_machine_id, sample_expiring_soon_credentials, monkeypatch):
    """
    Direct verification of Finding 1 fix:
    Mock Google library returning offset-naive UTC expiry on refresh.
    colab_get_auth_status_impl must NOT crash with TypeError and MUST NOT purge keystore.
    """
    store_credentials(0, sample_expiring_soon_credentials)

    def mock_google_refresh_naive(self, request):
        self.token = "ya29.refreshed_from_google_naive"
        # Naive datetime from utcnow() as google-auth standard
        self.expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=3600)

    monkeypatch.setattr(google.oauth2.credentials.Credentials, "refresh", mock_google_refresh_naive)
    monkeypatch.setattr("colab_mcp.auth.oauth.fetch_user_email_from_token", lambda token: "user@colab.org")

    status = await colab_get_auth_status_impl(auth_user_index=0)
    assert status.is_authenticated is True
    assert status.token_valid is True
    assert status.expires_in_seconds is not None
    assert status.expires_in_seconds > 3500

    stored = get_credentials(0)
    assert stored is not None
    assert stored.access_token == "ya29.refreshed_from_google_naive"
    assert stored.token_expiry.tzinfo == timezone.utc


@pytest.mark.asyncio
async def test_colab_get_auth_status_transient_error_preserves_keystore(mock_keystore_dir, mock_keyring, mock_machine_id, sample_expiring_soon_credentials, monkeypatch):
    """
    Direct verification of Finding 2 fix:
    Transient network errors (e.g. connection timeout or DNS error) during token refresh
    must NOT purge user credentials from keystore.
    """
    store_credentials(0, sample_expiring_soon_credentials)

    def mock_google_refresh_network_fail(self, request):
        raise ConnectionResetError("Connection reset by peer during token refresh")

    monkeypatch.setattr(google.oauth2.credentials.Credentials, "refresh", mock_google_refresh_network_fail)

    status = await colab_get_auth_status_impl(auth_user_index=0)
    # Status reflects current validity, but keystore credentials MUST remain intact
    assert get_credentials(0) is not None
    assert get_credentials(0).refresh_token == sample_expiring_soon_credentials.refresh_token


@pytest.mark.asyncio
async def test_colab_logout_guarantees_keystore_purge_on_error(mock_keystore_dir, mock_keyring, mock_machine_id, sample_credentials, monkeypatch):
    """
    Direct verification of Finding 6 fix:
    If runtime session termination throws an unhandled exception, keystore purge
    in the finally block is guaranteed to run.
    """
    store_credentials(0, sample_credentials)
    assert get_credentials(0) is not None

    # Simulate runtime session manager throwing an exception
    class MockBrokenManager:
        async def terminate_all_sessions(self, auth_user_index: int = 0):
            raise RuntimeError("Fatal cloud runtime deallocation failure")

    import sys
    fake_mod = type(sys)("colab_mcp.runtime.session_manager")
    fake_mod.get_session_manager = lambda: MockBrokenManager()
    monkeypatch.setitem(sys.modules, "colab_mcp.runtime.session_manager", fake_mod)

    with pytest.raises(RuntimeError, match="Fatal cloud runtime deallocation failure"):
        await colab_logout_impl(auth_user_index=0, terminate_active_runtimes=True)

    # Keystore purge MUST have executed via finally:
    assert get_credentials(0) is None

