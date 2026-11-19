"""
Unit tests for OAuth 2.0 lifecycle, flows, token refresh, and header assembly.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
import pytest
import google.oauth2.credentials
from google.auth.exceptions import RefreshError
from google_auth_oauthlib.flow import InstalledAppFlow

from colab_mcp.auth.keystore import get_credentials, store_credentials
from colab_mcp.auth.models import (
    AuthenticationRequiredError,
    AuthMethod,
    ColabAuthError,
    GoogleCredentials,
    TokenRefreshError,
)
from colab_mcp.auth.oauth import (
    COLAB_OAUTH_SCOPES,
    DEFAULT_CLIENT_ID,
    REMOTE_REDIRECT_URI,
    exchange_remote_auth_code,
    generate_remote_consent_url,
    get_valid_auth_headers,
    ingest_adc_credentials,
    refresh_credentials,
    run_loopback_flow,
)


def test_generate_remote_consent_url_parameters():
    url = generate_remote_consent_url(auth_user_index=0)
    assert "https://accounts.google.com/o/oauth2/" in url
    assert f"client_id={DEFAULT_CLIENT_ID}" in url
    assert "response_type=code" in url
    assert "access_type=offline" in url
    assert "prompt=consent" in url
    assert "token_usage=remote" in url


@pytest.mark.asyncio
async def test_exchange_remote_auth_code_success(monkeypatch):
    mock_credentials = MagicMock()
    mock_credentials.token = "ya29.new_mock_token_from_code"
    mock_credentials.refresh_token = "1//04new_mock_refresh_token"
    mock_credentials.expiry = datetime.now(timezone.utc) + timedelta(seconds=3600)
    mock_credentials.token_uri = "https://oauth2.googleapis.com/token"
    mock_credentials.client_id = DEFAULT_CLIENT_ID
    mock_credentials.client_secret = "secret"
    mock_credentials.scopes = COLAB_OAUTH_SCOPES

    mock_flow = MagicMock(spec=InstalledAppFlow)
    mock_flow.credentials = mock_credentials

    monkeypatch.setattr(
        "colab_mcp.auth.oauth.InstalledAppFlow.from_client_config",
        lambda *args, **kwargs: mock_flow,
    )
    monkeypatch.setattr(
        "colab_mcp.auth.oauth.fetch_user_email_from_token",
        lambda token: "code_user@example.com",
    )

    creds = await exchange_remote_auth_code("valid_auth_code_123", auth_user_index=0)
    assert creds.access_token == "ya29.new_mock_token_from_code"
    assert creds.refresh_token == "1//04new_mock_refresh_token"
    assert creds.user_email == "code_user@example.com"
    assert creds.auth_method == AuthMethod.OAUTH_REMOTE


@pytest.mark.asyncio
async def test_exchange_remote_auth_code_empty_raises():
    with pytest.raises(ColabAuthError):
        await exchange_remote_auth_code("", auth_user_index=0)


@pytest.mark.asyncio
async def test_proactive_refresh_threshold_triggered(mock_keystore_dir, mock_keyring, mock_machine_id, sample_expiring_soon_credentials, monkeypatch):
    store_credentials(0, sample_expiring_soon_credentials)

    refreshed_called = False

    def mock_refresh(self, request):
        nonlocal refreshed_called
        refreshed_called = True
        self.token = "ya29.refreshed_proactive_token_999"
        self.expiry = datetime.now(timezone.utc) + timedelta(seconds=3600)

    monkeypatch.setattr(google.oauth2.credentials.Credentials, "refresh", mock_refresh)

    headers = await get_valid_auth_headers(0)
    assert refreshed_called is True
    assert "ya29.refreshed_proactive_token_999" in headers["Authorization"]
    assert headers["X-Goog-AuthUser"] == "0"


@pytest.mark.asyncio
async def test_proactive_refresh_not_triggered_when_fresh(mock_keystore_dir, mock_keyring, mock_machine_id, sample_credentials, monkeypatch):
    store_credentials(0, sample_credentials)

    refreshed_called = False

    def mock_refresh(self, request):
        nonlocal refreshed_called
        refreshed_called = True

    monkeypatch.setattr(google.oauth2.credentials.Credentials, "refresh", mock_refresh)

    headers = await get_valid_auth_headers(0)
    assert refreshed_called is False
    assert sample_credentials.access_token in headers["Authorization"]


def test_ingest_adc_success(monkeypatch):
    mock_adc_creds = MagicMock()
    mock_adc_creds.valid = True
    mock_adc_creds.token = "ya29.adc_ingested_token"
    mock_adc_creds.refresh_token = "1//04adc_refresh_token"
    mock_adc_creds.expiry = datetime.now(timezone.utc) + timedelta(seconds=3600)
    mock_adc_creds.token_uri = "https://oauth2.googleapis.com/token"
    mock_adc_creds.client_id = DEFAULT_CLIENT_ID
    mock_adc_creds.client_secret = "secret"

    import google.auth
    monkeypatch.setattr(google.auth, "default", lambda scopes: (mock_adc_creds, "mock-project-id"))
    monkeypatch.setattr("colab_mcp.auth.oauth.fetch_user_email_from_token", lambda token: "adc@example.com")

    creds = ingest_adc_credentials(auth_user_index=1)
    assert creds.access_token == "ya29.adc_ingested_token"
    assert creds.auth_method == AuthMethod.OAUTH_ADC
    assert creds.auth_user_index == 1
    assert creds.is_adc is True


@pytest.mark.asyncio
async def test_refresh_error_revoked_token_purges_keystore(mock_keystore_dir, mock_keyring, mock_machine_id, sample_expiring_soon_credentials, monkeypatch):
    store_credentials(0, sample_expiring_soon_credentials)

    def mock_refresh_revoked(self, request):
        raise RefreshError("invalid_grant: Token has been expired or revoked.")

    monkeypatch.setattr(google.oauth2.credentials.Credentials, "refresh", mock_refresh_revoked)

    with pytest.raises(AuthenticationRequiredError):
        await refresh_credentials(sample_expiring_soon_credentials)

    # Keystore should have been wiped for user 0
    assert get_credentials(0) is None


@pytest.mark.asyncio
async def test_token_refresh_network_timeout(mock_keystore_dir, mock_keyring, mock_machine_id, sample_expiring_soon_credentials, monkeypatch):
    store_credentials(0, sample_expiring_soon_credentials)

    def mock_refresh_timeout(self, request):
        raise TimeoutError("Network timeout connecting to oauth2.googleapis.com")

    monkeypatch.setattr(google.oauth2.credentials.Credentials, "refresh", mock_refresh_timeout)

    with pytest.raises(TokenRefreshError):
        await refresh_credentials(sample_expiring_soon_credentials)

    # Keystore should NOT be purged on transient network timeout
    assert get_credentials(0) is not None


@pytest.mark.asyncio
async def test_loopback_flow_success(monkeypatch):
    mock_creds = MagicMock()
    mock_creds.token = "ya29.loopback_token"
    mock_creds.refresh_token = "1//04loopback_refresh"
    mock_creds.expiry = datetime.now(timezone.utc) + timedelta(seconds=3600)
    mock_creds.token_uri = "https://oauth2.googleapis.com/token"
    mock_creds.client_id = DEFAULT_CLIENT_ID
    mock_creds.client_secret = "secret"
    mock_creds.scopes = COLAB_OAUTH_SCOPES

    mock_flow = MagicMock(spec=InstalledAppFlow)
    mock_flow.run_local_server.return_value = mock_creds

    monkeypatch.setattr("colab_mcp.auth.oauth.InstalledAppFlow.from_client_config", lambda *args, **kwargs: mock_flow)
    monkeypatch.setattr("colab_mcp.auth.oauth.fetch_user_email_from_token", lambda token: "loopback@example.com")

    creds = await run_loopback_flow(auth_user_index=0, timeout_seconds=60)
    assert creds.access_token == "ya29.loopback_token"
    assert creds.auth_method == AuthMethod.OAUTH_LOOPBACK


@pytest.mark.asyncio
async def test_get_valid_auth_headers_unauthenticated_raises(mock_keystore_dir, mock_keyring, mock_machine_id):
    with pytest.raises(AuthenticationRequiredError):
        await get_valid_auth_headers(auth_user_index=42)


def test_google_credentials_naive_datetime_normalization():
    """Verify offset-naive datetime is automatically normalized to UTC across initialization and assignment."""
    naive_dt = datetime(2026, 11, 16, 12, 0, 0)  # tzinfo is None
    creds = GoogleCredentials(
        access_token="ya29.test",
        refresh_token="1//refresh",
        token_expiry=naive_dt,
        auth_user_index=0,
    )
    assert creds.token_expiry is not None
    assert creds.token_expiry.tzinfo == timezone.utc

    # Test direct property assignment
    later_naive = datetime(2026, 11, 16, 15, 30, 0)
    creds.token_expiry = later_naive
    assert creds.token_expiry.tzinfo == timezone.utc
    assert creds.token_expiry == later_naive.replace(tzinfo=timezone.utc)

    # Test alias property setter (creds.expiry = ...)
    creds.expiry = later_naive
    assert creds.expiry.tzinfo == timezone.utc
    assert creds.expiry == later_naive.replace(tzinfo=timezone.utc)


def test_google_credentials_is_expired_safe_with_naive_and_aware():
    """Verify is_expired() performs timezone-aware comparison without throwing TypeError or timezone skew."""
    now_utc = datetime.now(timezone.utc)
    future_aware = now_utc + timedelta(seconds=600)
    creds = GoogleCredentials(
        access_token="ya29.test",
        token_expiry=future_aware,
    )
    assert creds.is_expired(buffer_seconds=300) is False

    # Force naive expiry on internal attribute to test defensive check in is_expired
    object.__setattr__(creds, "token_expiry", datetime(2026, 1, 1, 0, 0, 0))
    # Must evaluate without TypeError
    assert creds.is_expired(buffer_seconds=300) is True


@pytest.mark.asyncio
async def test_refresh_credentials_normalizes_google_naive_expiry(mock_keystore_dir, mock_keyring, mock_machine_id, sample_expiring_soon_credentials, monkeypatch):
    """Verify that Google Credentials returning naive UTC expiry is converted to timezone-aware UTC datetime."""
    store_credentials(0, sample_expiring_soon_credentials)

    # Mock Google's Credentials.refresh returning naive datetime (standard behavior of google-auth)
    def mock_google_refresh(self, request):
        self.token = "ya29.new_token_from_google"
        # google-auth returns datetime.datetime.utcnow(), which is offset-naive!
        self.expiry = datetime(2026, 11, 16, 20, 0, 0)  # tzinfo=None

    monkeypatch.setattr(google.oauth2.credentials.Credentials, "refresh", mock_google_refresh)
    monkeypatch.setattr("colab_mcp.auth.oauth.fetch_user_email_from_token", lambda token: "test@example.com")

    refreshed = await refresh_credentials(sample_expiring_soon_credentials)
    assert refreshed.access_token == "ya29.new_token_from_google"
    assert refreshed.token_expiry is not None
    assert refreshed.token_expiry.tzinfo == timezone.utc
    assert refreshed.token_expiry == datetime(2026, 11, 16, 20, 0, 0, tzinfo=timezone.utc)

    # Check that keystore also has timezone-aware expiry
    stored = get_credentials(0)
    assert stored is not None
    assert stored.token_expiry.tzinfo == timezone.utc

