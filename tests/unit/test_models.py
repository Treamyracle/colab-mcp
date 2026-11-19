"""
Unit tests for authentication data models, enums, serializers, and domain exceptions.
"""

from datetime import datetime, timedelta, timezone
import pytest
from pydantic import ValidationError

from colab_mcp.auth.models import (
    AuthMethod,
    AuthStatus,
    AuthenticationRequiredError,
    BrowserAuthError,
    ColabAuthError,
    ColabAuthStatusResponse,
    ColabLoginResponse,
    ColabLogoutResponse,
    GoogleCredentials,
    KeystoreError,
    LoginResult,
    LogoutResult,
    TokenRefreshError,
)


# --- AuthMethod Enum Tests ---


def test_auth_method_enum_values():
    assert AuthMethod.OAUTH_REMOTE == "oauth_remote"
    assert AuthMethod.OAUTH_LOOPBACK == "oauth_loopback"
    assert AuthMethod.OAUTH_ADC == "oauth_adc"
    assert AuthMethod.CDP_BROWSER == "cdp_browser"
    assert AuthMethod.COOKIE_IMPORT == "cookie_import"


def test_auth_method_enum_membership():
    methods = [m.value for m in AuthMethod]
    assert "oauth_remote" in methods
    assert "oauth_loopback" in methods
    assert "oauth_adc" in methods
    assert "cdp_browser" in methods
    assert "cookie_import" in methods


# --- GoogleCredentials Tests ---


def test_google_credentials_defaults():
    creds = GoogleCredentials()
    assert creds.auth_method == AuthMethod.OAUTH_REMOTE
    assert creds.auth_user_index == 0
    assert creds.user_email is None
    assert creds.access_token is None
    assert creds.refresh_token is None
    assert creds.token_expiry is None
    assert creds.token_uri == "https://oauth2.googleapis.com/token"
    assert creds.client_id is None
    assert creds.client_secret is None
    assert creds.scopes == []
    assert creds.is_adc is False
    assert creds.cookies == {}
    assert creds.sapisid_hash is None
    assert creds.created_at is not None
    assert creds.created_at.tzinfo == timezone.utc


def test_google_credentials_auth_user_index_validation():
    creds = GoogleCredentials(auth_user_index=3)
    assert creds.auth_user_index == 3

    with pytest.raises(ValidationError):
        GoogleCredentials(auth_user_index=-1)


def test_parse_token_expiry_none():
    assert GoogleCredentials.parse_token_expiry(None) is None


def test_parse_token_expiry_datetime_naive():
    naive_dt = datetime(2026, 6, 15, 12, 0, 0)
    parsed = GoogleCredentials.parse_token_expiry(naive_dt)
    assert parsed is not None
    assert parsed.tzinfo == timezone.utc
    assert parsed.year == 2026
    assert parsed.month == 6


def test_parse_token_expiry_datetime_aware():
    tz_plus_2 = timezone(timedelta(hours=2))
    aware_dt = datetime(2026, 6, 15, 14, 0, 0, tzinfo=tz_plus_2)
    parsed = GoogleCredentials.parse_token_expiry(aware_dt)
    assert parsed is not None
    assert parsed.tzinfo == timezone.utc
    # 14:00 UTC+2 is 12:00 UTC
    assert parsed.hour == 12


def test_parse_token_expiry_timestamp_numeric():
    ts = 1780000000.0
    parsed = GoogleCredentials.parse_token_expiry(ts)
    assert parsed is not None
    assert parsed.tzinfo == timezone.utc
    assert parsed.timestamp() == ts

    int_ts = 1780000000
    parsed_int = GoogleCredentials.parse_token_expiry(int_ts)
    assert parsed_int is not None
    assert parsed_int.timestamp() == float(int_ts)


def test_parse_token_expiry_iso_string_with_z():
    iso_str = "2026-11-16T18:00:00Z"
    parsed = GoogleCredentials.parse_token_expiry(iso_str)
    assert parsed is not None
    assert parsed.tzinfo == timezone.utc
    assert parsed.year == 2026
    assert parsed.hour == 18


def test_parse_token_expiry_iso_string_with_offset():
    iso_str = "2026-11-16T20:00:00+02:00"
    parsed = GoogleCredentials.parse_token_expiry(iso_str)
    assert parsed is not None
    assert parsed.tzinfo == timezone.utc
    assert parsed.hour == 18


def test_parse_token_expiry_invalid_type_raises():
    with pytest.raises(ValueError, match="Unsupported format for token_expiry"):
        GoogleCredentials.parse_token_expiry(["not", "a", "date"])

    with pytest.raises(ValueError):
        GoogleCredentials.parse_token_expiry("not-a-valid-iso-date")


def test_expiry_property_and_setter():
    creds = GoogleCredentials()
    assert creds.expiry is None

    # Assign naive datetime via property setter
    naive = datetime(2026, 8, 1, 10, 0, 0)
    creds.expiry = naive
    assert creds.expiry is not None
    assert creds.expiry.tzinfo == timezone.utc
    assert creds.token_expiry == creds.expiry

    # Assign None
    creds.expiry = None
    assert creds.expiry is None
    assert creds.token_expiry is None


def test_serialize_token_expiry():
    dt = datetime(2026, 5, 20, 15, 30, 0, tzinfo=timezone.utc)
    creds = GoogleCredentials(token_expiry=dt)
    dumped = creds.model_dump(mode="json")
    assert dumped["token_expiry"] == "2026-05-20T15:30:00+00:00"

    creds_no_expiry = GoogleCredentials()
    dumped_none = creds_no_expiry.model_dump(mode="json")
    assert dumped_none["token_expiry"] is None


def test_is_oauth_property():
    creds_empty = GoogleCredentials()
    assert creds_empty.is_oauth is False

    creds_access = GoogleCredentials(access_token="ya29.xyz")
    assert creds_access.is_oauth is True

    creds_refresh = GoogleCredentials(refresh_token="1//0abc")
    assert creds_refresh.is_oauth is True


def test_is_cookie_based_property():
    creds_empty = GoogleCredentials()
    assert creds_empty.is_cookie_based is False

    creds_partial = GoogleCredentials(cookies={"SID": "secret_sid"})
    assert creds_partial.is_cookie_based is False

    creds_full = GoogleCredentials(cookies={"SID": "secret_sid", "SAPISID": "secret_sapisid"})
    assert creds_full.is_cookie_based is True


def test_has_refresh_token_property():
    creds_empty = GoogleCredentials()
    assert creds_empty.has_refresh_token is False

    creds_empty_str = GoogleCredentials(refresh_token="")
    assert creds_empty_str.has_refresh_token is False

    creds_with_token = GoogleCredentials(refresh_token="1//0abc")
    assert creds_with_token.has_refresh_token is True


def test_is_expired_cookie_only():
    creds_cookie = GoogleCredentials(
        cookies={"SID": "secret_sid", "SAPISID": "secret_sapisid"}
    )
    # Cookie-based credentials without OAuth return False
    assert creds_cookie.is_expired() is False


def test_is_expired_oauth_no_access_token():
    creds_no_token = GoogleCredentials(refresh_token="1//0abc")
    # OAuth with no access token is treated as expired
    assert creds_no_token.is_expired() is True


def test_is_expired_oauth_no_expiry():
    creds_no_expiry = GoogleCredentials(access_token="ya29.xyz")
    # Access token with unknown expiry is not marked expired
    assert creds_no_expiry.is_expired() is False


def test_is_expired_proactive_window():
    now = datetime.now(timezone.utc)

    # Token expires in 600s: not expired with default 300s buffer
    creds_fresh = GoogleCredentials(
        access_token="ya29.fresh",
        token_expiry=now + timedelta(seconds=600),
    )
    assert creds_fresh.is_expired(buffer_seconds=300) is False

    # Token expires in 200s: expired with default 300s buffer
    creds_near_expiry = GoogleCredentials(
        access_token="ya29.near",
        token_expiry=now + timedelta(seconds=200),
    )
    assert creds_near_expiry.is_expired(buffer_seconds=300) is True

    # Token expired 100s ago
    creds_past = GoogleCredentials(
        access_token="ya29.past",
        token_expiry=now - timedelta(seconds=100),
    )
    assert creds_past.is_expired(buffer_seconds=300) is True

    # Custom buffer: 100s buffer on token expiring in 200s -> not expired
    assert creds_near_expiry.is_expired(buffer_seconds=100) is False


# --- Response Models & Aliases Tests ---


def test_colab_login_response_and_alias():
    res = ColabLoginResponse(
        status="authenticated",
        auth_method="oauth_remote",
        auth_user_index=0,
        user_email="user@google.com",
        expires_at="2026-11-16T18:00:00Z",
        has_refresh_token=True,
        message="Login successful",
    )
    assert res.status == "authenticated"
    assert res.auth_user_index == 0
    assert LoginResult is ColabLoginResponse

    # Validation: invalid status raises error
    with pytest.raises(ValidationError):
        ColabLoginResponse(
            status="invalid_status",  # type: ignore[arg-type]
            auth_method="oauth_remote",
            message="Test",
        )


def test_colab_auth_status_response_and_alias():
    res = ColabAuthStatusResponse(
        is_authenticated=True,
        auth_method="oauth_remote",
        auth_user_index=0,
        user_email="user@google.com",
        token_valid=True,
        expires_in_seconds=3600,
        account_tier="PRO_PLUS",
        compute_units_remaining=100.5,
        active_sessions_count=2,
    )
    assert res.is_authenticated is True
    assert res.account_tier == "PRO_PLUS"
    assert res.compute_units_remaining == 100.5
    assert AuthStatus is ColabAuthStatusResponse

    # Validation: negative active_sessions_count raises error
    with pytest.raises(ValidationError):
        ColabAuthStatusResponse(
            is_authenticated=True,
            active_sessions_count=-1,
        )


def test_colab_logout_response_and_alias():
    res = ColabLogoutResponse(
        auth_user_index=1,
        keystore_purged=True,
        terminated_runtimes_count=1,
        message="Logged out",
    )
    assert res.status == "logged_out"
    assert res.auth_user_index == 1
    assert res.keystore_purged is True
    assert LogoutResult is ColabLogoutResponse

    # Validation: negative terminated_runtimes_count raises error
    with pytest.raises(ValidationError):
        ColabLogoutResponse(
            auth_user_index=0,
            terminated_runtimes_count=-1,
            message="Invalid",
        )


# --- Domain Exceptions Hierarchy Tests ---


def test_domain_exceptions_hierarchy():
    assert issubclass(AuthenticationRequiredError, ColabAuthError)
    assert issubclass(TokenRefreshError, ColabAuthError)
    assert issubclass(BrowserAuthError, ColabAuthError)
    assert issubclass(KeystoreError, ColabAuthError)
    assert issubclass(ColabAuthError, Exception)

    err = AuthenticationRequiredError("Auth required")
    assert isinstance(err, ColabAuthError)
    assert str(err) == "Auth required"
