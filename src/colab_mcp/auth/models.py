"""
Authentication data models, enums, serializers, and domain exceptions.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


class AuthMethod(str, Enum):
    """Supported authentication mechanisms."""
    OAUTH_REMOTE = "oauth_remote"
    OAUTH_LOOPBACK = "oauth_loopback"
    OAUTH_ADC = "oauth_adc"
    CDP_BROWSER = "cdp_browser"
    COOKIE_IMPORT = "cookie_import"


class GoogleCredentials(BaseModel):
    """
    Unified Google credential container for Colab authentication.
    Supports both OAuth 2.0 token-based auth and browser-extracted cookie auth.
    """
    model_config = ConfigDict(validate_assignment=True)

    auth_method: AuthMethod = AuthMethod.OAUTH_REMOTE
    auth_user_index: int = Field(default=0, ge=0, description="Google multi-account user index")
    user_email: Optional[str] = Field(default=None, description="Google user email identity")

    # OAuth 2.0 fields
    access_token: Optional[str] = Field(default=None, description="Bearer access token")
    refresh_token: Optional[str] = Field(default=None, description="OAuth refresh token")
    token_expiry: Optional[datetime] = Field(default=None, description="UTC access token expiration")
    token_uri: str = Field(default="https://oauth2.googleapis.com/token", description="OAuth token endpoint")
    client_id: Optional[str] = Field(default=None, description="OAuth client ID")
    client_secret: Optional[str] = Field(default=None, description="OAuth client secret")
    scopes: List[str] = Field(default_factory=list, description="Granted OAuth scopes")
    is_adc: bool = Field(default=False, description="Whether loaded from ambient ADC")

    # CDP / Cookie fields
    cookies: Dict[str, str] = Field(default_factory=dict, description="Session cookies (SAPISID, SID, OSID, etc.)")
    sapisid_hash: Optional[str] = Field(default=None, description="Cached or precomputed SAPISIDHASH")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def expiry(self) -> Optional[datetime]:
        """Alias for token_expiry for compatibility."""
        return self.token_expiry

    @expiry.setter
    def expiry(self, value: Optional[datetime]) -> None:
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        self.token_expiry = value

    @field_validator("token_expiry", mode="before")
    @classmethod
    def parse_token_expiry(cls, value: Any) -> Optional[datetime]:
        """Ensures token_expiry is parsed into a timezone-aware UTC datetime."""
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        if isinstance(value, str):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        raise ValueError(f"Unsupported format for token_expiry: {value}")

    @field_serializer("token_expiry")
    def serialize_token_expiry(self, value: Optional[datetime]) -> Optional[str]:
        """Serializes datetime to ISO-8601 UTC string."""
        if value is None:
            return None
        return value.astimezone(timezone.utc).isoformat()

    @property
    def is_oauth(self) -> bool:
        """True if credentials provide OAuth tokens."""
        return self.access_token is not None or self.refresh_token is not None

    @property
    def is_cookie_based(self) -> bool:
        """True if credentials provide Colab 1P session cookies."""
        return bool(self.cookies and "SAPISID" in self.cookies and "SID" in self.cookies)

    @property
    def has_refresh_token(self) -> bool:
        """True if a valid refresh token exists."""
        return bool(self.refresh_token)

    def is_expired(self, buffer_seconds: int = 300) -> bool:
        """
        Evaluates whether OAuth access token has expired or will expire
        within buffer_seconds (default: 5 minutes proactive window).
        Cookie-based credentials return False unless empty.
        """
        if self.is_cookie_based and not self.is_oauth:
            return False
        if not self.access_token:
            return True
        if not self.token_expiry:
            return False
        expiry = self.token_expiry if self.token_expiry.tzinfo else self.token_expiry.replace(tzinfo=timezone.utc)
        now_utc = datetime.now(timezone.utc)
        return (expiry.timestamp() - now_utc.timestamp()) <= buffer_seconds


class ColabLoginResponse(BaseModel):
    """Output schema for colab_login MCP tool."""
    status: Literal["authenticated", "authorization_required"] = Field(
        description="Current login status: 'authenticated' if credentials stored, 'authorization_required' if consent URL generated."
    )
    auth_method: str = Field(description="Authentication method used.")
    auth_user_index: int = Field(default=0, ge=0, description="Google multi-account user index.")
    user_email: Optional[str] = Field(default=None, description="Google account email if authenticated.")
    expires_at: Optional[str] = Field(default=None, description="ISO 8601 UTC timestamp of token expiry.")
    has_refresh_token: bool = Field(default=False, description="Whether a refresh token is persisted.")
    auth_url: Optional[str] = Field(default=None, description="Google OAuth authorization URL when authorization is required.")
    message: str = Field(description="Human-readable result or instruction.")


class ColabAuthStatusResponse(BaseModel):
    """Output schema for colab_get_auth_status MCP tool."""
    is_authenticated: bool = Field(description="True if credentials exist and token is valid.")
    auth_method: Optional[str] = Field(default=None, description="Authentication mechanism used.")
    auth_user_index: int = Field(default=0, ge=0, description="Google multi-account user index inspected.")
    user_email: Optional[str] = Field(default=None, description="Google account email.")
    token_valid: bool = Field(default=False, description="Whether the current token/cookie is valid.")
    expires_in_seconds: Optional[int] = Field(default=None, description="Remaining seconds until token expiration.")
    account_tier: Optional[str] = Field(default=None, description="Colab tier: 'FREE', 'PRO', 'PRO_PLUS' or None.")
    compute_units_remaining: Optional[float] = Field(default=None, description="Remaining Colab Compute Units (CCU).")
    active_sessions_count: int = Field(default=0, ge=0, description="Count of currently provisioned VM runtimes.")


class ColabLogoutResponse(BaseModel):
    """Output schema for colab_logout MCP tool."""
    status: Literal["logged_out"] = Field(default="logged_out", description="Status string, always 'logged_out'.")
    auth_user_index: int = Field(default=0, ge=0, description="Google multi-account user index logged out.")
    keystore_purged: bool = Field(default=True, description="True if credentials were removed from keystore.")
    terminated_runtimes_count: int = Field(default=0, ge=0, description="Number of cloud runtimes actively deallocated.")
    message: str = Field(description="Summary message.")


# Aliases for backward / cross-spec compatibility
LoginResult = ColabLoginResponse
AuthStatus = ColabAuthStatusResponse
LogoutResult = ColabLogoutResponse


# Domain Exceptions
class ColabAuthError(Exception):
    """Base exception for all authentication subsystem errors."""
    pass


class AuthenticationRequiredError(ColabAuthError):
    """Raised when an operation requires authentication but no valid credentials exist."""
    pass


class TokenRefreshError(ColabAuthError):
    """Raised when token refresh fails due to revocation, expiration, or network errors."""
    pass


class BrowserAuthError(ColabAuthError):
    """Raised when browser-assisted CDP cookie extraction fails or times out."""
    pass


class KeystoreError(ColabAuthError):
    """Raised when storing or retrieving credentials from Keyring / encrypted file fails."""
    pass
