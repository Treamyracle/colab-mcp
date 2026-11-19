"""
Colab MCP Authentication Subsystem.
"""

from colab_mcp.auth.models import (
    AuthenticationRequiredError,
    AuthMethod,
    AuthStatus,
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
from colab_mcp.auth.keystore import (
    get_credentials,
    list_stored_user_indices,
    purge_credentials,
    store_credentials,
)
from colab_mcp.auth.oauth import (
    exchange_remote_auth_code,
    generate_remote_consent_url,
    get_valid_auth_headers,
    ingest_adc_credentials,
    refresh_credentials,
    run_loopback_flow,
)
from colab_mcp.auth.sapisid import (
    generate_sapisid_hash,
    get_sapisid_auth_headers,
)
from colab_mcp.auth.tools import register_auth_tools

__all__ = [
    "AuthMethod",
    "GoogleCredentials",
    "ColabLoginResponse",
    "ColabAuthStatusResponse",
    "ColabLogoutResponse",
    "LoginResult",
    "AuthStatus",
    "LogoutResult",
    "ColabAuthError",
    "AuthenticationRequiredError",
    "TokenRefreshError",
    "BrowserAuthError",
    "KeystoreError",
    "store_credentials",
    "get_credentials",
    "purge_credentials",
    "list_stored_user_indices",
    "generate_remote_consent_url",
    "exchange_remote_auth_code",
    "run_loopback_flow",
    "ingest_adc_credentials",
    "refresh_credentials",
    "get_valid_auth_headers",
    "generate_sapisid_hash",
    "get_sapisid_auth_headers",
    "register_auth_tools",
]
