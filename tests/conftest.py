"""
Global pytest fixtures and test isolation environment for colabmcp.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Tuple
import pytest
from mcp.server.fastmcp import FastMCP

from colab_mcp.auth.models import AuthMethod, GoogleCredentials
from colab_mcp.auth.tools import register_auth_tools
from colab_mcp.utils.logging import unprotect_stdout


@pytest.fixture(autouse=True)
def cleanup_stdout():
    """Ensure sys.stdout is cleanly restored and active flows reset after each test."""
    from colab_mcp.auth.oauth import _ACTIVE_REMOTE_FLOWS
    _ACTIVE_REMOTE_FLOWS.clear()
    yield
    _ACTIVE_REMOTE_FLOWS.clear()
    unprotect_stdout()


@pytest.fixture
def mock_keystore_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide isolated directory for encrypted credentials fallback."""
    cred_dir = tmp_path / ".colabmcp"
    cred_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("colab_mcp.auth.keystore.get_credentials_dir", lambda: cred_dir)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return cred_dir


@pytest.fixture
def mock_keyring(monkeypatch: pytest.MonkeyPatch) -> Dict[Tuple[str, str], str]:
    """Provide in-memory keyring storage and mock keyring module."""
    vault: Dict[Tuple[str, str], str] = {}

    def mock_set_password(service_name: str, username: str, password: str) -> None:
        vault[(service_name, username)] = password

    def mock_get_password(service_name: str, username: str) -> str | None:
        return vault.get((service_name, username))

    def mock_delete_password(service_name: str, username: str) -> None:
        if (service_name, username) in vault:
            del vault[(service_name, username)]
        else:
            import keyring.errors
            raise keyring.errors.PasswordDeleteError("Password not found")

    import keyring
    monkeypatch.setattr(keyring, "set_password", mock_set_password)
    monkeypatch.setattr(keyring, "get_password", mock_get_password)
    monkeypatch.setattr(keyring, "delete_password", mock_delete_password)
    return vault


@pytest.fixture
def mock_machine_id(monkeypatch: pytest.MonkeyPatch) -> bytes:
    """Mock deterministic machine ID for AES-256-GCM PBKDF2 derivation."""
    deterministic_id = b"DETERMINISTIC_TEST_MACHINE_ID_BYTES_12345"
    monkeypatch.setattr("colab_mcp.auth.keystore.get_machine_id_bytes", lambda: deterministic_id)
    return deterministic_id


@pytest.fixture
def sample_credentials() -> GoogleCredentials:
    """Valid credentials with 1 hour remaining."""
    return GoogleCredentials(
        access_token="ya29.a0AfH6SMB_MOCK_ACCESS_TOKEN_XYZ1234567890",
        refresh_token="1//04MOCK_REFRESH_TOKEN_ABC1234567890",
        token_uri="https://oauth2.googleapis.com/token",
        client_id="764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com",
        scopes=["https://www.googleapis.com/auth/colaboratory"],
        token_expiry=datetime.now(timezone.utc) + timedelta(seconds=3600),
        user_email="researcher@example.com",
        cookies={"SAPISID": "sapisid_mock_123", "SID": "sid_mock_456", "OSID": "osid_mock_789"},
        auth_method=AuthMethod.OAUTH_REMOTE,
    )


@pytest.fixture
def sample_expiring_soon_credentials() -> GoogleCredentials:
    """Credentials expiring in 120 seconds (triggers proactive refresh)."""
    return GoogleCredentials(
        access_token="ya29.a0AfH6SMB_EXPIRING_SOON_ACCESS_TOKEN",
        refresh_token="1//04MOCK_REFRESH_TOKEN_EXPIRING",
        token_uri="https://oauth2.googleapis.com/token",
        token_expiry=datetime.now(timezone.utc) + timedelta(seconds=120),
        user_email="researcher@example.com",
        auth_method=AuthMethod.OAUTH_REMOTE,
    )


@pytest.fixture
def fastmcp_test_server() -> FastMCP:
    """FastMCP test server with auth tools registered."""
    server = FastMCP("test-colab-mcp")
    register_auth_tools(server)
    return server


@pytest.fixture
def isolated_keystore_dir(mock_keystore_dir: Path) -> Path:
    """Alias for mock_keystore_dir to satisfy all test fixtures."""
    return mock_keystore_dir


@pytest.fixture
def sample_tokens() -> Dict[str, Any]:
    """Sample tokens, secrets, and cookies for deterministic test assertions."""
    return {
        "access_token": "ya29.a0AfH6SMB_SAMPLE_ACCESS_TOKEN_FOR_COLAB_MCP_TESTS_1234567890",
        "refresh_token": "1//04abcDEF_SAMPLE_REFRESH_TOKEN_FOR_COLAB_MCP_TESTS_1234567890",
        "client_id": "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com",
        "client_secret": "d-FL95Q19q7MQmFpd7hHD0Ty",
        "token_uri": "https://oauth2.googleapis.com/token",
        "user_email": "researcher@example.com",
        "cookies": {
            "SAPISID": "test_sapisid_cookie_12345",
            "SID": "test_sid_cookie_67890",
            "OSID": "test_osid_cookie_colab_13579",
            "HSID": "test_hsid_cookie_24680",
            "SSID": "test_ssid_cookie_11223",
            "APISID": "test_apisid_cookie_33445",
            "__Secure-1PSID": "test_sec_1psid_55667",
            "__Secure-3PSID": "test_sec_3psid_77889",
            "__Secure-OSID": "test_sec_osid_99001",
        },
    }


@pytest.fixture
def mock_google_credentials(sample_credentials: GoogleCredentials) -> GoogleCredentials:
    """Alias for sample_credentials."""
    return sample_credentials


@pytest.fixture
def mock_cookie_credentials(sample_tokens: Dict[str, Any]) -> GoogleCredentials:
    """GoogleCredentials model initialized with cookies."""
    return GoogleCredentials(
        auth_method=AuthMethod.COOKIE_IMPORT,
        auth_user_index=0,
        user_email=sample_tokens["user_email"],
        cookies=sample_tokens["cookies"],
    )
