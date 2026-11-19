"""
Unit tests for Chrome DevTools Protocol (CDP) browser automation and cookie extraction.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from colab_mcp.auth import cdp_browser
from colab_mcp.auth.models import BrowserAuthError, GoogleCredentials


def test_mandatory_cookies_list():
    assert "SAPISID" in cdp_browser.MANDATORY_COOKIES
    assert "SID" in cdp_browser.MANDATORY_COOKIES
    assert "OSID" in cdp_browser.MANDATORY_COOKIES


def test_find_chrome_executable_raises_when_missing(monkeypatch):
    monkeypatch.setattr(cdp_browser.os.path, "isfile", lambda p: False)
    monkeypatch.setattr(cdp_browser.shutil, "which", lambda b: None)
    with pytest.raises(FileNotFoundError, match="Google Chrome executable not found"):
        cdp_browser.find_chrome_executable()


def test_find_chrome_executable_found(monkeypatch):
    if sys.platform == "win32":
        monkeypatch.setattr(cdp_browser.os.path, "isfile", lambda p: "chrome.exe" in p)
    else:
        monkeypatch.setattr(cdp_browser.shutil, "which", lambda b: "/usr/bin/google-chrome")

    chrome_path = cdp_browser.find_chrome_executable()
    assert chrome_path is not None


@pytest.mark.asyncio
async def test_wait_for_cdp_ready_timeout(monkeypatch):
    async def mock_get(*args, **kwargs):
        raise cdp_browser.httpx.ConnectError("Connection refused")

    monkeypatch.setattr(cdp_browser.httpx.AsyncClient, "get", mock_get)
    ready = await cdp_browser.wait_for_cdp_ready(port=9999, max_retries=2, interval=0.01)
    assert ready is False


@pytest.mark.asyncio
async def test_wait_for_cdp_ready_success(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    async def mock_get(*args, **kwargs):
        return mock_resp

    monkeypatch.setattr(cdp_browser.httpx.AsyncClient, "get", mock_get)
    ready = await cdp_browser.wait_for_cdp_ready(port=9222, max_retries=2, interval=0.01)
    assert ready is True


@pytest.mark.asyncio
async def test_launch_and_extract_cdp_cookies_timeout_on_readiness(monkeypatch):
    monkeypatch.setattr(cdp_browser, "find_chrome_executable", lambda: "fake_chrome")
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    monkeypatch.setattr(cdp_browser, "launch_isolated_subprocess", lambda *args, **kwargs: mock_proc)
    monkeypatch.setattr(cdp_browser, "wait_for_cdp_ready", AsyncMock(return_value=False))

    mock_playwright = MagicMock()
    monkeypatch.setitem(sys.modules, "playwright", mock_playwright)
    monkeypatch.setitem(sys.modules, "playwright.async_api", mock_playwright)

    with pytest.raises(BrowserAuthError, match="Timed out waiting for Chrome DevTools Protocol port"):
        await cdp_browser.launch_and_extract_cdp_cookies(auth_user_index=0)

    # Asserts process termination in finally block was called
    mock_proc.terminate.assert_called()


@pytest.mark.asyncio
async def test_launch_and_extract_cdp_cookies_raises_when_playwright_missing(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if "playwright" in name:
            raise ImportError("No module named 'playwright'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "playwright", raising=False)
    monkeypatch.delitem(sys.modules, "playwright.async_api", raising=False)

    with pytest.raises(BrowserAuthError, match="Playwright is not installed"):
        await cdp_browser.launch_and_extract_cdp_cookies(auth_user_index=0)
