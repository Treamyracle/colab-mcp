"""
Chrome DevTools Protocol (CDP) headed browser automation and session cookie extraction.
"""

import asyncio
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional

import httpx

from colab_mcp.auth.models import (
    AuthMethod,
    BrowserAuthError,
    GoogleCredentials,
)
from colab_mcp.utils.logging import get_logger
from colab_mcp.utils.process import launch_isolated_subprocess, terminate_process

logger = get_logger(__name__)

MANDATORY_COOKIES = ["SAPISID", "SID", "OSID"]
OPTIONAL_HARVEST_COOKIES = [
    "HSID",
    "SSID",
    "APISID",
    "__Secure-1PSID",
    "__Secure-3PSID",
    "__Secure-OSID",
]


def find_chrome_executable() -> str:
    """
    Discovers installed Google Chrome binary across Windows, macOS, and Linux.
    Raises FileNotFoundError if Chrome is not detected.
    """
    if sys.platform == "win32":
        candidates = [
            os.path.join(os.environ.get("PROGRAMFILES", "C:\\Program Files"), "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(os.environ.get("PROGRAMFILES(X86)", "C:\\Program Files (x86)"), "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
    elif sys.platform == "darwin":
        mac_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        if os.path.isfile(mac_path):
            return mac_path
    else:  # Linux / Unix
        for binary in ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]:
            found = shutil.which(binary)
            if found:
                return found

    raise FileNotFoundError(
        "Google Chrome executable not found on host machine. "
        "Please install Google Chrome or authenticate via OAuth 2.0."
    )


async def wait_for_cdp_ready(port: int = 9222, max_retries: int = 30, interval: float = 0.2) -> bool:
    """
    Polls http://127.0.0.1:{port}/json/version to eliminate browser launch race conditions.
    """
    url = f"http://127.0.0.1:{port}/json/version"
    async with httpx.AsyncClient(timeout=1.0) as client:
        for _ in range(max_retries):
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return True
            except (httpx.HTTPError, OSError):
                pass
            await asyncio.sleep(interval)
    return False


async def launch_and_extract_cdp_cookies(
    auth_user_index: int = 0,
    timeout_seconds: int = 180,
    cdp_port: int = 9222,
) -> GoogleCredentials:
    """
    Spawns headed Chrome on remote debugging port, attaches via Playwright CDP,
    awaits user authentication on colab.research.google.com, verifies OSID, SAPISID,
    and SID, terminates the browser process, and returns GoogleCredentials.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError as e:
        raise BrowserAuthError(
            "Playwright is not installed. Install optional browser dependencies via "
            "`pip install 'colab-mcp[browser]'` and run `playwright install chromium`."
        ) from e

    chrome_path = find_chrome_executable()
    user_data_dir = os.path.expanduser("~/.colabmcp/chrome_profile")
    os.makedirs(user_data_dir, exist_ok=True)

    chrome_cmd = [
        chrome_path,
        f"--remote-debugging-port={cdp_port}",
        f"--user-data-dir={user_data_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        f"https://colab.research.google.com/?authuser={auth_user_index}",
    ]

    proc = launch_isolated_subprocess(chrome_cmd)

    try:
        logger.info("Awaiting Chrome CDP readiness on port %d...", cdp_port)
        if not await wait_for_cdp_ready(port=cdp_port, max_retries=30, interval=0.2):
            raise BrowserAuthError(
                f"Timed out waiting for Chrome DevTools Protocol port {cdp_port} to become active."
            )

        extracted_cookies: Dict[str, str] = {}
        user_email: Optional[str] = None

        async with async_playwright() as p:
            logger.info("Connecting Playwright over CDP...")
            browser = await p.chromium.connect_over_cdp(f"http://127.0.0.1:{cdp_port}")
            try:
                context = browser.contexts[0]
                page = context.pages[0] if context.pages else await context.new_page()

                start_time = asyncio.get_event_loop().time()
                while asyncio.get_event_loop().time() - start_time < timeout_seconds:
                    current_url = page.url
                    if "colab.research.google.com" in current_url:
                        cookie_list = await context.cookies([
                            "https://colab.research.google.com",
                            "https://google.com",
                            "https://accounts.google.com",
                        ])
                        cookie_map = {c["name"]: c["value"] for c in cookie_list}

                        # Check mandatory cookies
                        if all(m in cookie_map for m in MANDATORY_COOKIES):
                            extracted_cookies = cookie_map
                            try:
                                page_title = await page.title()
                                logger.debug("Colab page title: %s", page_title)
                            except Exception:
                                pass
                            break

                    await asyncio.sleep(2.0)

                if not extracted_cookies:
                    raise BrowserAuthError(
                        f"Login timed out after {timeout_seconds}s before authenticated Colab cookies "
                        f"(including OSID, SAPISID, SID) were detected."
                    )
            finally:
                await browser.close()

        final_cookies = {}
        for k in MANDATORY_COOKIES + OPTIONAL_HARVEST_COOKIES:
            if k in extracted_cookies:
                final_cookies[k] = extracted_cookies[k]

        return GoogleCredentials(
            auth_method=AuthMethod.CDP_BROWSER,
            auth_user_index=auth_user_index,
            user_email=user_email,
            cookies=final_cookies,
        )

    finally:
        terminate_process(proc, timeout=3.0)


async def extract_colab_cookies(timeout_seconds: int = 180, auth_user_index: int = 0) -> Dict[str, str]:
    """Helper returning just the extracted cookies dictionary."""
    creds = await launch_and_extract_cdp_cookies(auth_user_index=auth_user_index, timeout_seconds=timeout_seconds)
    return creds.cookies
