"""
Google 1P API request signing via SAPISIDHASH header calculation.
"""

import hashlib
import time
from typing import Dict, Optional

DEFAULT_COLAB_ORIGIN = "https://colab.research.google.com"


def generate_sapisid_hash(
    sapisid: str,
    origin: str = DEFAULT_COLAB_ORIGIN,
    timestamp: Optional[int | float] = None,
) -> str:
    """
    Computes Google 1P authorization header using unix epoch seconds:
    Authorization: SAPISIDHASH <timestamp_sec>_<sha1(timestamp_sec + ' ' + sapisid + ' ' + origin)>
    """
    if sapisid is None:
        raise ValueError("sapisid cannot be None")
    if not isinstance(sapisid, str):
        raise TypeError(f"sapisid must be a string, got {type(sapisid).__name__}")
    sapisid_clean = sapisid.strip()
    if not sapisid_clean:
        raise ValueError("sapisid must be a non-empty, non-whitespace string")

    if timestamp is None:
        ts_sec = int(time.time())
    else:
        ts_sec = int(timestamp)

    origin_clean = origin.strip() if origin else DEFAULT_COLAB_ORIGIN
    payload = f"{ts_sec} {sapisid_clean} {origin_clean}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"SAPISIDHASH {ts_sec}_{digest}"


def get_sapisid_auth_headers(
    cookies: Dict[str, str],
    origin: str = DEFAULT_COLAB_ORIGIN,
    auth_user_index: int = 0,
) -> Dict[str, str]:
    """
    Builds the full HTTP headers dictionary required for Google Colab internal endpoints
    when authenticating via session cookies.
    """
    sapisid = cookies.get("SAPISID")
    if not sapisid:
        raise ValueError("Cannot calculate SAPISIDHASH: 'SAPISID' cookie is missing.")

    auth_header = generate_sapisid_hash(sapisid, origin=origin)
    cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())

    return {
        "Authorization": auth_header,
        "Cookie": cookie_str,
        "Origin": origin,
        "X-Goog-AuthUser": str(auth_user_index),
        "X-Colab-Client-Agent": "colab-mcp",
    }
