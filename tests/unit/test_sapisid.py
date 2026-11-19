"""
Unit tests for SAPISID request signing and header generation.
"""

import hashlib
import re
import time
import pytest

from colab_mcp.auth.sapisid import (
    DEFAULT_COLAB_ORIGIN,
    generate_sapisid_hash,
    get_sapisid_auth_headers,
)


def test_sapisidhash_known_test_vector():
    timestamp = 1700000000
    sapisid = "test_sapisid_token"
    origin = "https://colab.research.google.com"
    payload = f"{timestamp} {sapisid} {origin}"
    expected_digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    expected_header = f"SAPISIDHASH {timestamp}_{expected_digest}"

    result = generate_sapisid_hash(sapisid, origin=origin, timestamp=timestamp)
    assert result == expected_header


def test_sapisidhash_header_structure():
    result = generate_sapisid_hash("any_token")
    assert result.startswith("SAPISIDHASH ")
    parts = result.split(" ")[1].split("_")
    assert len(parts) == 2
    assert parts[0].isdigit()
    assert len(parts[1]) == 40


def test_sapisidhash_default_origin():
    timestamp = 1700000000
    sapisid = "my_token"
    expected = generate_sapisid_hash(sapisid, origin=DEFAULT_COLAB_ORIGIN, timestamp=timestamp)
    result = generate_sapisid_hash(sapisid, timestamp=timestamp)
    assert result == expected


def test_sapisidhash_custom_origin():
    timestamp = 1700000000
    sapisid = "my_token"
    origin = "https://drive.google.com"
    expected_digest = hashlib.sha1(f"{timestamp} {sapisid} {origin}".encode("utf-8")).hexdigest()
    result = generate_sapisid_hash(sapisid, origin=origin, timestamp=timestamp)
    assert result == f"SAPISIDHASH {timestamp}_{expected_digest}"


def test_sapisidhash_epoch_timestamp_auto():
    now_before = int(time.time())
    result = generate_sapisid_hash("test_token")
    now_after = int(time.time())
    ts_str = result.split(" ")[1].split("_")[0]
    ts = int(ts_str)
    assert now_before <= ts <= now_after


def test_sapisidhash_empty_string_raises():
    with pytest.raises(ValueError):
        generate_sapisid_hash("")


def test_sapisidhash_none_raises():
    with pytest.raises((ValueError, TypeError)):
        generate_sapisid_hash(None)


def test_sapisidhash_whitespace_only():
    with pytest.raises(ValueError):
        generate_sapisid_hash("   \t\n  ")


def test_sapisidhash_digest_is_lowercase_hex():
    result = generate_sapisid_hash("mixed_CASE_Token_123")
    digest = result.split("_")[1]
    assert re.fullmatch(r"[0-9a-f]{40}", digest) is not None


def test_sapisidhash_float_timestamp_truncation():
    result = generate_sapisid_hash("test", timestamp=1700000000.876)
    assert result.startswith("SAPISIDHASH 1700000000_")


def test_get_sapisid_auth_headers_success():
    cookies = {
        "SAPISID": "test_sapisid_123",
        "SID": "test_sid_456",
        "OSID": "test_osid_789",
    }
    headers = get_sapisid_auth_headers(cookies=cookies, auth_user_index=2)
    assert "Authorization" in headers
    assert headers["Authorization"].startswith("SAPISIDHASH ")
    assert headers["Origin"] == DEFAULT_COLAB_ORIGIN
    assert headers["X-Goog-AuthUser"] == "2"
    assert headers["X-Colab-Client-Agent"] == "colab-mcp"
    assert "SAPISID=test_sapisid_123" in headers["Cookie"]
    assert "SID=test_sid_456" in headers["Cookie"]
    assert "OSID=test_osid_789" in headers["Cookie"]


def test_get_sapisid_auth_headers_missing_sapisid_raises():
    cookies = {"SID": "test_sid_456"}
    with pytest.raises(ValueError, match="SAPISID"):
        get_sapisid_auth_headers(cookies)
