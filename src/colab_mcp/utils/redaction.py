"""
Credential Redaction Interceptor.
Enforces zero-leakage of Google OAuth tokens, session cookies, and SAPISID signatures.
"""

import logging
import re
from typing import Any

# Sensitive keys for Google cookies and tokens
SENSITIVE_KEYS: frozenset[str] = frozenset({
    "SAPISID",
    "APISID",
    "HSID",
    "SSID",
    "SID",
    "__Secure-1PSID",
    "__Secure-3PSID",
    "OSID",
    "__Secure-OSID",
    "access_token",
    "refresh_token",
})

_KEYS_REGEX: str = (
    r"SAPISID|APISID|HSID|SSID|SID|__Secure-1PSID|__Secure-3PSID|OSID|__Secure-OSID|access_token|refresh_token"
)

# Pattern matching JSON and dict style key-value pairs:
# e.g. "SAPISID": "secret" or 'OSID': 'secret'
# Preserves keys, quotes, and colons while redacting only the value.
JSON_KEY_VALUE_PATTERN: re.Pattern = re.compile(
    r'(?P<kv_q1>["\'])'
    rf'(?P<kv_key>{_KEYS_REGEX})'
    r'(?P=kv_q1)'
    r'(?P<kv_sep>\s*:\s*)'
    r'(?P<kv_q2>["\'])'
    r'(?P<kv_val>[^"\'\r\n\\]*(?:\\.[^"\'\r\n\\]*)*)'
    r'(?P=kv_q2)'
)

CREDENTIAL_REDACTION_PATTERN: re.Pattern = re.compile(
    r'(?:'
    r'ya29\.[a-zA-Z0-9_\.\-]+'                                                                  # OAuth 2.0 access tokens
    r'|1//0[a-zA-Z0-9_\.\-/]+'                                                                 # Google OAuth refresh tokens
    r'|Bearer\s+[a-zA-Z0-9_\.\-]+'                                                             # Generic Bearer auth headers
    rf'|\b(?:{_KEYS_REGEX})=(?:\\"[^"\\\r\n]*\\"|"[^"\r\n]*"|\'[^\'\r\n]*\'|[a-zA-Z0-9_\.\-%+=/]+)'  # HTTP header cookies & params
    r'|SAPISIDHASH\s+\d+_[a-fA-F0-9]+'                                                          # SAPISIDHASH auth headers
    r')'
)

REDACTED_SUBSTITUTION = "[REDACTED]"


def redact_credentials(text: str) -> str:
    """
    Scans input text and replaces sensitive Google tokens, session cookies,
    and authorization signatures with '[REDACTED]'.
    Preserves JSON / dict structures and non-sensitive properties.
    """
    if not isinstance(text, str):
        return text
    # First redact JSON and dict key-value pairs (preserving keys, quotes, and colons)
    text = JSON_KEY_VALUE_PATTERN.sub(
        r'\g<kv_q1>\g<kv_key>\g<kv_q1>\g<kv_sep>\g<kv_q2>[REDACTED]\g<kv_q2>',
        text,
    )
    # Then redact standalone tokens, Bearer headers, SAPISIDHASH, and HTTP header cookies
    return CREDENTIAL_REDACTION_PATTERN.sub(REDACTED_SUBSTITUTION, text)


def redact_data(data: Any) -> Any:
    """
    Recursively scans and redacts credentials from strings within nested data structures
    (dictionaries, lists, tuples, sets, frozensets, Pydantic models, or primitives).
    """
    if hasattr(data, "model_dump") and callable(getattr(data, "model_dump")) and not isinstance(data, type):
        return redact_data(data.model_dump())
    elif isinstance(data, str):
        return redact_credentials(data)
    elif isinstance(data, dict):
        result = {}
        for k, v in data.items():
            k_str = str(k)
            if k_str in SENSITIVE_KEYS and isinstance(v, str):
                result[k] = REDACTED_SUBSTITUTION
            elif k_str == "cookies" and isinstance(v, dict):
                redacted_cookies = {}
                for ck, cv in v.items():
                    if str(ck) in SENSITIVE_KEYS and isinstance(cv, str):
                        redacted_cookies[ck] = REDACTED_SUBSTITUTION
                    else:
                        redacted_cookies[ck] = redact_data(cv)
                result[k] = redacted_cookies
            else:
                result[k] = redact_data(v)
        return result
    elif isinstance(data, list):
        return [redact_data(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(redact_data(item) for item in data)
    elif isinstance(data, set):
        return {redact_data(item) for item in data}
    elif isinstance(data, frozenset):
        return frozenset(redact_data(item) for item in data)
    return data


class RedactingFormatter(logging.Formatter):
    """
    Logging formatter that automatically applies credential redaction to all log records.
    """

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        return redact_credentials(original)
