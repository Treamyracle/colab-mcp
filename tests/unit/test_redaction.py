"""
Unit tests for credential redaction and zero-leakage security policy.
"""

import ast
import json
from colab_mcp.utils.redaction import redact_credentials, redact_data


def test_redact_oauth_access_token():
    text = "ya29.a0AfH6SMA_secret_token_123456"
    assert redact_credentials(text) == "[REDACTED]"


def test_redact_bearer_token():
    text = "Bearer ya29.a0AfH6SMA_secret_token_123456"
    assert redact_credentials(text) == "[REDACTED]"

    text_generic = "Bearer metadata_adc_non_ya29_token"
    assert redact_credentials(text_generic) == "[REDACTED]"


def test_redact_oauth_refresh_token():
    text = "Refresh token: 1//04abcDEF_refresh-secret_xyz"
    assert redact_credentials(text) == "Refresh token: [REDACTED]"


def test_redact_cookie_sapisid():
    text = "SAPISID=abcd1234xyz;"
    assert redact_credentials(text) == "[REDACTED];"


def test_redact_cookie_sid_and_osid():
    text = "Cookie: SID=secret_sid_123; OSID=secret_osid_456"
    assert redact_credentials(text) == "Cookie: [REDACTED]; [REDACTED]"


def test_redact_cookie_with_url_chars():
    text = "Cookie: SID=abc+def=ghi%2B; OSID=secret+value=with%20encoding"
    redacted = redact_credentials(text)
    assert "abc+def=ghi%2B" not in redacted
    assert "secret+value=with%20encoding" not in redacted
    assert "Cookie: [REDACTED]; [REDACTED]" == redacted


def test_redact_json_cookie():
    payload = '{"cookies": {"SAPISID": "secret_sapisid_123"}}'
    redacted = redact_credentials(payload)
    assert "secret_sapisid_123" not in redacted
    assert "[REDACTED]" in redacted
    parsed = json.loads(redacted)
    assert parsed == {"cookies": {"SAPISID": "[REDACTED]"}}


def test_redact_dict_cookie():
    payload = "{'OSID': 'secret+osid%123=='}"
    redacted = redact_credentials(payload)
    assert "secret+osid%123==" not in redacted
    assert "[REDACTED]" in redacted
    parsed = ast.literal_eval(redacted)
    assert parsed == {"OSID": "[REDACTED]"}


def test_redact_json_cookie_mixed_preserves_structure():
    payload = '{"cookies": {"SAPISID": "secret123", "theme": "dark"}}'
    redacted = redact_credentials(payload)
    assert "secret123" not in redacted
    parsed = json.loads(redacted)
    assert parsed == {"cookies": {"SAPISID": "[REDACTED]", "theme": "dark"}}


def test_redact_dict_cookie_with_non_sensitive_fields():
    payload = "{'OSID': 'secret789', 'user': 'alice'}"
    redacted = redact_credentials(payload)
    assert "secret789" not in redacted
    parsed = ast.literal_eval(redacted)
    assert parsed == {"OSID": "[REDACTED]", "user": "alice"}


def test_redact_data_dict_cookies():
    d = {"cookies": {"SAPISID": "secret123", "theme": "dark"}}
    redacted = redact_data(d)
    assert redacted == {"cookies": {"SAPISID": "[REDACTED]", "theme": "dark"}}


def test_redact_data_direct_sensitive_key():
    d = {"OSID": "secret789", "user": "alice", "access_token": "token123"}
    redacted = redact_data(d)
    assert redacted == {"OSID": "[REDACTED]", "user": "alice", "access_token": "[REDACTED]"}


def test_redact_authorization_bearer_header():
    text = "Authorization: Bearer ya29.a0AfH6SMB_some_token"
    assert redact_credentials(text) == "Authorization: [REDACTED]"

    text_generic = "Authorization: Bearer metadata_service_adc_token"
    assert redact_credentials(text_generic) == "Authorization: [REDACTED]"


def test_redact_in_json_string():
    payload = json.dumps({"token": "ya29.a0AfH6SMB_secret", "normal": "hello"})
    redacted = redact_credentials(payload)
    parsed = json.loads(redacted)
    assert parsed["token"] == "[REDACTED]"
    assert parsed["normal"] == "hello"


def test_redact_in_multiline_traceback():
    traceback = (
        "Traceback (most recent call last):\n"
        "  File 'auth.py', line 42, in get_headers\n"
        "    request(headers={'Authorization': 'Bearer ya29.secret_token'})\n"
        "RuntimeError: Failed with SID=my_secret_sid\n"
    )
    redacted = redact_credentials(traceback)
    assert "ya29.secret_token" not in redacted
    assert "my_secret_sid" not in redacted
    assert "[REDACTED]" in redacted


def test_no_false_positives_on_normal_strings():
    text = "https://colab.research.google.com/tun/m/colab-abc?authuser=0 in year2026 with id 123456"
    assert redact_credentials(text) == text


def test_redact_sapisidhash_header():
    text = "Authorization: SAPISIDHASH 1700000000_abcdef1234567890abcdef1234567890abcdef12"
    assert redact_credentials(text) == "Authorization: [REDACTED]"


def test_redaction_idempotence():
    text = "Bearer ya29.a0AfH6SMA_token"
    first = redact_credentials(text)
    second = redact_credentials(first)
    assert first == "[REDACTED]"
    assert second == first


def test_redact_data_nested_structures():
    data = {
        "user": "test@example.com",
        "auth": {"token": "ya29.xyz123"},
        "cookie_list": ["OSID=12345", "clean_cookie=abc"],
        "tuple_items": ("SID=secret", 42),
    }
    redacted = redact_data(data)
    assert redacted["user"] == "test@example.com"
    assert redacted["auth"]["token"] == "[REDACTED]"
    assert redacted["cookie_list"][0] == "[REDACTED]"
    assert redacted["cookie_list"][1] == "clean_cookie=abc"
    assert redacted["tuple_items"][0] == "[REDACTED]"
    assert redacted["tuple_items"][1] == 42


def test_redact_data_non_string_primitives():
    assert redact_data(12345) == 12345
    assert redact_data(3.14159) == 3.14159
    assert redact_data(True) is True
    assert redact_data(None) is None


def test_redos_safety_unclosed_quotes_with_backslashes():
    import time

    # An unclosed quote followed by 100 backslashes must execute instantly (<0.01s)
    malicious = '"SAPISID": "' + ("\\" * 100)
    start = time.perf_counter()
    result = redact_credentials(malicious)
    duration = time.perf_counter() - start

    assert duration < 0.01, f"ReDoS vulnerability detected! Execution took {duration:.4f}s"
    assert "SAPISID" in result

    # Additional test with Windows path-like backslashes
    path_backslashes = '"access_token": "' + ("C:\\\\" * 50)
    start = time.perf_counter()
    result_path = redact_credentials(path_backslashes)
    duration_path = time.perf_counter() - start
    assert duration_path < 0.01, f"ReDoS on Windows paths detected! Took {duration_path:.4f}s"


def test_redact_refresh_token_query_string():
    # Standalone query param
    text = "refresh_token=1//04abcDEF_refresh-secret_xyz"
    redacted = redact_credentials(text)
    assert redacted == "[REDACTED]"
    assert "04abcDEF" not in redacted
    assert "//0" not in redacted

    # In URL query string
    url = "https://oauth2.googleapis.com/token?grant_type=refresh_token&refresh_token=1//04abcDEF_refresh-secret_xyz"
    redacted_url = redact_credentials(url)
    assert "04abcDEF" not in redacted_url
    assert "//0" not in redacted_url
    assert "grant_type=refresh_token&[REDACTED]" in redacted_url

    # In form body
    body = "grant_type=refresh_token&refresh_token=1//04abcDEF"
    assert redact_credentials(body) == "grant_type=refresh_token&[REDACTED]"


def test_redact_base64_cookie_with_slashes():
    text = "Cookie: SID=AA/BB==; OSID=CC/DD"
    redacted = redact_credentials(text)
    assert redacted == "Cookie: [REDACTED]; [REDACTED]"
    assert "/BB==" not in redacted
    assert "/DD" not in redacted
    assert "AA" not in redacted


def test_redact_stringified_json_cookie_preserves_syntax():
    payload = json.dumps({"Cookie": "SID=mysecretval"})
    redacted = redact_credentials(payload)
    assert "mysecretval" not in redacted
    assert redacted == '{"Cookie": "[REDACTED]"}'
    parsed = json.loads(redacted)
    assert parsed == {"Cookie": "[REDACTED]"}


def test_redact_stringified_python_dict_cookie_preserves_syntax():
    payload = str({"Cookie": "SID=mysecretval"})
    redacted = redact_credentials(payload)
    assert "mysecretval" not in redacted
    parsed = ast.literal_eval(redacted)
    assert parsed == {"Cookie": "[REDACTED]"}


def test_redact_escaped_quoted_cookie_in_json():
    payload = r'{"Cookie": "SID=\"mysecretval\""}'
    redacted = redact_credentials(payload)
    assert "mysecretval" not in redacted
    parsed = json.loads(redacted)
    assert parsed == {"Cookie": "[REDACTED]"}


def test_redact_data_set_and_frozenset():
    # Set of credentials and normal strings
    s = {"ya29.secret_token_123", "normal_value", "Cookie: SID=mysecretval"}
    redacted_set = redact_data(s)
    assert isinstance(redacted_set, set)
    assert "normal_value" in redacted_set
    assert "ya29.secret_token_123" not in redacted_set
    assert "[REDACTED]" in redacted_set
    assert "Cookie: [REDACTED]" in redacted_set

    # Frozenset
    fs = frozenset({"1//04abcDEF_secret", 42})
    redacted_fs = redact_data(fs)
    assert isinstance(redacted_fs, frozenset)
    assert "[REDACTED]" in redacted_fs
    assert 42 in redacted_fs
    assert "1//04abcDEF_secret" not in redacted_fs


def test_redact_data_pydantic_model():
    from colab_mcp.auth.models import GoogleCredentials

    creds = GoogleCredentials(
        access_token="ya29.secret_access_token_123",
        refresh_token="1//0secret_refresh_token_456",
        cookies={"SAPISID": "cookie_secret_789", "theme": "dark"},
        user_email="test@example.com",
    )
    redacted = redact_data(creds)
    assert isinstance(redacted, dict)
    assert redacted["access_token"] == "[REDACTED]"
    assert redacted["refresh_token"] == "[REDACTED]"
    assert redacted["cookies"]["SAPISID"] == "[REDACTED]"
    assert redacted["cookies"]["theme"] == "dark"
    assert redacted["user_email"] == "test@example.com"

