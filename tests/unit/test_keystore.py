"""
Unit tests for dual-layer credential keystore (OS Keyring and AES-256-GCM fallback).
"""

import base64
import json
import os
import sys
from pathlib import Path
import pytest

from colab_mcp.auth.keystore import (
    derive_aes_key,
    get_credentials,
    get_credentials_file,
    list_stored_user_indices,
    purge_credentials,
    read_encrypted_store,
    store_credentials,
    write_encrypted_store,
)
from colab_mcp.auth.models import GoogleCredentials, KeystoreError


def test_keyring_store_and_retrieve(mock_keystore_dir, mock_keyring, mock_machine_id, sample_credentials):
    store_credentials(0, sample_credentials)
    retrieved = get_credentials(0)
    assert retrieved is not None
    assert retrieved.access_token == sample_credentials.access_token
    assert retrieved.refresh_token == sample_credentials.refresh_token
    assert retrieved.user_email == sample_credentials.user_email
    assert retrieved.cookies == sample_credentials.cookies


def test_keyring_purge_success(mock_keystore_dir, mock_keyring, mock_machine_id, sample_credentials):
    store_credentials(0, sample_credentials)
    assert get_credentials(0) is not None
    purged = purge_credentials(0)
    assert purged is True
    assert get_credentials(0) is None


def test_multi_account_isolation(mock_keystore_dir, mock_keyring, mock_machine_id, sample_credentials):
    creds_user0 = sample_credentials.model_copy(update={"user_email": "user0@example.com"})
    creds_user1 = sample_credentials.model_copy(update={"user_email": "user1@example.com", "access_token": "token_user1"})

    store_credentials(0, creds_user0)
    store_credentials(1, creds_user1)

    loaded0 = get_credentials(0)
    loaded1 = get_credentials(1)

    assert loaded0 is not None
    assert loaded1 is not None
    assert loaded0.user_email == "user0@example.com"
    assert loaded1.user_email == "user1@example.com"
    assert loaded0.access_token != loaded1.access_token


def test_encrypted_file_fallback_on_keyring_failure(mock_keystore_dir, mock_machine_id, monkeypatch, sample_credentials):
    import keyring
    def failing_set_password(*args, **kwargs):
        raise RuntimeError("OS Keyring service is unavailable")

    def failing_get_password(*args, **kwargs):
        raise RuntimeError("OS Keyring service is unavailable")

    monkeypatch.setattr(keyring, "set_password", failing_set_password)
    monkeypatch.setattr(keyring, "get_password", failing_get_password)

    store_credentials(0, sample_credentials)
    fallback_file = get_credentials_file()
    assert fallback_file.is_file()

    loaded = get_credentials(0)
    assert loaded is not None
    assert loaded.access_token == sample_credentials.access_token


def test_encrypted_file_decrypt_roundtrip(mock_keystore_dir, mock_machine_id, sample_credentials):
    store = {"0": sample_credentials.model_dump(mode="json")}
    write_encrypted_store(store)

    fallback_file = get_credentials_file()
    assert fallback_file.is_file()

    read_back = read_encrypted_store()
    assert "0" in read_back
    assert read_back["0"]["access_token"] == sample_credentials.access_token


def test_posix_0600_permissions_enforced(mock_keystore_dir, mock_machine_id, sample_credentials):
    store_credentials(0, sample_credentials)
    fallback_file = get_credentials_file()
    if sys.platform != "win32":
        mode = fallback_file.stat().st_mode & 0o777
        assert mode == 0o600


def test_corrupted_ciphertext_raises_error(mock_keystore_dir, mock_machine_id, sample_credentials):
    store = {"0": sample_credentials.model_dump(mode="json")}
    write_encrypted_store(store)

    fallback_file = get_credentials_file()
    with open(fallback_file, "r", encoding="utf-8") as f:
        envelope = json.load(f)

    # Corrupt the ciphertext bytes
    raw_ct = base64.b64decode(envelope["ciphertext"])
    corrupted_ct = bytearray(raw_ct)
    corrupted_ct[0] ^= 0xFF
    envelope["ciphertext"] = base64.b64encode(corrupted_ct).decode("ascii")

    with open(fallback_file, "w", encoding="utf-8") as f:
        json.dump(envelope, f)

    with pytest.raises(KeystoreError):
        read_encrypted_store()


def test_tampered_auth_tag_rejection(mock_keystore_dir, mock_machine_id, sample_credentials):
    store = {"0": sample_credentials.model_dump(mode="json")}
    write_encrypted_store(store)

    fallback_file = get_credentials_file()
    with open(fallback_file, "r", encoding="utf-8") as f:
        envelope = json.load(f)

    # AESGCM tag is the last 16 bytes of the ciphertext
    raw_ct = bytearray(base64.b64decode(envelope["ciphertext"]))
    raw_ct[-1] ^= 0x01  # Flip 1 bit in authentication tag
    envelope["ciphertext"] = base64.b64encode(raw_ct).decode("ascii")

    with open(fallback_file, "w", encoding="utf-8") as f:
        json.dump(envelope, f)

    with pytest.raises(KeystoreError):
        read_encrypted_store()


def test_purge_nonexistent_returns_false(mock_keystore_dir, mock_keyring, mock_machine_id):
    assert purge_credentials(999) is False


def test_different_machine_id_fails_decryption(mock_keystore_dir, mock_machine_id, monkeypatch, sample_credentials):
    store = {"0": sample_credentials.model_dump(mode="json")}
    write_encrypted_store(store)

    # Change machine ID
    monkeypatch.setattr("colab_mcp.auth.keystore.get_machine_id_bytes", lambda: b"DIFFERENT_MACHINE_ID")

    with pytest.raises(KeystoreError):
        read_encrypted_store()


def test_list_stored_user_indices(mock_keystore_dir, mock_keyring, mock_machine_id, sample_credentials):
    store_credentials(0, sample_credentials)
    store_credentials(2, sample_credentials)
    store_credentials(5, sample_credentials)

    indices = list_stored_user_indices()
    assert indices == [0, 2, 5]
