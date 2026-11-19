"""
Encrypted user secrets and API token vault for Google Colab notebooks.

Persists arbitrary named secrets (e.g. ``HF_TOKEN``, ``WANDB_API_KEY``,
``OPENAI_API_KEY``) using the dual-layer OS Keyring + AES-256-GCM encrypted
file store and registers their values with the global egress redactor.
"""

from __future__ import annotations

import base64
import json
import os
import re
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import keyring
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel, Field

from colab_mcp.auth import keystore as _keystore
from colab_mcp.auth.models import KeystoreError
from colab_mcp.utils.logging import get_logger
from colab_mcp.utils.redaction import (
    register_dynamic_secret,
    unregister_dynamic_secret,
)

logger = get_logger(__name__)

SECRETS_KEYRING_USERNAME = "colab_user_secrets"
SECRET_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SECRET_INTERPOLATION_PREFIX = "$SECRET:"


class SecretMetadata(BaseModel):
    """Non-sensitive metadata view for a stored user secret."""

    name: str
    masked_preview: str
    length: int
    updated_at: datetime


class SetSecretResponse(BaseModel):
    """Response model for ``colab_set_secret``."""

    status: str = "stored"
    name: str
    masked_preview: str
    length: int
    injected_into_active_kernel: bool = False
    message: str


class ListSecretsResponse(BaseModel):
    """Response model for ``colab_list_secrets``."""

    count: int = 0
    secrets: List[SecretMetadata] = Field(default_factory=list)


class DeleteSecretResponse(BaseModel):
    """Response model for ``colab_delete_secret``."""

    status: str = "deleted"
    name: str
    existed: bool
    removed_from_active_kernel: bool = False
    message: str


def get_secrets_file() -> Path:
    """Return path to encrypted secrets fallback store ``~/.colabmcp/secrets.enc``."""
    return _keystore.get_credentials_dir() / "secrets.enc"


def validate_secret_name(name: str) -> str:
    """Validate that a secret name is a valid environment identifier."""
    if not isinstance(name, str) or not SECRET_NAME_PATTERN.match(name.strip()):
        raise ValueError(
            "Secret name must be a valid identifier matching [A-Za-z_][A-Za-z0-9_]* "
            f"(e.g. 'HF_TOKEN'), got {name!r}."
        )
    return name.strip()


def mask_secret_value(value: str) -> str:
    """Produce a safe masked preview without leaking secret contents."""
    if len(value) <= 6:
        return "*" * len(value)
    return f"{value[:3]}***{value[-2:]}"


def _read_encrypted_secrets_file() -> Dict[str, Dict[str, Any]]:
    filepath = get_secrets_file()
    if not filepath.is_file():
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            envelope = json.load(f)
        salt = base64.b64decode(envelope["salt"])
        nonce = base64.b64decode(envelope["nonce"])
        ciphertext = base64.b64decode(envelope["ciphertext"])
        aesgcm = AESGCM(_keystore.derive_aes_key(salt))
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        data = json.loads(plaintext.decode("utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Failed to decrypt secrets fallback file: %s", exc)
        raise KeystoreError(f"Secret store decryption failed: {exc}") from exc


def _write_encrypted_secrets_file(store: Dict[str, Dict[str, Any]]) -> None:
    cred_dir = _keystore.get_credentials_dir()
    cred_dir.mkdir(parents=True, exist_ok=True)
    filepath = get_secrets_file()

    salt = os.urandom(16)
    nonce = os.urandom(12)
    aesgcm = AESGCM(_keystore.derive_aes_key(salt))
    plaintext = json.dumps(store).encode("utf-8")
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)

    envelope = {
        "version": 1,
        "salt": base64.b64encode(salt).decode("ascii"),
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
    }

    temp_file = cred_dir / f"{filepath.name}.tmp"
    if sys.platform != "win32":
        fd = os.open(
            str(temp_file),
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            stat.S_IRUSR | stat.S_IWUSR,
        )
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(envelope, f)
    else:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(envelope, f)

    os.replace(temp_file, filepath)
    _keystore.enforce_file_permissions(filepath)


def _load_raw_secrets_store() -> Dict[str, Dict[str, Any]]:
    try:
        payload = keyring.get_password(_keystore.KEYRING_SERVICE_NAME, SECRETS_KEYRING_USERNAME)
        if payload:
            data = json.loads(payload)
            if isinstance(data, dict):
                return data
    except Exception as exc:
        logger.debug("OS Keyring secret retrieval unavailable: %s", exc)

    if get_secrets_file().is_file():
        return _read_encrypted_secrets_file()
    return {}


def _save_raw_secrets_store(store: Dict[str, Dict[str, Any]]) -> None:
    payload = json.dumps(store)
    keyring_succeeded = False
    try:
        keyring.set_password(_keystore.KEYRING_SERVICE_NAME, SECRETS_KEYRING_USERNAME, payload)
        keyring_succeeded = True
    except Exception as exc:
        logger.info("OS Keyring unavailable (%s); storing secrets in encrypted fallback.", exc)

    try:
        _write_encrypted_secrets_file(store)
    except Exception as exc:
        if not keyring_succeeded:
            raise KeystoreError(
                f"Failed to persist secrets to both OS Keyring and encrypted fallback: {exc}"
            ) from exc


def set_secret(name: str, value: str) -> SecretMetadata:
    """Validate, encrypt, and persist a named secret in the vault."""
    clean_name = validate_secret_name(name)
    if not isinstance(value, str) or not value:
        raise ValueError("Secret value must be a non-empty string.")

    now = datetime.now(timezone.utc)
    store = _load_raw_secrets_store()
    previous = store.get(clean_name)
    if isinstance(previous, dict) and isinstance(previous.get("value"), str):
        unregister_dynamic_secret(previous["value"])

    store[clean_name] = {
        "value": value,
        "updated_at": now.isoformat(),
    }
    _save_raw_secrets_store(store)
    register_dynamic_secret(value)

    return SecretMetadata(
        name=clean_name,
        masked_preview=mask_secret_value(value),
        length=len(value),
        updated_at=now,
    )


def get_secret(name: str) -> Optional[str]:
    """Retrieve a stored secret value by name, or None if not found."""
    clean_name = validate_secret_name(name)
    store = _load_raw_secrets_store()
    entry = store.get(clean_name)
    if not isinstance(entry, dict):
        return None
    val = entry.get("value")
    if isinstance(val, str):
        register_dynamic_secret(val)
        return val
    return None


def get_all_secrets() -> Dict[str, str]:
    """Return all stored ``{name: value}`` pairs and register them for egress redaction."""
    store = _load_raw_secrets_store()
    result: Dict[str, str] = {}
    for name, entry in store.items():
        if isinstance(entry, dict) and isinstance(entry.get("value"), str):
            val = entry["value"]
            register_dynamic_secret(val)
            result[str(name)] = val
    return result


def list_secrets() -> List[SecretMetadata]:
    """List metadata and masked previews for all stored secrets (never plaintext)."""
    store = _load_raw_secrets_store()
    items: List[SecretMetadata] = []
    for name in sorted(store.keys()):
        entry = store.get(name)
        if not isinstance(entry, dict):
            continue
        val = entry.get("value")
        if not isinstance(val, str):
            continue
        register_dynamic_secret(val)
        raw_ts = entry.get("updated_at")
        try:
            ts = datetime.fromisoformat(str(raw_ts)) if raw_ts else datetime.now(timezone.utc)
        except ValueError:
            ts = datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        items.append(
            SecretMetadata(
                name=str(name),
                masked_preview=mask_secret_value(val),
                length=len(val),
                updated_at=ts,
            )
        )
    return items


def delete_secret(name: str) -> bool:
    """Remove a stored secret by name. Returns True if it existed."""
    clean_name = validate_secret_name(name)
    store = _load_raw_secrets_store()
    if clean_name not in store:
        return False
    entry = store.pop(clean_name)
    if isinstance(entry, dict) and isinstance(entry.get("value"), str):
        unregister_dynamic_secret(entry["value"])
    _save_raw_secrets_store(store)
    return True


def resolve_secret_interpolation(value: str) -> str:
    """
    Resolve ``$SECRET:<NAME>`` placeholders to the stored secret value.

    Allows agents and notebooks to supply stored tokens via ``stdin_inputs`` or
    ``colab_send_input`` without placing raw secret strings inside tool payloads.
    """
    if not isinstance(value, str):
        raise ValueError("value must be a string.")
    if not value.startswith(SECRET_INTERPOLATION_PREFIX):
        return value
    secret_name = value[len(SECRET_INTERPOLATION_PREFIX):].strip()
    resolved = get_secret(secret_name)
    if resolved is None:
        raise ValueError(
            f"Referenced secret {secret_name!r} was not found in the colab-mcp secret vault. "
            f"Store it first with colab_set_secret(name={secret_name!r}, value=...)."
        )
    return resolved


__all__ = [
    "DeleteSecretResponse",
    "ListSecretsResponse",
    "SECRET_INTERPOLATION_PREFIX",
    "SecretMetadata",
    "SetSecretResponse",
    "delete_secret",
    "get_all_secrets",
    "get_secret",
    "get_secrets_file",
    "list_secrets",
    "mask_secret_value",
    "resolve_secret_interpolation",
    "set_secret",
    "validate_secret_name",
]
