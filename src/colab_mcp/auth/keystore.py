"""
Dual-layer credential persistence: OS Keyring with AES-256-GCM encrypted file fallback.
"""

import base64
import json
import os
from pathlib import Path
import stat
import sys
import uuid
from typing import Any, Dict, List, Optional

import keyring
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from colab_mcp.auth.models import GoogleCredentials, KeystoreError
from colab_mcp.utils.logging import get_logger

logger = get_logger(__name__)

KEYRING_SERVICE_NAME = "google-colab-mcp"
PBKDF2_ITERATIONS = 100_000


def get_credentials_dir() -> Path:
    """Returns directory path where fallback encrypted credentials are stored."""
    return Path(os.path.expanduser("~/.colabmcp"))


def get_credentials_file() -> Path:
    """Returns absolute path to credentials.enc."""
    return get_credentials_dir() / "credentials.enc"


def get_keyring_username(auth_user_index: int) -> str:
    """Formats standard username for OS keyring."""
    return f"colab_auth_credentials_{auth_user_index}"


# ---------------------------------------------------------------------------
# Machine ID Extraction & Key Derivation
# ---------------------------------------------------------------------------

def get_machine_id_bytes() -> bytes:
    """
    Extracts a stable host hardware/OS identifier for PBKDF2 key derivation.
    """
    ident = ""
    try:
        if sys.platform == "linux":
            for path in ["/etc/machine-id", "/var/lib/dbus/machine-id"]:
                if os.path.isfile(path):
                    with open(path, "r", encoding="utf-8") as f:
                        ident = f.read().strip()
                        if ident:
                            break
        elif sys.platform == "win32":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
                val, _ = winreg.QueryValueEx(key, "MachineGuid")
                if val:
                    ident = str(val).strip()
        elif sys.platform == "darwin":
            output = os.popen("ioreg -rd1 -c IOPlatformExpertDevice | grep IOPlatformUUID").read()
            if "IOPlatformUUID" in output:
                ident = output.split('"')[-2].strip()
    except Exception as e:
        logger.debug("Failed to read OS machine identifier: %s", e)

    if not ident:
        ident = f"colabmcp-node-{uuid.getnode()}"

    env_key = os.environ.get("COLAB_MCP_KEY", "")
    return f"{ident}:{env_key}".encode("utf-8")


def derive_aes_key(salt: bytes) -> bytes:
    """
    Derives 256-bit AES key using PBKDF2-HMAC-SHA256 (100,000 iterations)
    from host machine ID and optional COLAB_MCP_KEY environment variable.
    """
    secret_material = get_machine_id_bytes()
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    return kdf.derive(secret_material)


# ---------------------------------------------------------------------------
# Encrypted File Fallback Implementation (AES-256-GCM)
# ---------------------------------------------------------------------------

def enforce_file_permissions(filepath: Path) -> None:
    """Enforces strict 0600 permissions on POSIX systems."""
    if sys.platform != "win32" and filepath.exists():
        os.chmod(filepath, stat.S_IRUSR | stat.S_IWUSR)


def read_encrypted_store() -> Dict[str, Any]:
    """Reads and decrypts fallback credentials file."""
    filepath = get_credentials_file()
    if not filepath.is_file():
        return {}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            envelope = json.load(f)

        salt = base64.b64decode(envelope["salt"])
        nonce = base64.b64decode(envelope["nonce"])
        ciphertext = base64.b64decode(envelope["ciphertext"])

        key = derive_aes_key(salt)
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return json.loads(plaintext.decode("utf-8"))
    except KeystoreError:
        raise
    except Exception as e:
        logger.warning("Failed to decrypt credentials fallback file: %s", e)
        raise KeystoreError(f"Decryption failed: {e}") from e


def write_encrypted_store(store: Dict[str, Any]) -> None:
    """Encrypts and safely writes credentials fallback file with 0600 permissions."""
    cred_dir = get_credentials_dir()
    cred_dir.mkdir(parents=True, exist_ok=True)
    filepath = get_credentials_file()

    salt = os.urandom(16)
    nonce = os.urandom(12)  # 12-byte standard GCM IV
    key = derive_aes_key(salt)
    aesgcm = AESGCM(key)

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
        fd = os.open(str(temp_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(envelope, f)
    else:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(envelope, f)

    os.replace(temp_file, filepath)
    enforce_file_permissions(filepath)


# ---------------------------------------------------------------------------
# Public Keystore API
# ---------------------------------------------------------------------------

def store_credentials(auth_user_index: int, creds: GoogleCredentials) -> None:
    """
    Persists credentials for auth_user_index into OS keyring with encrypted file fallback.
    """
    creds.auth_user_index = auth_user_index
    payload = creds.model_dump_json()
    username = get_keyring_username(auth_user_index)

    keyring_succeeded = False
    try:
        keyring.set_password(KEYRING_SERVICE_NAME, username, payload)
        keyring_succeeded = True
        logger.debug("Credentials successfully persisted in OS keyring for user %d.", auth_user_index)
    except Exception as e:
        logger.info("OS Keyring unavailable (%s); using encrypted file fallback.", e)

    # Always mirror or fallback to encrypted store
    try:
        try:
            store = read_encrypted_store()
        except KeystoreError:
            store = {}
        store[str(auth_user_index)] = creds.model_dump(mode="json")
        write_encrypted_store(store)
    except Exception as e:
        if not keyring_succeeded:
            raise KeystoreError(f"Failed to persist credentials to both Keyring and Encrypted file: {e}") from e


def get_credentials(auth_user_index: int = 0) -> Optional[GoogleCredentials]:
    """
    Retrieves stored credentials for auth_user_index from OS Keyring or encrypted fallback.
    """
    username = get_keyring_username(auth_user_index)

    # 1. Try OS Keyring
    try:
        payload = keyring.get_password(KEYRING_SERVICE_NAME, username)
        if payload:
            data = json.loads(payload)
            return GoogleCredentials.model_validate(data)
    except Exception as e:
        logger.debug("Keyring retrieval failed for user %d: %s", auth_user_index, e)

    # 2. Try Encrypted Fallback file
    filepath = get_credentials_file()
    if filepath.is_file():
        store = read_encrypted_store()
        user_key = str(auth_user_index)
        if user_key in store:
            return GoogleCredentials.model_validate(store[user_key])

    return None


def purge_credentials(auth_user_index: int = 0) -> bool:
    """
    Purges credentials for auth_user_index from both OS Keyring and encrypted file.
    Returns True if credentials were removed.
    """
    username = get_keyring_username(auth_user_index)
    purged = False

    try:
        keyring.delete_password(KEYRING_SERVICE_NAME, username)
        purged = True
    except Exception:
        pass

    filepath = get_credentials_file()
    if filepath.is_file():
        try:
            store = read_encrypted_store()
            user_key = str(auth_user_index)
            if user_key in store:
                del store[user_key]
                write_encrypted_store(store)
                purged = True
        except Exception:
            pass

    return purged


def list_stored_user_indices() -> List[int]:
    """Returns list of all auth_user_index values currently saved in the encrypted store."""
    try:
        store = read_encrypted_store()
        return sorted([int(k) for k in store.keys() if k.isdigit()])
    except Exception:
        return []
