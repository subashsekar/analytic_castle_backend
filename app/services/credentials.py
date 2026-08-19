"""Encrypt and retrieve data-source credentials.

Ciphertext format ``v1:<nonce>:<ciphertext>`` keeps the nonce next to the
payload so AES-256-GCM can decrypt without a separate IV column.

A later ``v2:<key_id>:<nonce>:<ciphertext>`` can introduce key identifiers
for rotation. Existing ``v1`` values would be re-encrypted with the new key
and rewritten. This module does not implement rotation, KMS, or Vault.
"""

from __future__ import annotations

import base64
import binascii
import os
from typing import TYPE_CHECKING

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.connectors.types import ConnectorConfig
from app.core.config import settings

if TYPE_CHECKING:
    from app.db.models.data_source_connection import DataSourceConnection

CIPHERTEXT_VERSION = "v1"
NONCE_SIZE = 12
KEY_SIZE = 32
MAX_SECRET_LENGTH = 512

_B64_ALPHABET = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
)


class CredentialError(Exception):
    """Application-level credential failure. Messages omit crypto details."""


class InvalidEncryptionKeyError(CredentialError):
    """The supplied encryption key cannot be used."""

    def __init__(self, message: str = "Encryption key is invalid") -> None:
        super().__init__(message)


class CredentialEncryptionError(CredentialError):
    """A credential could not be encrypted."""

    def __init__(self, message: str = "Unable to encrypt credential") -> None:
        super().__init__(message)


class CredentialDecryptionError(CredentialError):
    """A credential could not be decrypted."""

    def __init__(self, message: str = "Unable to decrypt credential") -> None:
        super().__init__(message)


def encrypt_secret(value: str, *, key: bytes | None = None) -> str:
    """Encrypt a data-source secret with AES-256-GCM.

    Returns versioned ciphertext. Each call uses a unique nonce, so the
    same plaintext encrypts to a different value every time.
    """
    if not isinstance(value, str) or value == "":
        raise CredentialEncryptionError("Credential cannot be empty")
    if len(value) > MAX_SECRET_LENGTH:
        raise CredentialEncryptionError("Credential exceeds maximum length")

    nonce = os.urandom(NONCE_SIZE)
    plaintext = value.encode("utf-8")
    ciphertext = AESGCM(_resolve_key(key)).encrypt(nonce, plaintext, None)
    return f"{CIPHERTEXT_VERSION}:{_b64encode(nonce)}:{_b64encode(ciphertext)}"


def decrypt_secret(value: str, *, key: bytes | None = None) -> str:
    """Decrypt versioned ciphertext produced by ``encrypt_secret``."""
    if not isinstance(value, str) or value == "":
        raise CredentialDecryptionError("Unable to decrypt credential")

    aes_key = _resolve_key(key)
    version, nonce_b64, payload_b64 = _split_ciphertext(value)
    if version != CIPHERTEXT_VERSION:
        raise CredentialDecryptionError("Unable to decrypt credential")

    try:
        nonce = _b64decode(nonce_b64)
        payload = _b64decode(payload_b64)
    except (ValueError, binascii.Error, UnicodeEncodeError):
        raise CredentialDecryptionError("Unable to decrypt credential") from None
    if len(nonce) != NONCE_SIZE or not payload:
        raise CredentialDecryptionError("Unable to decrypt credential")

    try:
        plaintext = AESGCM(aes_key).decrypt(nonce, payload, None)
    except InvalidTag:
        raise CredentialDecryptionError("Unable to decrypt credential") from None
    except (ValueError, TypeError):
        raise CredentialDecryptionError("Unable to decrypt credential") from None

    try:
        return plaintext.decode("utf-8")
    except UnicodeDecodeError:
        raise CredentialDecryptionError("Unable to decrypt credential") from None


def connector_config_from_connection(
    connection: DataSourceConnection,
    *,
    key: bytes | None = None,
) -> ConnectorConfig:
    """Build connector config by decrypting the stored password in memory.

    The returned ``credential`` is plaintext and must not be persisted.
    """
    password = decrypt_secret(connection.encrypted_password, key=key)
    return ConnectorConfig(
        host=connection.host,
        port=connection.port,
        database_name=connection.database_name,
        username=connection.username,
        credential=password,
        ssl_mode=connection.ssl_mode,
    )


def _resolve_key(key: bytes | None) -> bytes:
    resolved = settings.data_source_encryption_key_bytes if key is None else key
    if len(resolved) != KEY_SIZE:
        raise InvalidEncryptionKeyError("Encryption key is invalid")
    return resolved


def _split_ciphertext(value: str) -> tuple[str, str, str]:
    parts = value.split(":")
    if len(parts) != 3 or not all(parts):
        raise CredentialDecryptionError("Unable to decrypt credential")
    version, nonce_b64, payload_b64 = parts
    if not _is_b64url(nonce_b64) or not _is_b64url(payload_b64):
        raise CredentialDecryptionError("Unable to decrypt credential")
    return version, nonce_b64, payload_b64


def _is_b64url(value: str) -> bool:
    return all(char in _B64_ALPHABET or char == "=" for char in value)


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    padded = value + "=" * ((-len(value)) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))
