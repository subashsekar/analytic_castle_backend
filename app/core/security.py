import hashlib
import hmac
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from app.core.config import settings

TOKEN_TYPE_ACCESS = "access"
TOKEN_TYPE_REFRESH = "refresh"

_hasher = PasswordHasher()


class InvalidTokenError(Exception):
    """Raised when a JWT is malformed, expired, or has the wrong type."""


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def generate_secure_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def hash_refresh_token(token: str) -> str:
    return hash_token(token)


def tokens_match(stored_hash: str, raw_token: str) -> bool:
    return hmac.compare_digest(stored_hash, hash_token(raw_token))


def create_access_token(subject: uuid.UUID) -> str:
    return _encode_token(
        subject,
        token_type=TOKEN_TYPE_ACCESS,
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )


def create_refresh_token(subject: uuid.UUID) -> tuple[str, datetime]:
    expires_delta = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    expires_at = datetime.now(UTC) + expires_delta
    token = _encode_token(
        subject,
        token_type=TOKEN_TYPE_REFRESH,
        expires_delta=expires_delta,
        extra_claims={"jti": str(uuid.uuid4())},
    )
    return token, expires_at


def decode_token(token: str, *, expected_type: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"require": ["sub", "type", "exp", "iat"]},
        )
    except jwt.PyJWTError as exc:
        raise InvalidTokenError("Invalid token") from exc

    if payload.get("type") != expected_type:
        raise InvalidTokenError("Invalid token")

    subject = payload.get("sub")
    if not isinstance(subject, str) or not subject:
        raise InvalidTokenError("Invalid token")
    try:
        uuid.UUID(subject)
    except ValueError as exc:
        raise InvalidTokenError("Invalid token") from exc

    return payload


def _encode_token(
    subject: uuid.UUID,
    *,
    token_type: str,
    expires_delta: timedelta,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(
        payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )


# Valid Argon2id hash used only to keep unknown-email login timing closer to a real verify.
DUMMY_PASSWORD_HASH = hash_password("not-a-real-password")
