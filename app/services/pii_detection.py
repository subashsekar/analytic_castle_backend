"""Conservative PII classification from column metadata and obvious values.

Column names, types, and descriptions are the primary signal. Value-level
checks only match strong, whole-value patterns and never send data to an LLM.
"""

from __future__ import annotations

import re
from enum import Enum

from app.enums import ColumnSensitivity

_SEPARATORS = re.compile(r"[^a-z0-9]+")
_EMAIL_VALUE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_SEPARATORS = re.compile(r"[\s().+\-]")
_CARD_SEPARATORS = re.compile(r"[\s\-]")

SECRET_NAME_PATTERNS = frozenset(
    {
        "password",
        "passwd",
        "password_hash",
        "secret",
        "token",
        "api_key",
        "access_token",
        "refresh_token",
        "private_key",
    }
)
PII_EMAIL_PATTERNS = frozenset({"email", "email_address", "e_mail"})
PII_PHONE_PATTERNS = frozenset({"phone", "phone_number", "mobile", "mobile_number"})
PII_NAME_PATTERNS = frozenset({"first_name", "last_name", "full_name", "given_name"})
PII_ADDRESS_PATTERNS = frozenset(
    {"address", "street_address", "home_address", "mailing_address"}
)
PII_ID_PATTERNS = frozenset(
    {
        "ssn",
        "social_security",
        "aadhaar",
        "pan",
        "passport",
        "passport_number",
        "date_of_birth",
        "dob",
        "credit_card",
        "card_number",
        "account_number",
    }
)
PII_NAME_PATTERNS_ALL = (
    PII_EMAIL_PATTERNS
    | PII_PHONE_PATTERNS
    | PII_NAME_PATTERNS
    | PII_ADDRESS_PATTERNS
    | PII_ID_PATTERNS
)
SENSITIVE_NAME_PATTERNS: frozenset[str] = frozenset()


class MaskKind(str, Enum):
    EMAIL = "email"
    PHONE = "phone"
    CARD = "card"
    NAME = "name"
    SECRET = "secret"
    GENERIC = "generic"


def classify_column(
    name: str,
    *,
    data_type: str = "",
    description: str | None = None,
) -> ColumnSensitivity:
    """Classify a column from metadata. Names are the strongest signal."""
    if _matches_patterns(name, SECRET_NAME_PATTERNS) or _description_matches(
        description, SECRET_NAME_PATTERNS
    ):
        return ColumnSensitivity.SECRET
    if _matches_patterns(name, PII_NAME_PATTERNS_ALL) or _description_matches(
        description, PII_NAME_PATTERNS_ALL
    ):
        return ColumnSensitivity.PII
    if _matches_patterns(name, SENSITIVE_NAME_PATTERNS) or _description_matches(
        description, SENSITIVE_NAME_PATTERNS
    ):
        return ColumnSensitivity.SENSITIVE
    _ = data_type
    return ColumnSensitivity.PUBLIC


def mask_kind_for_column(name: str) -> MaskKind:
    if _matches_patterns(name, SECRET_NAME_PATTERNS):
        return MaskKind.SECRET
    if _matches_patterns(name, PII_EMAIL_PATTERNS):
        return MaskKind.EMAIL
    if _matches_patterns(name, PII_PHONE_PATTERNS):
        return MaskKind.PHONE
    if _matches_patterns(name, frozenset({"credit_card", "card_number"})):
        return MaskKind.CARD
    if _matches_patterns(name, PII_NAME_PATTERNS):
        return MaskKind.NAME
    return MaskKind.GENERIC


def detect_value_pii(value: object) -> MaskKind | None:
    """Return a mask kind when the whole value is an obvious PII pattern.

    Ordinary integers, IDs, and partial text matches are ignored.
    """
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or len(stripped) > 256:
        return None
    if _EMAIL_VALUE.fullmatch(stripped):
        return MaskKind.EMAIL
    if _looks_like_card(stripped):
        return MaskKind.CARD
    if _looks_like_phone(stripped):
        return MaskKind.PHONE
    return None


def _matches_patterns(name: str, patterns: frozenset[str]) -> bool:
    normalized = _normalize_identifier(name)
    if not normalized:
        return False
    compact = normalized.replace("_", "")
    compact_patterns = {pattern.replace("_", "") for pattern in patterns}
    if normalized in patterns or compact in compact_patterns:
        return True
    tokens = tuple(part for part in normalized.split("_") if part)
    for pattern in patterns:
        pattern_tokens = tuple(part for part in pattern.split("_") if part)
        if not pattern_tokens:
            continue
        if _contains_token_sequence(tokens, pattern_tokens):
            return True
    return False


def _description_matches(description: str | None, patterns: frozenset[str]) -> bool:
    if description is None or not description.strip():
        return False
    return _matches_patterns(description, patterns)


def _normalize_identifier(value: str) -> str:
    lowered = value.strip().lower()
    collapsed = _SEPARATORS.sub("_", lowered).strip("_")
    return collapsed


def _contains_token_sequence(
    tokens: tuple[str, ...],
    pattern_tokens: tuple[str, ...],
) -> bool:
    if len(pattern_tokens) > len(tokens):
        return False
    span = len(pattern_tokens)
    for index in range(len(tokens) - span + 1):
        if tokens[index : index + span] == pattern_tokens:
            return True
    return False


def _looks_like_phone(value: str) -> bool:
    if _EMAIL_VALUE.fullmatch(value):
        return False
    compact = _PHONE_SEPARATORS.sub("", value)
    if not compact.isdigit():
        return False
    return 10 <= len(compact) <= 15


def _looks_like_card(value: str) -> bool:
    compact = _CARD_SEPARATORS.sub("", value)
    if not compact.isdigit() or not 13 <= len(compact) <= 19:
        return False
    return _luhn_ok(compact)


def _luhn_ok(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        number = ord(char) - 48
        if index % 2 == 1:
            number *= 2
            if number > 9:
                number -= 9
        total += number
    return total % 10 == 0
