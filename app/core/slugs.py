import re
import unicodedata
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

_SLUG_MAX_LENGTH = 255
_INVALID_SLUG_CHARS = re.compile(r"[^a-z0-9]+")
_REPEAT_HYPHENS = re.compile(r"-{2,}")


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = _INVALID_SLUG_CHARS.sub("-", ascii_text.lower()).strip("-")
    slug = _REPEAT_HYPHENS.sub("-", slug)
    return slug[:_SLUG_MAX_LENGTH]


def unique_slug(
    db: Session,
    slug_column: Any,
    base: str,
    *filters: Any,
) -> str:
    candidate = base[:_SLUG_MAX_LENGTH]
    suffix = 2
    while _slug_exists(db, slug_column, candidate, *filters):
        suffix_text = f"-{suffix}"
        candidate = f"{base[: _SLUG_MAX_LENGTH - len(suffix_text)]}{suffix_text}"
        suffix += 1
    return candidate


def _slug_exists(
    db: Session,
    slug_column: Any,
    candidate: str,
    *filters: Any,
) -> bool:
    stmt = select(slug_column).where(slug_column == candidate)
    for condition in filters:
        stmt = stmt.where(condition)
    return db.scalar(stmt) is not None
