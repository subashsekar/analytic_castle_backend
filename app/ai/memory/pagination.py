"""Pagination helpers for conversation memory."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from app.ai.memory.errors import InvalidPaginationError
from app.ai.memory.models import Page

T = TypeVar("T")


def normalize_page_params(
    *, page: int, page_size: int, max_page_size: int
) -> tuple[int, int]:
    if page < 1:
        raise InvalidPaginationError("Page must be at least 1")
    if page_size < 1:
        raise InvalidPaginationError("Page size must be at least 1")
    if page_size > max_page_size:
        raise InvalidPaginationError(f"Page size must not exceed {max_page_size}")
    return page, page_size


def paginate_items(items: Sequence[T], *, page: int, page_size: int) -> Page[T]:
    total = len(items)
    start = (page - 1) * page_size
    if start >= total > 0:
        raise InvalidPaginationError("Page exceeds available results")
    end = start + page_size
    return Page(
        items=tuple(items[start:end]),
        page=page,
        page_size=page_size,
        total=total,
    )
