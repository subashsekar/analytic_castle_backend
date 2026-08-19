"""Errors raised while searching persisted metadata."""

from __future__ import annotations


class MetadataSearchError(Exception):
    """Base error for internal metadata search failures."""

    def __init__(self, message: str = "Metadata search failed") -> None:
        super().__init__(message)


class MetadataSearchLimitError(MetadataSearchError):
    """The requested search limit is outside the allowed range."""

    def __init__(self, message: str = "Metadata search limit is invalid") -> None:
        super().__init__(message)
