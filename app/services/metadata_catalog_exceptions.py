"""Errors raised while reading persisted metadata for API responses."""

from __future__ import annotations


class MetadataCatalogError(Exception):
    """Base error for metadata catalog read failures."""

    def __init__(self, message: str = "Unable to load metadata") -> None:
        super().__init__(message)


class MetadataNotFoundError(MetadataCatalogError):
    """The requested metadata resource is missing or is outside the data source."""

    def __init__(self, message: str = "Metadata not found") -> None:
        super().__init__(message)


class MetadataPageError(MetadataCatalogError):
    """The requested page or page size is outside the allowed range."""

    def __init__(self, message: str = "Pagination parameters are invalid") -> None:
        super().__init__(message)
