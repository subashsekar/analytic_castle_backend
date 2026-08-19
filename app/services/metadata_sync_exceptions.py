"""Errors raised while synchronizing discovered metadata.

Messages are sanitized so credentials never reach logs or callers.
"""

from __future__ import annotations

from app.connectors.exceptions import sanitize_connector_message


class MetadataSyncError(Exception):
    """Base error for metadata synchronization failures."""

    def __init__(self, message: str = "Metadata synchronization failed") -> None:
        super().__init__(sanitize_connector_message(message))


class ConcurrentMetadataSyncError(MetadataSyncError):
    """Another synchronization of this data source is already running."""

    def __init__(
        self,
        message: str = "Metadata synchronization is already running",
    ) -> None:
        super().__init__(message)


class MetadataSyncValidationError(MetadataSyncError):
    """Discovered metadata is incomplete or internally inconsistent."""

    def __init__(self, message: str = "Discovered metadata is invalid") -> None:
        super().__init__(message)


class MetadataSyncPersistenceError(MetadataSyncError):
    """Application-database persistence of metadata failed."""

    def __init__(self, message: str = "Unable to save metadata") -> None:
        super().__init__(message)
