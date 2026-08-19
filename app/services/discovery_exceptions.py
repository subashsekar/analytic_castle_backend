"""Errors raised while inspecting customer-database metadata.

Messages are sanitized so credentials never reach logs or callers.
"""

from __future__ import annotations

from app.connectors.exceptions import sanitize_connector_message


class MetadataDiscoveryError(Exception):
    """Base error for metadata discovery failures."""

    def __init__(self, message: str = "Metadata discovery failed") -> None:
        super().__init__(sanitize_connector_message(message))


class SchemaDiscoveryError(MetadataDiscoveryError):
    """Schema metadata could not be read from the customer database."""

    def __init__(self, message: str = "Unable to discover database schemas") -> None:
        super().__init__(message)


class TableDiscoveryError(MetadataDiscoveryError):
    """Table metadata could not be read from the customer database."""

    def __init__(self, message: str = "Unable to discover database tables") -> None:
        super().__init__(message)


class ColumnDiscoveryError(MetadataDiscoveryError):
    """Column metadata could not be read from the customer database."""

    def __init__(self, message: str = "Unable to discover database columns") -> None:
        super().__init__(message)


class RelationshipDiscoveryError(MetadataDiscoveryError):
    """Foreign-key metadata could not be read from the customer database."""

    def __init__(
        self,
        message: str = "Unable to discover database relationships",
    ) -> None:
        super().__init__(message)


class MetadataDiscoveryLimitError(MetadataDiscoveryError):
    """A discovery safeguard limit was exceeded. Metadata is not truncated."""

    def __init__(self, message: str = "Metadata discovery limit exceeded") -> None:
        super().__init__(message)
