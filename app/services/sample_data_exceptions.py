"""Errors raised while retrieving safe sample data.

Messages are sanitized so credentials and driver details never reach callers.
"""

from __future__ import annotations

from app.connectors.exceptions import sanitize_connector_message


class SampleDataError(Exception):
    """Base error for internal sample-data failures."""

    def __init__(self, message: str = "Unable to retrieve sample data") -> None:
        super().__init__(sanitize_connector_message(message))


class SampleTableNotFoundError(SampleDataError):
    """The requested table is missing or does not belong to the data source."""

    def __init__(self, message: str = "Table not found") -> None:
        super().__init__(message)


class SampleDataLimitError(SampleDataError):
    """The requested sample row limit is invalid."""

    def __init__(self, message: str = "Sample row limit is invalid") -> None:
        super().__init__(message)


class SampleIdentifierError(SampleDataError):
    """A schema, table, or column identifier cannot be used in a sample query."""

    def __init__(self, message: str = "Invalid identifier") -> None:
        super().__init__(message)


class SampleQueryError(SampleDataError):
    """The customer database rejected or failed the sample query."""

    def __init__(self, message: str = "Unable to retrieve sample data") -> None:
        super().__init__(message)


class SampleSerializationError(SampleDataError):
    """A sample value could not be converted into a safe representation."""

    def __init__(self, message: str = "Unable to serialize sample data") -> None:
        super().__init__(message)
