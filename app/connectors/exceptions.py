from __future__ import annotations

import re

from app.core.logging import redact_secret

_URI_CREDENTIALS = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://[^:/?#\s]+):([^@/?#\s]+)@")


def sanitize_connector_message(message: str) -> str:
    """Strip credentials and connection secrets from a connector error message."""
    redacted = redact_secret(message)
    return _URI_CREDENTIALS.sub(r"\1:[REDACTED]@", redacted)


class ConnectorError(Exception):
    """Base error for customer data-source connector failures.

    Messages are sanitized so credentials never reach API responses or logs.
    """

    def __init__(self, message: str = "Connector operation failed") -> None:
        super().__init__(sanitize_connector_message(message))


class ConnectorConnectionError(ConnectorError):
    """The customer data source could not be reached."""

    def __init__(self, message: str = "Failed to connect to data source") -> None:
        super().__init__(message)


class ConnectorAuthenticationError(ConnectorError):
    """The customer data source rejected the supplied credentials."""

    def __init__(self, message: str = "Data source authentication failed") -> None:
        super().__init__(message)


class ConnectorQueryError(ConnectorError):
    """A query against the customer data source failed."""

    def __init__(self, message: str = "Data source query failed") -> None:
        super().__init__(message)


class UnsupportedOperationError(ConnectorError):
    """The connector does not support the requested operation."""

    def __init__(
        self,
        message: str = "Operation is not supported for this connector",
    ) -> None:
        super().__init__(message)


class UnsupportedConnectorError(ConnectorError):
    """The data source type is known but has no connector implementation."""

    def __init__(self, message: str = "Connector type is not implemented") -> None:
        super().__init__(message)
