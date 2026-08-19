from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ConnectorConfig:
    """Internal connection settings for a customer data source.

    This object is not an API schema. ``credential`` is the already-decrypted
    password supplied by the credential service. It is omitted from
    ``repr`` / ``str`` and must not be logged or returned by API schemas.
    """

    host: str
    port: int
    database_name: str
    username: str
    credential: str = field(repr=False)
    ssl_mode: str | None = None

    def __repr__(self) -> str:
        return (
            "ConnectorConfig("
            f"host={self.host!r}, "
            f"port={self.port!r}, "
            f"database_name={self.database_name!r}, "
            f"username={self.username!r}, "
            f"ssl_mode={self.ssl_mode!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()


@dataclass(frozen=True)
class ConnectionTestResult:
    success: bool
    message: str = ""


@dataclass(frozen=True)
class SchemaInfo:
    name: str


@dataclass(frozen=True)
class TableInfo:
    name: str
    schema: str | None = None


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    data_type: str
    nullable: bool = True


@dataclass(frozen=True)
class QueryResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    truncated: bool = False
