from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ai.glossary import MAX_GLOSSARY_ENTRIES, GlossaryEntry
from app.connectors.postgresql import ALLOWED_SSL_MODES
from app.enums import DataSourceStatus, DataSourceType

_DEFAULT_POSTGRES_PORT = 5432
_DEFAULT_SSL_MODE = "require"
_MAX_SECRET_LENGTH = 512


def _strip_text(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


class ConnectionConfigCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = Field(min_length=1, max_length=255)
    port: int = Field(default=_DEFAULT_POSTGRES_PORT, ge=1, le=65535)
    database_name: str = Field(min_length=1, max_length=255)
    username: str = Field(min_length=1, max_length=255)
    password: str = Field(
        min_length=1,
        max_length=_MAX_SECRET_LENGTH,
        repr=False,
        json_schema_extra={"writeOnly": True, "format": "password"},
    )
    ssl_mode: str = Field(default=_DEFAULT_SSL_MODE, min_length=1, max_length=32)

    @field_validator("host", "database_name", "username", "ssl_mode", mode="before")
    @classmethod
    def strip_text_fields(cls, value: Any) -> Any:
        return _strip_text(value)

    @field_validator("ssl_mode")
    @classmethod
    def validate_ssl_mode(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in ALLOWED_SSL_MODES:
            raise ValueError("Invalid PostgreSQL SSL mode")
        return normalized


class DataSourceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: UUID
    name: str = Field(min_length=1, max_length=255)
    type: DataSourceType
    connection: ConnectionConfigCreate

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value: Any) -> Any:
        return _strip_text(value)


class DataSourceUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=255)

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value: Any) -> Any:
        return _strip_text(value)


class DataSourceGlossary(BaseModel):
    """Business glossary: metric synonyms/definitions used by the AI analyst."""

    model_config = ConfigDict(extra="forbid")

    entries: list[GlossaryEntry] = Field(default_factory=list, max_length=MAX_GLOSSARY_ENTRIES)


class DataSourceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    workspace_id: UUID
    name: str
    type: DataSourceType
    status: DataSourceStatus
    created_by: UUID
    created_at: datetime
    updated_at: datetime
    last_tested_at: datetime | None


class DataSourceConnectionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    data_source_id: UUID
    host: str
    port: int
    database_name: str
    username: str
    ssl_mode: str
    created_at: datetime
    updated_at: datetime


class ConnectionTestResponse(BaseModel):
    success: bool
    message: str
