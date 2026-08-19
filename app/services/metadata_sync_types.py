"""In-memory results of metadata synchronization."""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from app.enums import MetadataSyncStatus
from app.services.discovery_types import (
    DiscoveryLimits,
    DiscoveryResult,
    SchemaFilter,
)


@dataclass(frozen=True)
class PersistedMetadataCounts:
    schemas: int
    tables: int
    columns: int
    relationships: int


@dataclass(frozen=True)
class MetadataSyncResult:
    data_source_id: UUID
    status: MetadataSyncStatus
    started_at: datetime | None
    completed_at: datetime | None
    schema_count: int | None
    table_count: int | None
    column_count: int | None
    relationship_count: int | None
    error_message: str | None


class MetadataDiscoverer(Protocol):
    """Discovery surface used by synchronization. Phase 4.2 implements this."""

    def discover(
        self,
        data_source_id: UUID,
        *,
        workspace_id: UUID,
        schema_filter: SchemaFilter | None = None,
        limits: DiscoveryLimits | None = None,
    ) -> Awaitable[DiscoveryResult]: ...
