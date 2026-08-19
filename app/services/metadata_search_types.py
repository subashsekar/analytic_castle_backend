"""In-memory results of internal metadata search.

These structures are not ORM models and are not HTTP schemas. Search reads
the Phase 4.1 metadata tables directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.enums import MetadataSearchType


@dataclass(frozen=True)
class MetadataSearchResult:
    metadata_type: MetadataSearchType
    schema_name: str
    table_name: str | None
    column_name: str | None
    matched_name: str
    description: str | None
    match_rank: int


@dataclass(frozen=True)
class MetadataSearchPage:
    results: tuple[MetadataSearchResult, ...]
    limit: int
    truncated: bool

    @property
    def result_count(self) -> int:
        return len(self.results)
