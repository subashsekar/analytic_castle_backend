"""Search persisted metadata for one workspace-scoped data source.

Queries run in PostgreSQL against the authoritative metadata tables. This
module does not implement HTTP, RBAC, discovery, or synchronization.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from sqlalchemy import (
    String,
    and_,
    case,
    cast,
    func,
    literal,
    null,
    or_,
    select,
    union_all,
)
from sqlalchemy.engine import RowMapping
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select, Subquery

from app.core.config import settings
from app.db.models import (
    DataSource,
    DataSourceColumn,
    DataSourceSchema,
    DataSourceTable,
)
from app.enums import MetadataSearchType
from app.services.data_source_connections import DataSourceNotFoundError
from app.services.metadata_search_exceptions import (
    MetadataSearchError,
    MetadataSearchLimitError,
)
from app.services.metadata_search_types import MetadataSearchPage, MetadataSearchResult

logger = logging.getLogger(__name__)

_LIKE_ESCAPE = "\\"
_ALL_SEARCH_TYPES = (
    MetadataSearchType.SCHEMA,
    MetadataSearchType.TABLE,
    MetadataSearchType.COLUMN,
)
_SearchSelect = Select[Any]


def escape_like_pattern(value: str) -> str:
    """Treat LIKE metacharacters in user input as literals."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class MetadataSearchService:
    """Find schema, table, and column metadata within one data source."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def search_metadata(
        self,
        data_source_id: UUID,
        query: str,
        *,
        workspace_id: UUID,
        metadata_type: MetadataSearchType | None = None,
        limit: int | None = None,
    ) -> MetadataSearchPage:
        self._load_workspace_data_source(data_source_id, workspace_id)
        resolved_limit = self._resolve_limit(limit)
        started = time.perf_counter()
        normalized = query.strip()
        if not normalized:
            page = MetadataSearchPage(results=(), limit=resolved_limit, truncated=False)
            self._log_completed(
                data_source_id,
                metadata_type=metadata_type,
                result_count=0,
                limit=resolved_limit,
                started=started,
            )
            return page

        combined = self._search_subquery(
            data_source_id,
            query=normalized,
            metadata_type=metadata_type,
        )
        fetch_limit = resolved_limit + 1
        stmt = (
            select(combined)
            .order_by(
                combined.c.match_rank,
                combined.c.schema_name,
                combined.c.table_name.nulls_first(),
                combined.c.column_name.nulls_first(),
                combined.c.matched_name,
            )
            .limit(fetch_limit)
        )
        rows = self._session.execute(stmt).mappings().all()
        truncated = len(rows) > resolved_limit
        results = tuple(self._row_to_result(row) for row in rows[:resolved_limit])
        page = MetadataSearchPage(
            results=results,
            limit=resolved_limit,
            truncated=truncated,
        )
        self._log_completed(
            data_source_id,
            metadata_type=metadata_type,
            result_count=page.result_count,
            limit=resolved_limit,
            started=started,
        )
        return page

    def _load_workspace_data_source(
        self,
        data_source_id: UUID,
        workspace_id: UUID,
    ) -> DataSource:
        data_source = self._session.get(DataSource, data_source_id)
        if data_source is None or data_source.workspace_id != workspace_id:
            raise DataSourceNotFoundError("Data source not found")
        return data_source

    def _resolve_limit(self, limit: int | None) -> int:
        if limit is None:
            return settings.METADATA_SEARCH_DEFAULT_LIMIT
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise MetadataSearchLimitError("Metadata search limit is invalid")
        if limit < 1 or limit > settings.METADATA_SEARCH_MAX_LIMIT:
            raise MetadataSearchLimitError(
                "Metadata search limit must be between 1 and "
                f"{settings.METADATA_SEARCH_MAX_LIMIT}"
            )
        return limit

    def _search_subquery(
        self,
        data_source_id: UUID,
        *,
        query: str,
        metadata_type: MetadataSearchType | None,
    ) -> Subquery:
        statements = self._search_statements(
            data_source_id,
            query=query,
            metadata_type=metadata_type,
        )
        inner = statements[0] if len(statements) == 1 else union_all(*statements)
        return inner.subquery("metadata_search")

    def _search_statements(
        self,
        data_source_id: UUID,
        *,
        query: str,
        metadata_type: MetadataSearchType | None,
    ) -> Sequence[_SearchSelect]:
        types = _ALL_SEARCH_TYPES if metadata_type is None else (metadata_type,)
        query_lower = query.lower()
        escaped = escape_like_pattern(query_lower)
        prefix_pattern = f"{escaped}%"
        contains_pattern = f"%{escaped}%"
        statements: list[_SearchSelect] = []
        if MetadataSearchType.SCHEMA in types:
            statements.append(
                self._schema_statement(
                    data_source_id,
                    query_lower=query_lower,
                    prefix_pattern=prefix_pattern,
                    contains_pattern=contains_pattern,
                )
            )
        if MetadataSearchType.TABLE in types:
            statements.append(
                self._table_statement(
                    data_source_id,
                    query_lower=query_lower,
                    prefix_pattern=prefix_pattern,
                    contains_pattern=contains_pattern,
                )
            )
        if MetadataSearchType.COLUMN in types:
            statements.append(
                self._column_statement(
                    data_source_id,
                    query_lower=query_lower,
                    prefix_pattern=prefix_pattern,
                    contains_pattern=contains_pattern,
                )
            )
        if not statements:
            raise MetadataSearchError("Metadata search type is invalid")
        return statements

    def _schema_statement(
        self,
        data_source_id: UUID,
        *,
        query_lower: str,
        prefix_pattern: str,
        contains_pattern: str,
    ) -> _SearchSelect:
        return select(
            literal(MetadataSearchType.SCHEMA.value).label("metadata_type"),
            DataSourceSchema.name.label("schema_name"),
            cast(null(), String).label("table_name"),
            cast(null(), String).label("column_name"),
            cast(null(), String).label("description"),
            DataSourceSchema.name.label("matched_name"),
            self._rank_expr(
                DataSourceSchema.name,
                query_lower=query_lower,
                prefix_pattern=prefix_pattern,
                contains_pattern=contains_pattern,
            ).label("match_rank"),
        ).where(
            DataSourceSchema.data_source_id == data_source_id,
            self._name_matches(DataSourceSchema.name, contains_pattern),
        )

    def _table_statement(
        self,
        data_source_id: UUID,
        *,
        query_lower: str,
        prefix_pattern: str,
        contains_pattern: str,
    ) -> _SearchSelect:
        return (
            select(
                literal(MetadataSearchType.TABLE.value).label("metadata_type"),
                DataSourceSchema.name.label("schema_name"),
                DataSourceTable.name.label("table_name"),
                cast(null(), String).label("column_name"),
                DataSourceTable.description.label("description"),
                DataSourceTable.name.label("matched_name"),
                self._rank_expr(
                    DataSourceTable.name,
                    query_lower=query_lower,
                    prefix_pattern=prefix_pattern,
                    contains_pattern=contains_pattern,
                    description_column=DataSourceTable.description,
                ).label("match_rank"),
            )
            .join(
                DataSourceSchema,
                DataSourceTable.schema_id == DataSourceSchema.id,
            )
            .where(
                DataSourceSchema.data_source_id == data_source_id,
                or_(
                    self._name_matches(DataSourceTable.name, contains_pattern),
                    self._description_matches(
                        DataSourceTable.description,
                        contains_pattern,
                    ),
                ),
            )
        )

    def _column_statement(
        self,
        data_source_id: UUID,
        *,
        query_lower: str,
        prefix_pattern: str,
        contains_pattern: str,
    ) -> _SearchSelect:
        return (
            select(
                literal(MetadataSearchType.COLUMN.value).label("metadata_type"),
                DataSourceSchema.name.label("schema_name"),
                DataSourceTable.name.label("table_name"),
                DataSourceColumn.name.label("column_name"),
                DataSourceColumn.description.label("description"),
                DataSourceColumn.name.label("matched_name"),
                self._rank_expr(
                    DataSourceColumn.name,
                    query_lower=query_lower,
                    prefix_pattern=prefix_pattern,
                    contains_pattern=contains_pattern,
                    description_column=DataSourceColumn.description,
                ).label("match_rank"),
            )
            .join(
                DataSourceTable,
                DataSourceColumn.table_id == DataSourceTable.id,
            )
            .join(
                DataSourceSchema,
                DataSourceTable.schema_id == DataSourceSchema.id,
            )
            .where(
                DataSourceSchema.data_source_id == data_source_id,
                or_(
                    self._name_matches(DataSourceColumn.name, contains_pattern),
                    self._description_matches(
                        DataSourceColumn.description,
                        contains_pattern,
                    ),
                ),
            )
        )

    def _rank_expr(
        self,
        name_column: Any,
        *,
        query_lower: str,
        prefix_pattern: str,
        contains_pattern: str,
        description_column: Any | None = None,
    ) -> Any:
        whens: list[tuple[Any, int]] = [
            (func.lower(name_column) == query_lower, 1),
            (func.lower(name_column).like(prefix_pattern, escape=_LIKE_ESCAPE), 2),
            (func.lower(name_column).like(contains_pattern, escape=_LIKE_ESCAPE), 3),
        ]
        if description_column is not None:
            whens.append(
                (self._description_matches(description_column, contains_pattern), 4)
            )
        return case(*whens, else_=4)

    def _name_matches(self, name_column: Any, contains_pattern: str) -> Any:
        return func.lower(name_column).like(contains_pattern, escape=_LIKE_ESCAPE)

    def _description_matches(
        self, description_column: Any, contains_pattern: str
    ) -> Any:
        return and_(
            description_column.is_not(None),
            func.lower(description_column).like(contains_pattern, escape=_LIKE_ESCAPE),
        )

    def _row_to_result(self, row: RowMapping) -> MetadataSearchResult:
        return MetadataSearchResult(
            metadata_type=MetadataSearchType(row["metadata_type"]),
            schema_name=str(row["schema_name"]),
            table_name=row["table_name"],
            column_name=row["column_name"],
            matched_name=str(row["matched_name"]),
            description=row["description"],
            match_rank=int(row["match_rank"]),
        )

    def _log_completed(
        self,
        data_source_id: UUID,
        *,
        metadata_type: MetadataSearchType | None,
        result_count: int,
        limit: int,
        started: float,
    ) -> None:
        duration_ms = (time.perf_counter() - started) * 1000
        search_type = metadata_type.value if metadata_type is not None else "all"
        logger.info(
            "Metadata search completed data_source_id=%s search_type=%s "
            "result_count=%s limit=%s duration_ms=%.0f",
            data_source_id,
            search_type,
            result_count,
            limit,
            duration_ms,
        )
