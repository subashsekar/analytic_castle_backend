"""Deterministic metadata context resolution for Phase 5.3.

Resolves Phase 5.2 intent against Phase 4 persisted metadata. Does not call an
LLM, generate SQL, or query customer business tables.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Literal, TypeVar
from uuid import UUID

from sqlalchemy import func, or_, select, tuple_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, aliased, joinedload

from app.ai.exceptions import AIContextError
from app.ai.glossary import (
    GlossaryEntry,
    concept_variants,
    match_glossary,
    normalize_concept,
    parse_glossary,
)
from app.ai.intent_types import AggregationType, AIIntent, AIIntentType, _looks_like_sql
from app.ai.metadata_types import (
    ConceptColumnResolution,
    MetadataColumnCandidate,
    MetadataMatchReason,
    MetadataPrimaryKey,
    MetadataRelationshipCandidate,
    MetadataTableCandidate,
    ResolvedMetadataContext,
    empty_resolved_context,
    is_identifier_column,
    is_numeric_type,
)
from app.ai.types import AIContext
from app.core.config import settings
from app.core.request_id import get_request_id
from app.db.models import (
    DataSource,
    DataSourceColumn,
    DataSourceRelationship,
    DataSourceSchema,
    DataSourceTable,
)
from app.enums import MetadataSearchType
from app.services.data_source_connections import DataSourceNotFoundError
from app.services.metadata_search import MetadataSearchService
from app.services.metadata_search_exceptions import MetadataSearchError
from app.services.metadata_search_types import MetadataSearchResult

logger = logging.getLogger(__name__)

_RANK_REASON = {
    1: MetadataMatchReason.EXACT,
    2: MetadataMatchReason.PREFIX,
    3: MetadataMatchReason.CONTAINS,
    4: MetadataMatchReason.DESCRIPTION,
}
_RANK_SCORE = {1: 100, 2: 80, 3: 60, 4: 40}
_AGGREGATION_PREFIX = re.compile(
    r"^(total|sum|avg|average|count|number of|number|distinct)\s+",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CATALOG_NOISE_RE = re.compile(
    r"\b(tables?|columns?|fields?|schema|database)\b",
    re.IGNORECASE,
)
_MEASURE_NOISE_RE = re.compile(
    r"\b(sold|made|shipped|booked|recorded|completed|purchased|placed)\b",
    re.IGNORECASE,
)
_SEARCH_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "by",
        "can",
        "column",
        "columns",
        "data",
        "does",
        "for",
        "from",
        "have",
        "how",
        "in",
        "is",
        "look",
        "looks",
        "me",
        "more",
        "most",
        "of",
        "or",
        "related",
        "show",
        "table",
        "tables",
        "the",
        "to",
        "what",
        "which",
        "with",
    }
)
_TEMPORAL_MARKERS = ("timestamp", "timestamptz", "datetime", "timetz", "date", "time")
_DATA_INTENTS = {
    AIIntentType.ANALYTICAL_QUERY,
    AIIntentType.DATA_LOOKUP,
    AIIntentType.AGGREGATION,
    AIIntentType.COMPARISON,
    AIIntentType.TREND_ANALYSIS,
    AIIntentType.RANKING,
    AIIntentType.SUMMARY,
}
_ConceptKind = Literal["subject", "metric", "dimension", "filter", "sort"]
TableKey = tuple[str, str]
ColumnKey = tuple[str, str, str]
K = TypeVar("K")


@dataclass(frozen=True)
class _ConceptWork:
    kind: _ConceptKind
    requested: str
    terms: tuple[str, ...]
    schema_name: str | None
    table_name: str | None
    column_name: str | None


class MetadataContextResolver:
    """Map a structured intent to compact, existing catalog metadata."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._search = MetadataSearchService(session)

    def resolve(
        self,
        intent: AIIntent,
        context: AIContext,
        *,
        message: str | None = None,
    ) -> ResolvedMetadataContext:
        started = time.perf_counter()
        self._validate_context(context)
        if intent.intent is AIIntentType.UNSUPPORTED:
            return empty_resolved_context(context.data_source_id)

        try:
            resolved = self._resolve_authorized(intent, context, message=message)
        except DataSourceNotFoundError as exc:
            raise AIContextError("Data source not found") from exc
        except MetadataSearchError:
            logger.warning(
                "AI metadata resolution search failed request_id=%s data_source_id=%s",
                get_request_id(),
                context.data_source_id,
            )
            return empty_resolved_context(context.data_source_id).model_copy(
                update={
                    "requires_clarification": True,
                    "clarification_question": (
                        "Metadata is currently unavailable for this data source."
                    ),
                }
            )

        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "AI metadata resolved request_id=%s data_source_id=%s "
            "table_count=%s column_count=%s relationship_count=%s "
            "unresolved_count=%s duration_ms=%.0f",
            get_request_id(),
            context.data_source_id,
            len(resolved.tables),
            len(resolved.columns),
            len(resolved.relationships),
            len(resolved.unresolved_concepts),
            duration_ms,
        )
        return resolved

    def _validate_context(self, context: AIContext) -> None:
        if context.workspace_id is None or context.data_source_id is None:
            raise AIContextError()

    def _resolve_authorized(
        self,
        intent: AIIntent,
        context: AIContext,
        *,
        message: str | None = None,
    ) -> ResolvedMetadataContext:
        glossary = self._load_glossary(context)
        work, glossary_hits = _apply_glossary(
            _concepts_from_intent(intent, message=message), glossary
        )
        cache = self._search_terms(context, work)
        table_ranks, column_ranks, concept_columns = _collect_hits(work, cache)
        tables_by_key = self._load_tables(context.data_source_id, set(table_ranks))
        columns_by_key = self._load_columns(context.data_source_id, set(column_ranks))
        table_ids = {table.id for table in tables_by_key.values()}
        primary_keys = self._load_primary_keys(context.data_source_id, table_ids)
        # Always load temporal columns for matched tables so monthly/date SQL
        # can validate even when intent.time_range was not parsed.
        time_columns = (
            self._load_time_columns(context.data_source_id, table_ids)
            if table_ids
            else ()
        )

        table_candidates = _limit_tables(
            _table_candidates(table_ranks, tables_by_key, primary_keys)
        )
        allowed_table_ids = {item.table_id for item in table_candidates}
        column_candidates = [
            item
            for item in _column_candidates(column_ranks, columns_by_key)
            if item.table_id in allowed_table_ids
        ][: settings.AI_MAX_METADATA_COLUMNS]
        # Analytical queries need the full matched-table column set (e.g. sale_date),
        # not only search hits — otherwise SQL validation rejects UNKNOWN_COLUMN.
        if table_candidates and (
            intent.intent in _DATA_INTENTS or intent.requires_data_access
        ):
            column_candidates = _merge_column_candidates(
                column_candidates,
                self._load_schema_question_columns(
                    context.data_source_id,
                    {item.table_id for item in table_candidates},
                ),
            )
        elif (
            intent.intent is AIIntentType.SCHEMA_QUESTION
            and table_candidates
            and not column_candidates
        ):
            column_candidates = self._load_schema_question_columns(
                context.data_source_id,
                {item.table_id for item in table_candidates},
            )
        allowed_column_ids = {item.column_id for item in column_candidates}
        relationship_candidates = self._load_relationships(
            context.data_source_id,
            allowed_table_ids,
        )

        resolved_metrics = [
            _column_resolution(
                item, concept_columns, columns_by_key, allowed_column_ids
            )
            for item in work
            if item.kind == "metric"
        ]
        resolved_dimensions = [
            _column_resolution(
                item, concept_columns, columns_by_key, allowed_column_ids
            )
            for item in work
            if item.kind == "dimension"
        ]
        resolved_filters = [
            _column_resolution(
                item, concept_columns, columns_by_key, allowed_column_ids
            )
            for item in work
            if item.kind == "filter"
        ]
        resolved_sorts = [
            _column_resolution(
                item, concept_columns, columns_by_key, allowed_column_ids
            )
            for item in work
            if item.kind == "sort"
        ]
        time_candidates = [
            _column_from_loaded(column, match_rank=1)
            for column in time_columns
            if column.table_id in allowed_table_ids
        ][: settings.AI_MAX_METADATA_COLUMNS]

        aggregations = {metric.name: metric.aggregation for metric in intent.metrics}
        resolved_metrics = [
            _refine_metric(
                item,
                aggregation=aggregations.get(item.requested, AggregationType.NONE),
                glossary_entry=glossary_hits.get(f"metric:{item.requested}"),
                tables=table_candidates,
                columns=column_candidates,
            )
            for item in resolved_metrics
        ]
        # A metric's own table decides the date column; only ask when that
        # table itself has several temporal columns.
        metric_tables = {
            candidate.table_id
            for item in resolved_metrics
            for candidate in item.candidates
        }
        metric_time = [item for item in time_candidates if item.table_id in metric_tables]
        if metric_time:
            time_candidates = metric_time + [
                item for item in time_candidates if item.table_id not in metric_tables
            ]

        unresolved = _unresolved_concepts(
            intent=intent,
            work=work,
            table_candidates=table_candidates,
            metrics=resolved_metrics,
            dimensions=resolved_dimensions,
            filters=resolved_filters,
            sorts=resolved_sorts,
            time_candidates=time_candidates,
        )
        ambiguous = _ambiguous_items(
            metrics=resolved_metrics,
            dimensions=resolved_dimensions,
            filters=resolved_filters,
            sorts=resolved_sorts,
            time_candidates=metric_time or time_candidates,
            table_candidates=table_candidates,
            require_time_disambiguation=intent.time_range is not None,
        )
        requires_clarification = bool(unresolved or ambiguous)
        question = None
        if intent.intent is AIIntentType.UNKNOWN:
            requires_clarification = True
            question = intent.clarification_question
        elif requires_clarification:
            question = _clarification_question(ambiguous, unresolved, table_candidates)

        return ResolvedMetadataContext(
            data_source_id=context.data_source_id,
            tables=table_candidates,
            columns=column_candidates,
            relationships=relationship_candidates,
            resolved_metrics=resolved_metrics,
            resolved_dimensions=resolved_dimensions,
            resolved_filters=resolved_filters,
            resolved_time_columns=time_candidates,
            unresolved_concepts=unresolved,
            glossary=_relevant_glossary(glossary, glossary_hits, message),
            requires_clarification=requires_clarification,
            clarification_question=question,
        )

    def _load_glossary(self, context: AIContext) -> list[GlossaryEntry]:
        # Savepoint: a database without the glossary column must not poison the
        # caller's transaction; resolution simply proceeds without a glossary.
        try:
            with self._session.begin_nested():
                raw = self._session.scalar(
                    select(DataSource.business_glossary).where(
                        DataSource.id == context.data_source_id,
                        DataSource.workspace_id == context.workspace_id,
                    )
                )
        except SQLAlchemyError:
            logger.warning(
                "AI glossary unavailable request_id=%s data_source_id=%s",
                get_request_id(),
                context.data_source_id,
            )
            return []
        return parse_glossary(raw)

    def _search_terms(
        self,
        context: AIContext,
        work: list[_ConceptWork],
    ) -> dict[str, tuple[MetadataSearchResult, ...]]:
        cache: dict[str, tuple[MetadataSearchResult, ...]] = {}
        for term in _unique_terms(work):
            query = term.strip()[: settings.METADATA_SEARCH_MAX_QUERY_LENGTH]
            if not query or _looks_like_sql(query):
                cache[term] = ()
                continue
            limit = min(
                settings.AI_METADATA_RESOLVE_SEARCH_LIMIT,
                settings.METADATA_SEARCH_MAX_LIMIT,
            )
            page = self._search.search_metadata(
                context.data_source_id,
                query,
                workspace_id=context.workspace_id,
                limit=limit,
            )
            cache[term] = page.results
        return cache

    def _load_tables(
        self,
        data_source_id: UUID,
        keys: set[TableKey],
    ) -> dict[TableKey, DataSourceTable]:
        if not keys:
            return {}
        stmt = (
            select(DataSourceTable)
            .join(DataSourceSchema, DataSourceTable.schema_id == DataSourceSchema.id)
            .options(joinedload(DataSourceTable.schema))
            .where(
                DataSourceSchema.data_source_id == data_source_id,
                tuple_(DataSourceSchema.name, DataSourceTable.name).in_(sorted(keys)),
            )
        )
        loaded = self._session.scalars(stmt).unique().all()
        return {(table.schema.name, table.name): table for table in loaded}

    def _load_columns(
        self,
        data_source_id: UUID,
        keys: set[ColumnKey],
    ) -> dict[ColumnKey, DataSourceColumn]:
        if not keys:
            return {}
        stmt = (
            select(DataSourceColumn)
            .join(DataSourceTable, DataSourceColumn.table_id == DataSourceTable.id)
            .join(DataSourceSchema, DataSourceTable.schema_id == DataSourceSchema.id)
            .options(
                joinedload(DataSourceColumn.table).joinedload(DataSourceTable.schema)
            )
            .where(
                DataSourceSchema.data_source_id == data_source_id,
                tuple_(
                    DataSourceSchema.name,
                    DataSourceTable.name,
                    DataSourceColumn.name,
                ).in_(sorted(keys)),
            )
        )
        loaded = self._session.scalars(stmt).unique().all()
        return {
            (column.table.schema.name, column.table.name, column.name): column
            for column in loaded
        }

    def _load_schema_question_columns(
        self,
        data_source_id: UUID,
        table_ids: set[UUID],
    ) -> list[MetadataColumnCandidate]:
        if not table_ids:
            return []
        stmt = (
            select(DataSourceColumn)
            .join(DataSourceTable, DataSourceColumn.table_id == DataSourceTable.id)
            .join(DataSourceSchema, DataSourceTable.schema_id == DataSourceSchema.id)
            .options(
                joinedload(DataSourceColumn.table).joinedload(DataSourceTable.schema)
            )
            .where(
                DataSourceSchema.data_source_id == data_source_id,
                DataSourceColumn.table_id.in_(table_ids),
            )
            .order_by(
                DataSourceColumn.table_id,
                DataSourceColumn.ordinal_position,
                DataSourceColumn.name,
            )
        )
        loaded = self._session.scalars(stmt).unique().all()
        return [
            _column_from_loaded(column, match_rank=1)
            for column in loaded[: settings.AI_MAX_METADATA_COLUMNS]
        ]

    def _load_primary_keys(
        self,
        data_source_id: UUID,
        table_ids: set[UUID],
    ) -> dict[UUID, list[DataSourceColumn]]:
        if not table_ids:
            return {}
        stmt = (
            select(DataSourceColumn)
            .join(DataSourceTable, DataSourceColumn.table_id == DataSourceTable.id)
            .join(DataSourceSchema, DataSourceTable.schema_id == DataSourceSchema.id)
            .where(
                DataSourceSchema.data_source_id == data_source_id,
                DataSourceColumn.table_id.in_(table_ids),
                DataSourceColumn.is_primary_key.is_(True),
            )
            .order_by(DataSourceColumn.table_id, DataSourceColumn.ordinal_position)
        )
        grouped: dict[UUID, list[DataSourceColumn]] = {}
        for column in self._session.scalars(stmt):
            grouped.setdefault(column.table_id, []).append(column)
        return grouped

    def _load_time_columns(
        self,
        data_source_id: UUID,
        table_ids: set[UUID],
    ) -> tuple[DataSourceColumn, ...]:
        if not table_ids:
            return ()
        type_filters = [
            or_(
                func.lower(DataSourceColumn.data_type).like(f"%{marker}%"),
                func.lower(DataSourceColumn.database_type).like(f"%{marker}%"),
            )
            for marker in _TEMPORAL_MARKERS
        ]
        stmt = (
            select(DataSourceColumn)
            .join(DataSourceTable, DataSourceColumn.table_id == DataSourceTable.id)
            .join(DataSourceSchema, DataSourceTable.schema_id == DataSourceSchema.id)
            .options(
                joinedload(DataSourceColumn.table).joinedload(DataSourceTable.schema)
            )
            .where(
                DataSourceSchema.data_source_id == data_source_id,
                DataSourceColumn.table_id.in_(table_ids),
                or_(*type_filters),
            )
            .order_by(
                DataSourceSchema.name,
                DataSourceTable.name,
                DataSourceColumn.ordinal_position,
            )
        )
        return tuple(self._session.scalars(stmt).unique().all())

    def _load_relationships(
        self,
        data_source_id: UUID,
        table_ids: set[UUID],
    ) -> list[MetadataRelationshipCandidate]:
        if not table_ids:
            return []
        source_table = aliased(DataSourceTable)
        target_table = aliased(DataSourceTable)
        source_schema = aliased(DataSourceSchema)
        target_schema = aliased(DataSourceSchema)
        source_column = aliased(DataSourceColumn)
        target_column = aliased(DataSourceColumn)
        stmt = (
            select(
                DataSourceRelationship,
                source_schema.name,
                source_table.name,
                source_column.name,
                target_schema.name,
                target_table.name,
                target_column.name,
            )
            .join(
                source_table,
                DataSourceRelationship.source_table_id == source_table.id,
            )
            .join(source_schema, source_table.schema_id == source_schema.id)
            .join(
                source_column,
                DataSourceRelationship.source_column_id == source_column.id,
            )
            .join(
                target_table,
                DataSourceRelationship.target_table_id == target_table.id,
            )
            .join(target_schema, target_table.schema_id == target_schema.id)
            .join(
                target_column,
                DataSourceRelationship.target_column_id == target_column.id,
            )
            .where(
                source_schema.data_source_id == data_source_id,
                target_schema.data_source_id == data_source_id,
                DataSourceRelationship.source_table_id.in_(table_ids),
                DataSourceRelationship.target_table_id.in_(table_ids),
            )
            .order_by(
                DataSourceRelationship.constraint_name.nulls_last(),
                source_schema.name,
                source_table.name,
                DataSourceRelationship.id,
            )
            .limit(settings.AI_MAX_METADATA_RELATIONSHIPS)
        )
        rows = self._session.execute(stmt).all()
        return [
            MetadataRelationshipCandidate(
                relationship_id=relation.id,
                source_schema=source_schema_name,
                source_table=source_table_name,
                source_column=source_column_name,
                source_table_id=relation.source_table_id,
                source_column_id=relation.source_column_id,
                target_schema=target_schema_name,
                target_table=target_table_name,
                target_column=target_column_name,
                target_table_id=relation.target_table_id,
                target_column_id=relation.target_column_id,
                relationship_type=relation.relationship_type,
                constraint_name=relation.constraint_name,
            )
            for (
                relation,
                source_schema_name,
                source_table_name,
                source_column_name,
                target_schema_name,
                target_table_name,
                target_column_name,
            ) in rows
        ]


def _concepts_from_intent(
    intent: AIIntent,
    *,
    message: str | None = None,
) -> list[_ConceptWork]:
    items: list[_ConceptWork] = []
    if intent.subject:
        items.append(_concept("subject", intent.subject))
    for metric in intent.metrics:
        items.append(_concept("metric", metric.name))
    for dimension in intent.dimensions:
        items.append(_concept("dimension", dimension.name))
    for filt in intent.filters:
        items.append(_concept("filter", filt.field))
    if intent.sort is not None:
        items.append(_concept("sort", intent.sort.field))
    if not items and message:
        for term in expand_metadata_search_terms(message):
            if term.lower() in _SEARCH_STOPWORDS:
                continue
            if len(term) < 3:
                continue
            if " " in term.strip():
                continue
            items.append(_concept("subject", term))
    return items


def _concept(kind: _ConceptKind, requested: str) -> _ConceptWork:
    schema_name, table_name, column_name, search_name = _parse_qualified(requested)
    return _ConceptWork(
        kind=kind,
        requested=requested,
        terms=_search_terms(search_name),
        schema_name=schema_name,
        table_name=table_name,
        column_name=column_name,
    )


def _parse_qualified(name: str) -> tuple[str | None, str | None, str | None, str]:
    parts = [part.strip() for part in name.split(".") if part.strip()]
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], None, parts[1]
    return None, None, None, name.strip()


def expand_metadata_search_terms(concept: str) -> tuple[str, ...]:
    """Expand a natural-language question into catalog search terms."""
    terms = list(_search_terms(concept))
    for token in _TOKEN_RE.findall(concept):
        if len(token) < 3:
            continue
        if token.lower() in _SEARCH_STOPWORDS:
            continue
        terms.append(token)
    return tuple(dict.fromkeys(term for term in terms if term))


def _search_terms(concept: str) -> tuple[str, ...]:
    terms = [concept]
    spaced = " ".join(concept.lower().replace("_", " ").split())
    if spaced and spaced != concept.lower():
        terms.append(spaced)
    words = spaced.split()
    if len(words) >= 2:
        acronym = "".join(word[0] for word in words if word)
        if len(acronym) >= 3:
            terms.append(acronym)
    normalized = normalize_concept(concept)
    if normalized and normalized not in {concept.lower(), spaced}:
        terms.append(normalized)
    stripped = normalized or spaced or concept
    while True:
        updated = _AGGREGATION_PREFIX.sub("", stripped, count=1).strip()
        if not updated or updated.lower() == stripped.lower():
            break
        terms.append(updated)
        stripped = updated
    cleaned = " ".join(_CATALOG_NOISE_RE.sub(" ", stripped).split()).strip()
    if cleaned and cleaned.lower() != stripped.lower():
        terms.append(cleaned)
        stripped = cleaned
    de_noised = " ".join(_MEASURE_NOISE_RE.sub(" ", stripped).split()).strip()
    if de_noised and de_noised.lower() != stripped.lower():
        terms.append(de_noised)
        stripped = de_noised
    for token in stripped.split():
        if len(token) >= 3 and token.lower() not in _SEARCH_STOPWORDS:
            terms.append(token)
    # "sales count" / "sales_count" / "number of customers" -> also search the entity.
    if _COUNT_CUE_RE.search(concept) or _COUNT_CUE_RE.search(normalized or spaced):
        entity = " ".join(_COUNT_WORDS_RE.sub(" ", normalized or spaced).split()).strip()
        if entity and entity.lower() != concept.lower():
            terms.append(entity)
    return tuple(dict.fromkeys(term for term in terms if term))


def _unique_terms(work: list[_ConceptWork]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in work:
        for term in item.terms:
            key = term.lower()
            if key not in seen:
                seen.add(key)
                ordered.append(term)
    return ordered


def _collect_hits(
    work: list[_ConceptWork],
    cache: dict[str, tuple[MetadataSearchResult, ...]],
) -> tuple[
    dict[TableKey, int],
    dict[ColumnKey, int],
    dict[str, dict[ColumnKey, int]],
]:
    table_ranks: dict[TableKey, int] = {}
    column_ranks: dict[ColumnKey, int] = {}
    concept_columns: dict[str, dict[ColumnKey, int]] = {}
    for item in work:
        per_concept: dict[ColumnKey, int] = {}
        for term in item.terms:
            for result in cache.get(term, ()):
                if (
                    result.metadata_type is MetadataSearchType.TABLE
                    and result.table_name
                    and _table_allowed(item, result.schema_name, result.table_name)
                ):
                    _keep_best(
                        table_ranks,
                        (result.schema_name, result.table_name),
                        result.match_rank,
                    )
                elif (
                    result.metadata_type is MetadataSearchType.COLUMN
                    and result.table_name
                    and result.column_name
                    and _column_allowed(
                        item,
                        result.schema_name,
                        result.table_name,
                        result.column_name,
                    )
                ):
                    key = (
                        result.schema_name,
                        result.table_name,
                        result.column_name,
                    )
                    _keep_best(column_ranks, key, result.match_rank)
                    _keep_best(per_concept, key, result.match_rank)
                    _keep_best(
                        table_ranks,
                        (result.schema_name, result.table_name),
                        result.match_rank,
                    )
        concept_columns[f"{item.kind}:{item.requested}"] = per_concept
    return table_ranks, column_ranks, concept_columns


def _table_allowed(item: _ConceptWork, schema_name: str, table_name: str) -> bool:
    if item.column_name is not None:
        return _ci_eq(item.schema_name, schema_name) and _ci_eq(
            item.table_name, table_name
        )
    if item.schema_name is not None and item.table_name is not None:
        schema_table = _ci_eq(item.schema_name, schema_name) and _ci_eq(
            item.table_name, table_name
        )
        return schema_table or _ci_eq(item.table_name, table_name)
    return True


def _column_allowed(
    item: _ConceptWork,
    schema_name: str,
    table_name: str,
    column_name: str,
) -> bool:
    if item.column_name is not None:
        return (
            _ci_eq(item.schema_name, schema_name)
            and _ci_eq(item.table_name, table_name)
            and _ci_eq(item.column_name, column_name)
        )
    if item.schema_name is not None and item.table_name is not None:
        schema_table = _ci_eq(item.schema_name, schema_name) and _ci_eq(
            item.table_name, table_name
        )
        table_column = _ci_eq(item.schema_name, table_name) and _ci_eq(
            item.table_name, column_name
        )
        return schema_table or table_column
    return True


def _safe_concept_label(value: str) -> str:
    if _looks_like_sql(value):
        return "requested field"
    return value


def _ci_eq(expected: str | None, actual: str) -> bool:
    if expected is None:
        return True
    return expected.lower() == actual.lower()


def _keep_best(store: dict[K, int], key: K, rank: int) -> None:
    current = store.get(key)
    if current is None or rank < current:
        store[key] = rank


def _reason(rank: int) -> MetadataMatchReason:
    return _RANK_REASON.get(rank, MetadataMatchReason.DESCRIPTION)


def _score(rank: int) -> int:
    return _RANK_SCORE.get(rank, 40)


def _limit_tables(
    candidates: list[MetadataTableCandidate],
) -> list[MetadataTableCandidate]:
    return candidates[: settings.AI_MAX_METADATA_TABLES]


def _merge_column_candidates(
    primary: list[MetadataColumnCandidate],
    extra: list[MetadataColumnCandidate],
) -> list[MetadataColumnCandidate]:
    seen: set[UUID] = set()
    merged: list[MetadataColumnCandidate] = []
    for item in (*primary, *extra):
        if item.column_id in seen:
            continue
        seen.add(item.column_id)
        merged.append(item)
        if len(merged) >= settings.AI_MAX_METADATA_COLUMNS:
            break
    return merged


def _table_candidates(
    ranks: dict[TableKey, int],
    tables: dict[TableKey, DataSourceTable],
    primary_keys: dict[UUID, list[DataSourceColumn]],
) -> list[MetadataTableCandidate]:
    candidates: list[MetadataTableCandidate] = []
    for key, rank in ranks.items():
        table = tables.get(key)
        if table is None:
            continue
        keys = [
            MetadataPrimaryKey(column_id=column.id, column_name=column.name)
            for column in primary_keys.get(table.id, ())
        ]
        candidates.append(
            MetadataTableCandidate(
                table_id=table.id,
                schema_name=table.schema.name,
                table_name=table.name,
                match_reason=_reason(rank),
                relevance_score=_score(rank),
                primary_key_columns=keys,
            )
        )
    candidates.sort(
        key=lambda item: (-item.relevance_score, item.schema_name, item.table_name)
    )
    return candidates


def _column_candidates(
    ranks: dict[ColumnKey, int],
    columns: dict[ColumnKey, DataSourceColumn],
) -> list[MetadataColumnCandidate]:
    candidates: list[MetadataColumnCandidate] = []
    for key, rank in ranks.items():
        column = columns.get(key)
        if column is None:
            continue
        candidates.append(_column_from_loaded(column, match_rank=rank))
    candidates.sort(
        key=lambda item: (
            -item.relevance_score,
            item.schema_name,
            item.table_name,
            item.column_name,
        )
    )
    return candidates


def _column_from_loaded(
    column: DataSourceColumn, *, match_rank: int
) -> MetadataColumnCandidate:
    return MetadataColumnCandidate(
        column_id=column.id,
        table_id=column.table_id,
        schema_name=column.table.schema.name,
        table_name=column.table.name,
        column_name=column.name,
        data_type=column.data_type,
        is_primary_key=column.is_primary_key,
        is_unique=bool(column.is_unique),
        description=(column.description or "").strip()[:200] or None,
        match_reason=_reason(match_rank),
        relevance_score=_score(match_rank),
    )


_COUNT_WORDS_RE = re.compile(
    r"\b(count|counts|number|numbers|num|how many|total number|no\.?)\b(\s+of)?",
    re.IGNORECASE,
)
_COUNT_CUE_RE = re.compile(
    r"^\s*(how many|count\b|(total\s+)?(number|no\.?|#)\s+of\b)|\bcounts?\s*$",
    re.IGNORECASE,
)
_GENERIC_ROW_WORDS = frozenset({"", "record", "records", "row", "rows", "entry", "entries"})
_COUNT_AGGREGATIONS = {AggregationType.COUNT, AggregationType.COUNT_DISTINCT}
_MEASURE_AGGREGATIONS = {AggregationType.SUM, AggregationType.AVG}


def _apply_glossary(
    work: list[_ConceptWork], glossary: list[GlossaryEntry]
) -> tuple[list[_ConceptWork], dict[str, GlossaryEntry]]:
    """Point glossary-defined concepts at their declared catalog column."""
    if not glossary:
        return work, {}
    hits: dict[str, GlossaryEntry] = {}
    updated: list[_ConceptWork] = []
    for item in work:
        entry = (
            match_glossary(glossary, item.requested)
            if item.kind != "subject" and item.column_name is None
            else None
        )
        if entry is None:
            updated.append(item)
            continue
        hits[f"{item.kind}:{item.requested}"] = entry
        if not entry.column:
            updated.append(item)
            continue
        parts = entry.column.split(".")
        schema_name = parts[-3] if len(parts) == 3 else None
        table_name = parts[-2] if len(parts) >= 2 else None
        updated.append(
            _ConceptWork(
                kind=item.kind,
                requested=item.requested,
                terms=(parts[-1],),
                schema_name=schema_name,
                table_name=table_name,
                column_name=parts[-1] if table_name else None,
            )
        )
    return updated, hits


def _relevant_glossary(
    glossary: list[GlossaryEntry],
    hits: dict[str, GlossaryEntry],
    message: str | None,
) -> list[str]:
    chosen = list(dict.fromkeys(id(entry) for entry in hits.values()))
    by_id = {id(entry): entry for entry in glossary}
    lowered = f" {normalize_concept(message or '')} "
    for entry in glossary:
        if id(entry) in chosen:
            continue
        if any(
            f" {variant} " in lowered
            for name in (entry.term, *entry.synonyms)
            for variant in concept_variants(name)
        ):
            chosen.append(id(entry))
    return [by_id[key].describe() for key in chosen[:10] if key in by_id]


def _refine_metric(
    item: ConceptColumnResolution,
    *,
    aggregation: AggregationType,
    glossary_entry: GlossaryEntry | None,
    tables: list[MetadataTableCandidate],
    columns: list[MetadataColumnCandidate],
) -> ConceptColumnResolution:
    """Schema-aware metric mapping: counts, numeric measures, glossary notes."""
    if glossary_entry is not None and glossary_entry.aggregation is not None:
        aggregation = glossary_entry.aggregation
    note = (
        f"glossary: {glossary_entry.describe()}"
        if glossary_entry is not None and item.resolved
        else None
    )
    normalized = normalize_concept(item.requested)
    counting = aggregation in _COUNT_AGGREGATIONS or bool(
        _COUNT_CUE_RE.search(item.requested) or _COUNT_CUE_RE.search(normalized)
    ) or bool(re.search(r"\bdistinct\b", item.requested, flags=re.IGNORECASE))
    if counting:
        return _resolve_count(item, note=note, tables=tables, columns=columns)
    if aggregation in _MEASURE_AGGREGATIONS and item.candidates:
        numeric = [c for c in item.candidates if is_numeric_type(c.data_type)]
        if not numeric:
            # SUM/AVG over a text/date column would be a silent substitution.
            return ConceptColumnResolution(requested=item.requested, resolved=False)
        item = item.model_copy(
            update={"candidates": numeric, "ambiguous": item.ambiguous and len(numeric) > 1}
        )
    return item.model_copy(update={"resolution_note": note}) if note else item


def _resolve_count(
    item: ConceptColumnResolution,
    *,
    note: str | None,
    tables: list[MetadataTableCandidate],
    columns: list[MetadataColumnCandidate],
) -> ConceptColumnResolution:
    # Underscore / measure noise ("products sold", "sales_count") before entity match.
    cleaned = _MEASURE_NOISE_RE.sub(" ", normalize_concept(item.requested))
    cleaned = re.sub(r"\bdistinct\b", " ", cleaned, flags=re.IGNORECASE)
    entity = normalize_concept(_COUNT_WORDS_RE.sub(" ", cleaned))
    wanted = concept_variants(entity) if entity else set()
    best = item.candidates[0] if item.resolved and item.candidates else None
    exact_column = best is not None and (
        best.match_reason is MetadataMatchReason.EXACT or is_identifier_column(best)
    )
    table = None
    if not exact_column:
        table = next(
            (t for t in tables if wanted and concept_variants(t.table_name) & wanted),
            None,
        )
        if table is None and best is None and entity in _GENERIC_ROW_WORDS and len(tables) == 1:
            table = tables[0]
    if best is not None and table is None:
        ref = f"{best.schema_name}.{best.table_name}.{best.column_name}"
        if best.is_primary_key:
            detail = f"COUNT(*) of {best.schema_name}.{best.table_name} rows"
        elif is_identifier_column(best):
            detail = f"COUNT(DISTINCT {ref})"
        elif is_numeric_type(best.data_type):
            detail = (
                f"{ref} is a numeric per-row measure: SUM it for a total count; "
                "use COUNT(*) only when counting rows"
            )
        else:
            detail = f"COUNT(DISTINCT {ref})"
        return item.model_copy(update={"resolution_note": note or detail})
    activity_cue = bool(
        _MEASURE_NOISE_RE.search(item.requested)
        or _MEASURE_NOISE_RE.search(normalize_concept(item.requested))
        or re.search(r"\bdistinct\b", item.requested, flags=re.IGNORECASE)
    )
    id_names = {f"{variant}_id" for variant in wanted} | {f"{variant}id" for variant in wanted}
    id_column = next(
        (c for c in columns if c.column_name.lower() in id_names),
        None,
    )
    # "products sold" / "distinct products" → COUNT DISTINCT product_id, not catalog row count.
    if id_column is not None and (table is None or activity_cue):
        ref = f"{id_column.schema_name}.{id_column.table_name}.{id_column.column_name}"
        return ConceptColumnResolution(
            requested=item.requested,
            resolved=True,
            candidates=[id_column],
            resolution_note=note or f"COUNT(DISTINCT {ref})",
        )
    if table is not None:
        pk_ids = {pk.column_id for pk in table.primary_key_columns}
        pk_columns = [c for c in columns if c.column_id in pk_ids]
        return ConceptColumnResolution(
            requested=item.requested,
            resolved=True,
            candidates=pk_columns[:1],
            resolution_note=note or f"COUNT(*) of {table.schema_name}.{table.table_name} rows",
        )
    return item


def _column_resolution(
    item: _ConceptWork,
    hits_by_concept: dict[str, dict[ColumnKey, int]],
    columns: dict[ColumnKey, DataSourceColumn],
    allowed_column_ids: set[UUID],
) -> ConceptColumnResolution:
    ranks = hits_by_concept.get(f"{item.kind}:{item.requested}", {})
    candidates = [
        candidate
        for candidate in _column_candidates(ranks, columns)
        if candidate.column_id in allowed_column_ids
    ]
    if not candidates:
        return ConceptColumnResolution(
            requested=item.requested,
            resolved=False,
            ambiguous=False,
            candidates=[],
        )
    best = max(candidate.relevance_score for candidate in candidates)
    top = [candidate for candidate in candidates if candidate.relevance_score == best]
    exact_tie = len(top) > 1 and top[0].match_reason is MetadataMatchReason.EXACT
    return ConceptColumnResolution(
        requested=item.requested,
        resolved=True,
        ambiguous=exact_tie,
        candidates=top if exact_tie else candidates[:5],
    )


def _unresolved_concepts(
    *,
    intent: AIIntent,
    work: list[_ConceptWork],
    table_candidates: list[MetadataTableCandidate],
    metrics: list[ConceptColumnResolution],
    dimensions: list[ConceptColumnResolution],
    filters: list[ConceptColumnResolution],
    sorts: list[ConceptColumnResolution],
    time_candidates: list[MetadataColumnCandidate],
) -> list[str]:
    unresolved: list[str] = []
    if (
        intent.intent in _DATA_INTENTS
        and any(item.kind == "subject" for item in work)
        and not table_candidates
    ):
        unresolved.append(
            _safe_concept_label(
                next(item.requested for item in work if item.kind == "subject")
            )
        )
    for resolution in (*metrics, *dimensions, *filters, *sorts):
        if not resolution.resolved:
            unresolved.append(_safe_concept_label(resolution.requested))
    if intent.time_range is not None and not time_candidates:
        unresolved.append("time range")
    return list(dict.fromkeys(unresolved))


def _ambiguous_items(
    *,
    metrics: list[ConceptColumnResolution],
    dimensions: list[ConceptColumnResolution],
    filters: list[ConceptColumnResolution],
    sorts: list[ConceptColumnResolution],
    time_candidates: list[MetadataColumnCandidate],
    table_candidates: list[MetadataTableCandidate],
    require_time_disambiguation: bool = False,
) -> list[ConceptColumnResolution]:
    ambiguous = [
        item for item in (*metrics, *dimensions, *filters, *sorts) if item.ambiguous
    ]
    if require_time_disambiguation and len(time_candidates) > 1:
        ambiguous.append(
            ConceptColumnResolution(
                requested="time range",
                resolved=False,
                ambiguous=True,
                candidates=time_candidates,
            )
        )
    exact_tables = [
        item
        for item in table_candidates
        if item.match_reason is MetadataMatchReason.EXACT
    ]
    names = {item.table_name.lower() for item in exact_tables}
    if len(exact_tables) > 1 and len(names) == 1:
        ambiguous.append(
            ConceptColumnResolution(
                requested=exact_tables[0].table_name,
                resolved=False,
                ambiguous=True,
            )
        )
    return ambiguous


def _clarification_question(
    ambiguous: list[ConceptColumnResolution],
    unresolved: list[str],
    table_candidates: list[MetadataTableCandidate],
) -> str:
    for item in ambiguous:
        if item.requested == "time range" and item.candidates:
            options = " or ".join(
                f"{candidate.schema_name}.{candidate.table_name}.{candidate.column_name}"
                for candidate in item.candidates[:4]
            )
            return f"Which date column should I use: {options}?"
        if item.candidates:
            return (
                f"Which {item.requested} source should I use: "
                f"{_choice_labels(item.candidates)}?"
            )
        matching = [
            table
            for table in table_candidates
            if table.table_name.lower() == item.requested.lower()
        ]
        if matching:
            options = " or ".join(
                f"{table.schema_name}.{table.table_name}" for table in matching[:4]
            )
            return f"Which {item.requested} table should I use: {options}?"
    if unresolved:
        return f"I could not find metadata for: {', '.join(unresolved)}."
    return "I need more detail before I can use the catalog for this request."


def _choice_labels(candidates: list[MetadataColumnCandidate]) -> str:
    schemas = list(dict.fromkeys(item.schema_name for item in candidates))
    if len(schemas) > 1:
        labels = schemas[:4]
    else:
        tables = list(
            dict.fromkeys(
                f"{item.schema_name}.{item.table_name}" for item in candidates
            )
        )
        if len(tables) > 1:
            labels = tables[:4]
        else:
            labels = [
                f"{item.schema_name}.{item.table_name}.{item.column_name}"
                for item in candidates[:4]
            ]
    if len(labels) == 2:
        return f"{labels[0]} or {labels[1]}"
    return ", ".join(labels)
