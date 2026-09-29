"""Format Phase 4 resolved metadata into bounded SQL-generation prompt context."""

from __future__ import annotations

from dataclasses import dataclass

from app.ai.metadata_types import ResolvedMetadataContext
from app.core.config import settings


@dataclass(frozen=True)
class SchemaPromptContext:
    text: str
    table_count: int
    column_count: int
    relationship_count: int
    truncated: bool


def build_schema_prompt_context(
    metadata: ResolvedMetadataContext,
    *,
    max_tables: int | None = None,
    max_columns: int | None = None,
    max_relationships: int | None = None,
    max_chars: int | None = None,
) -> SchemaPromptContext:
    """Render catalog facts for the SQL prompt. Never invents schema."""
    table_limit = (
        max_tables if max_tables is not None else settings.AI_MAX_METADATA_TABLES
    )
    column_limit = (
        max_columns if max_columns is not None else settings.AI_MAX_METADATA_COLUMNS
    )
    relationship_limit = (
        max_relationships
        if max_relationships is not None
        else settings.AI_MAX_METADATA_RELATIONSHIPS
    )
    char_limit = (
        max_chars if max_chars is not None else settings.AI_SQL_MAX_SCHEMA_CHARS
    )

    tables = list(metadata.tables[:table_limit])
    table_keys = {
        (table.schema_name.lower(), table.table_name.lower()) for table in tables
    }
    columns = _dedupe_columns(metadata)[:column_limit]
    # Only expose relationships whose endpoints are both present in selected tables.
    complete_relationships = [
        rel
        for rel in metadata.relationships
        if (rel.source_schema.lower(), rel.source_table.lower()) in table_keys
        and (rel.target_schema.lower(), rel.target_table.lower()) in table_keys
    ]
    relationships = complete_relationships[:relationship_limit]
    truncated = (
        len(metadata.tables) > table_limit
        or len(_dedupe_columns(metadata)) > column_limit
        or len(complete_relationships) > relationship_limit
        or len(complete_relationships) < len(metadata.relationships)
    )

    lines: list[str] = ["Authorized catalog metadata (read-only):"]
    if not tables and not columns:
        lines.append("(no tables or columns available)")
    else:
        lines.append("Tables:")
        for table in tables:
            pk = ", ".join(pk.column_name for pk in table.primary_key_columns) or "none"
            lines.append(f"- {table.schema_name}.{table.table_name} (pk={pk})")
        lines.append("Columns:")
        for column in columns:
            pk_flag = " pk" if column.is_primary_key else ""
            lines.append(
                f"- {column.schema_name}.{column.table_name}.{column.column_name}"
                f" ({column.data_type}{pk_flag})"
            )
        if relationships:
            lines.append("Relationships:")
            for rel in relationships:
                lines.append(
                    f"- {rel.source_schema}.{rel.source_table}.{rel.source_column}"
                    f" -> {rel.target_schema}.{rel.target_table}.{rel.target_column}"
                    f" ({rel.relationship_type.value})"
                )

    if metadata.unresolved_concepts:
        unresolved = ", ".join(metadata.unresolved_concepts[:20])
        lines.append(f"Unresolved concepts: {unresolved}")

    text = "\n".join(lines)
    if len(text) > char_limit:
        text = text[: max(0, char_limit - 20)].rstrip() + "\n...[truncated]"
        truncated = True

    return SchemaPromptContext(
        text=text,
        table_count=len(tables),
        column_count=len(columns),
        relationship_count=len(relationships),
        truncated=truncated,
    )


def _dedupe_columns(metadata: ResolvedMetadataContext) -> list:
    """Merge top-level columns with resolved metric/dimension/time candidates."""
    from app.ai.metadata_types import MetadataColumnCandidate

    seen: set[tuple[str, str, str]] = set()
    out: list[MetadataColumnCandidate] = []

    def add(column: MetadataColumnCandidate) -> None:
        key = (
            column.schema_name.lower(),
            column.table_name.lower(),
            column.column_name.lower(),
        )
        if key in seen:
            return
        seen.add(key)
        out.append(column)

    for column in metadata.columns:
        add(column)
    for column in metadata.resolved_time_columns:
        add(column)
    for group in (
        metadata.resolved_metrics,
        metadata.resolved_dimensions,
        metadata.resolved_filters,
    ):
        for item in group:
            for candidate in item.candidates:
                add(candidate)
    return out


def has_usable_schema(metadata: ResolvedMetadataContext) -> bool:
    return bool(metadata.tables) or bool(metadata.columns)
