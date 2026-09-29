"""In-process cache for rendered schema prompt context.

Keyed by data source + a fingerprint of the resolved catalog. Invalidate after
metadata sync so prompts never serve stale schema text.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from uuid import UUID

from app.ai.metadata_types import ResolvedMetadataContext
from app.ai.sql_generation.schema_context import (
    SchemaPromptContext,
    build_schema_prompt_context,
)

_LOCK = threading.Lock()
_CACHE: OrderedDict[str, SchemaPromptContext] = OrderedDict()
_MAX_ENTRIES = 128


def schema_fingerprint(metadata: ResolvedMetadataContext) -> str:
    parts: list[str] = [str(metadata.data_source_id)]
    for table in metadata.tables:
        parts.append(f"t:{table.schema_name}.{table.table_name}")
    for column in metadata.columns:
        parts.append(
            f"c:{column.schema_name}.{column.table_name}.{column.column_name}:"
            f"{column.data_type}:{column.description or ''}"
        )
    for rel in metadata.relationships:
        parts.append(
            f"r:{rel.source_schema}.{rel.source_table}.{rel.source_column}>"
            f"{rel.target_schema}.{rel.target_table}.{rel.target_column}"
        )
    for line in metadata.glossary:
        parts.append(f"g:{line}")
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:32]
    return f"{metadata.data_source_id}:{digest}"


def cached_schema_prompt_context(metadata: ResolvedMetadataContext) -> SchemaPromptContext:
    key = schema_fingerprint(metadata)
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit
    built = build_schema_prompt_context(metadata)
    with _LOCK:
        _CACHE[key] = built
        _CACHE.move_to_end(key)
        while len(_CACHE) > _MAX_ENTRIES:
            _CACHE.popitem(last=False)
    return built


def invalidate_schema_context_cache(data_source_id: UUID | None = None) -> None:
    """Drop cached schema prompts. Pass a data source id to scope the clear."""
    with _LOCK:
        if data_source_id is None:
            _CACHE.clear()
            return
        prefix = f"{data_source_id}:"
        for key in [key for key in _CACHE if key.startswith(prefix)]:
            del _CACHE[key]


def cache_size() -> int:
    with _LOCK:
        return len(_CACHE)
