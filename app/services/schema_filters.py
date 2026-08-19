"""Decide which PostgreSQL schemas are useful for analytics discovery."""

from __future__ import annotations

from app.services.discovery_types import SchemaFilter

# Exact catalog names excluded from discovery. Broader pg_* temp/toast
# schemas are matched by prefix in is_system_schema(). Extension schemas
# such as _timescaledb_internal are not hardcoded; pass them via
# SchemaFilter.exclude_schemas.
EXCLUDED_SYSTEM_SCHEMAS = frozenset(
    {
        "pg_catalog",
        "information_schema",
        "pg_toast",
    }
)


def is_system_schema(schema_name: str) -> bool:
    """Return True for PostgreSQL catalog, toast, and temp schemas."""
    name = schema_name.strip()
    if not name:
        return True
    lowered = name.lower()
    if lowered in EXCLUDED_SYSTEM_SCHEMAS:
        return True
    return lowered.startswith("pg_")


def should_include_schema(
    schema_name: str,
    schema_filter: SchemaFilter | None = None,
) -> bool:
    """Return whether ``schema_name`` should appear in discovery results.

    System schemas are always excluded. ``include_schemas`` is an allowlist of
    remaining names; ``exclude_schemas`` removes additional user schemas.
    Filtering is applied in Python, not by interpolating names into SQL.
    """
    name = schema_name.strip()
    if not name or is_system_schema(name):
        return False
    if schema_filter is None:
        return True
    excluded = schema_filter.exclude_schemas
    if excluded is not None and name in excluded:
        return False
    if schema_filter.include_schemas is not None:
        return name in schema_filter.include_schemas
    return True
