from __future__ import annotations

import pytest

from app.services.discovery_types import SchemaFilter
from app.services.postgresql_types import UNKNOWN_POSTGRES_TYPE, normalize_postgres_type
from app.services.schema_filters import is_system_schema, should_include_schema


@pytest.mark.parametrize(
    ("native", "expected"),
    [
        ("smallint", "integer"),
        ("integer", "integer"),
        ("bigint", "integer"),
        ("int2", "integer"),
        ("int4", "integer"),
        ("int8", "integer"),
        ("numeric", "numeric"),
        ("decimal", "numeric"),
        ("numeric(10,2)", "numeric"),
        ("real", "float"),
        ("double precision", "float"),
        ("float8", "float"),
        ("boolean", "boolean"),
        ("bool", "boolean"),
        ("character", "string"),
        ("character varying", "string"),
        ("character varying(255)", "string"),
        ("varchar", "string"),
        ("varchar(64)", "string"),
        ("text", "string"),
        ("date", "date"),
        ("timestamp", "datetime"),
        ("timestamp with time zone", "datetime"),
        ("timestamp without time zone", "datetime"),
        ("timestamp(6) with time zone", "datetime"),
        ("timestamptz", "datetime"),
        ("time", "time"),
        ("time with time zone", "time"),
        ("time without time zone", "time"),
        ("uuid", "uuid"),
        ("json", "json"),
        ("jsonb", "json"),
        ("bytea", "binary"),
    ],
)
def test_normalize_postgres_type(native: str, expected: str) -> None:
    assert normalize_postgres_type(native) == expected


def test_unknown_postgres_type_is_other() -> None:
    assert normalize_postgres_type("hstore") == UNKNOWN_POSTGRES_TYPE
    assert normalize_postgres_type("geometry") == UNKNOWN_POSTGRES_TYPE
    assert normalize_postgres_type("ARRAY") == UNKNOWN_POSTGRES_TYPE


def test_system_schemas_are_detected() -> None:
    assert is_system_schema("pg_catalog") is True
    assert is_system_schema("information_schema") is True
    assert is_system_schema("pg_toast") is True
    assert is_system_schema("pg_temp_1") is True
    assert is_system_schema("pg_toast_temp_1") is True
    assert is_system_schema("public") is False
    assert is_system_schema("analytics") is False


def test_should_include_schema_filters_system_and_allow_deny_lists() -> None:
    assert should_include_schema("public") is True
    assert should_include_schema("pg_catalog") is False
    assert (
        should_include_schema(
            "analytics",
            SchemaFilter(include_schemas=frozenset({"public"})),
        )
        is False
    )
    assert (
        should_include_schema(
            "public",
            SchemaFilter(include_schemas=frozenset({"public"})),
        )
        is True
    )
    assert (
        should_include_schema(
            "legacy",
            SchemaFilter(exclude_schemas=frozenset({"legacy"})),
        )
        is False
    )
    assert (
        should_include_schema(
            "_timescaledb_internal",
            SchemaFilter(exclude_schemas=frozenset({"_timescaledb_internal"})),
        )
        is False
    )
