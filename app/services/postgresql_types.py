"""Normalize PostgreSQL native types into semantic analysis types."""

from __future__ import annotations

import re

_PRECISION = re.compile(r"\([^)]*\)")

_TYPE_MAP: dict[str, str] = {
    "smallint": "integer",
    "integer": "integer",
    "bigint": "integer",
    "int2": "integer",
    "int4": "integer",
    "int8": "integer",
    "int": "integer",
    "serial": "integer",
    "smallserial": "integer",
    "bigserial": "integer",
    "numeric": "numeric",
    "decimal": "numeric",
    "real": "float",
    "double precision": "float",
    "float4": "float",
    "float8": "float",
    "float": "float",
    "boolean": "boolean",
    "bool": "boolean",
    "character": "string",
    "character varying": "string",
    "varchar": "string",
    "char": "string",
    "bpchar": "string",
    "text": "string",
    "name": "string",
    "citext": "string",
    "date": "date",
    "timestamp": "datetime",
    "timestamp without time zone": "datetime",
    "timestamp with time zone": "datetime",
    "timestamptz": "datetime",
    "time": "time",
    "time without time zone": "time",
    "time with time zone": "time",
    "timetz": "time",
    "uuid": "uuid",
    "json": "json",
    "jsonb": "json",
    "bytea": "binary",
}

UNKNOWN_POSTGRES_TYPE = "other"


def normalize_postgres_type(database_type: str) -> str:
    """Map a native PostgreSQL type name to a semantic type.

    Unknown and extension types become ``other``. Precision and length
    modifiers such as ``numeric(10, 2)`` are ignored.
    """
    key = _canonical_type_name(database_type)
    return _TYPE_MAP.get(key, UNKNOWN_POSTGRES_TYPE)


def _canonical_type_name(database_type: str) -> str:
    stripped = database_type.strip().lower()
    without_precision = _PRECISION.sub("", stripped)
    return " ".join(without_precision.split())
