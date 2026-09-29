"""Structured output schema helpers for LLM requests."""

from __future__ import annotations

import copy
import re
from typing import Any

from pydantic import BaseModel

from app.ai.llm.models import StructuredOutputConfig


def structured_output_from_model(
    model_type: type[BaseModel],
    *,
    strict: bool = True,
    name: str | None = None,
) -> StructuredOutputConfig:
    """Build LLM structured-output config from a Pydantic response model."""
    schema = model_type.model_json_schema()
    if strict:
        schema = make_openai_strict_schema(schema)
    return StructuredOutputConfig(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": name or _schema_name(model_type),
                "schema": schema,
                "strict": strict,
            },
        }
    )


def json_object_output() -> StructuredOutputConfig:
    """Request generic JSON object output from the provider."""
    return StructuredOutputConfig()


def response_schema(model_type: type[BaseModel]) -> dict[str, Any]:
    """Return the JSON schema for a structured response model."""
    return model_type.model_json_schema()


def make_openai_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Pydantic JSON schema for OpenAI/OpenRouter strict mode.

    Strict providers require every object to set ``additionalProperties: false``
    and list every property key in ``required``. They also reject ``$ref`` nodes
    that carry sibling keywords such as ``default``.
    """
    normalized = copy.deepcopy(schema)
    _normalize_schema_node(normalized)
    return normalized


def _normalize_schema_node(node: Any) -> None:
    if not isinstance(node, dict):
        return

    # OpenAI rejects $ref with sibling keywords (e.g. default).
    if "$ref" in node:
        ref = node["$ref"]
        node.clear()
        node["$ref"] = ref
        return

    node.pop("default", None)

    for key in ("anyOf", "oneOf", "allOf"):
        variants = node.get(key)
        if isinstance(variants, list):
            for variant in variants:
                _normalize_schema_node(variant)

    properties = node.get("properties")
    if isinstance(properties, dict):
        node["type"] = "object"
        node["additionalProperties"] = False
        node["required"] = list(properties.keys())
        for child in properties.values():
            _normalize_schema_node(child)

    items = node.get("items")
    if isinstance(items, dict):
        _normalize_schema_node(items)
    elif isinstance(items, list):
        for item in items:
            _normalize_schema_node(item)

    defs = node.get("$defs")
    if isinstance(defs, dict):
        for child in defs.values():
            _normalize_schema_node(child)

    definitions = node.get("definitions")
    if isinstance(definitions, dict):
        for child in definitions.values():
            _normalize_schema_node(child)


def _schema_name(model_type: type[BaseModel]) -> str:
    raw = model_type.__name__
    normalized = re.sub(r"[^A-Za-z0-9_-]", "_", raw)
    return normalized[:64] or "response"
