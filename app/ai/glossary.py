"""Data-source business glossary: metric synonyms and definitions.

Entries are user-supplied semantics ("sales" means ``public.sales.revenue``) that
the metadata resolver may use as schema evidence. Entries never contain SQL;
``column`` must name an existing catalog column to have any effect.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ai.intent_types import AggregationType, _looks_like_sql

MAX_GLOSSARY_ENTRIES = 200
_COLUMN_REF_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*(\.[A-Za-z_][A-Za-z0-9_$]*){0,2}$")
_LEADING_AGGREGATION_RE = re.compile(
    r"^(total|sum of|sum|avg|average|count of|count|number of|distinct)\s+",
    re.IGNORECASE,
)


class GlossaryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    term: str = Field(min_length=1, max_length=128)
    synonyms: list[str] = Field(default_factory=list, max_length=20)
    definition: str | None = Field(default=None, max_length=500)
    column: str | None = Field(default=None, max_length=192)
    aggregation: AggregationType | None = None

    @field_validator("term", "definition", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("synonyms")
    @classmethod
    def _clean_synonyms(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if any(len(item) > 128 for item in cleaned):
            raise ValueError("synonyms must be at most 128 characters")
        return list(dict.fromkeys(cleaned))

    @field_validator("column")
    @classmethod
    def _validate_column(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        value = value.strip()
        if not _COLUMN_REF_RE.match(value):
            raise ValueError("column must be column, table.column, or schema.table.column")
        return value

    @field_validator("definition")
    @classmethod
    def _reject_sql(cls, value: str | None) -> str | None:
        if value and _looks_like_sql(value):
            raise ValueError("definition must describe the metric, not contain SQL")
        return value

    @property
    def column_name(self) -> str | None:
        return self.column.split(".")[-1] if self.column else None

    def describe(self) -> str:
        parts = [self.term]
        if self.synonyms:
            parts.append(f"(also: {', '.join(self.synonyms[:5])})")
        if self.column:
            agg = f"{self.aggregation.value} of " if self.aggregation else ""
            parts.append(f"= {agg}{self.column}")
        if self.definition:
            parts.append(f"- {self.definition}")
        return " ".join(parts)


def parse_glossary(raw: object) -> list[GlossaryEntry]:
    """Best-effort parse of stored JSON; invalid entries are skipped, never raised."""
    if not isinstance(raw, list):
        return []
    entries: list[GlossaryEntry] = []
    for item in raw[:MAX_GLOSSARY_ENTRIES]:
        try:
            entries.append(GlossaryEntry.model_validate(item))
        except ValueError:
            continue
    return entries


def normalize_concept(value: str) -> str:
    text = " ".join(value.lower().replace("_", " ").split())
    while True:
        stripped = _LEADING_AGGREGATION_RE.sub("", text, count=1).strip()
        if not stripped or stripped == text:
            break
        text = stripped
    return text


def concept_variants(value: str) -> set[str]:
    """Normalized singular/plural forms used for case-insensitive concept matching."""
    base = normalize_concept(value)
    out = {base}
    if base.endswith("ies") and len(base) > 4:
        out.add(base[:-3] + "y")
    elif base.endswith("s") and not base.endswith("ss") and len(base) > 3:
        out.add(base[:-1])
    else:
        out.add(base + "s")
    return out


def match_glossary(
    entries: Iterable[GlossaryEntry], concept: str
) -> GlossaryEntry | None:
    """Entry whose term or synonym equals the concept (case/plural-insensitive)."""
    wanted = concept_variants(concept)
    for entry in entries:
        for name in (entry.term, *entry.synonyms):
            if concept_variants(name) & wanted:
                return entry
    return None
