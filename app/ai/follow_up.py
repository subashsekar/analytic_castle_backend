"""Resolve conversational follow-ups from the prior analysis context.

Stores last SQL / analysis spec / intent on the session agent-state payload
(already scoped by user, workspace, and organization). Follow-up messages only
patch filters, dimensions, or date ranges; every resulting SQL still goes through
the existing validate → authorize → execute path.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from typing import Any

from app.ai.intent_types import (
    AIDimension,
    AIIntent,
    AIIntentType,
    AIOperationType,
    AITimeRange,
    TimeRangePreset,
)
from app.ai.metadata_types import ResolvedMetadataContext
from app.ai.state.models import AgentStatePayload

LAST_SQL_KEY = "last_sql"
LAST_SPEC_KEY = "last_analysis_spec"
LAST_INTENT_KEY = "last_intent"
_MAX_STORED_CHARS = 4_000

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}  # fmt: skip

_STRONG_FOLLOW_UP = re.compile(
    r"(?i)("
    r"\bbreak\s+(?:that|it|this)\s+down\b"
    r"|\bsame\s+(?:for|thing|query|analysis)\b"
    r"|\bonly\s+\w+"
    r"|\bjust\s+(?:for|in|the)\s+\w+"
    r"|\bcompare\s+(?:with|to|against)\b"
    r"|\bvs\.?\s+(?:last|previous|prior)\b"
    r"|\bversus\s+(?:last|previous|prior)\b"
    r"|\blast\s+year\b"
    r"|\bprevious\s+year\b"
    r"|\byear\s+over\s+year\b"
    r"|\byoy\b"
    r"|\bfilter\s+(?:to|by|on)\b"
    r")"
)
_SHORT_BY_DIM = re.compile(r"(?i)\bby\s+[a-z_][\w\s-]{0,40}")
_BREAKDOWN_RE = re.compile(
    r"(?i)\b(?:break\s+(?:that|it|this)\s+down\s+by|broken\s+down\s+by|by)\s+"
    r"([a-z_][\w\s-]{0,40})"
)
_ONLY_MONTH_RE = re.compile(
    r"(?i)\b(?:only|just|for|in)\s+("
    + "|".join(sorted(_MONTHS, key=len, reverse=True))
    + r")(?:\s*((?:19|20|21)\d{2}))?\b"
)
_LAST_YEAR_RE = re.compile(
    r"(?i)\b(?:compare\s+(?:with|to|against)\s+)?(?:last|previous)\s+year\b|\byoy\b|\byear\s+over\s+year\b"
)
_PRONOUN_RE = re.compile(r"(?i)\b(that|those|it|this|same)\b")


def pack_analysis_context(
    *,
    sql: str,
    analysis_spec: str,
    intent: AIIntent,
) -> dict[str, str]:
    """Compact prior query context for agent-state ``custom`` (string values only)."""
    intent_payload = intent.model_dump(
        mode="json",
        exclude_none=True,
        exclude={"clarification_question", "unsupported_reason"},
    )
    return {
        LAST_SQL_KEY: _clip(sql),
        LAST_SPEC_KEY: _clip(analysis_spec),
        LAST_INTENT_KEY: _clip(json.dumps(intent_payload, separators=(",", ":"))),
    }


def unpack_prior_intent(custom: dict[str, str] | AgentStatePayload | None) -> AIIntent | None:
    raw_custom = custom.custom if isinstance(custom, AgentStatePayload) else (custom or {})
    blob = raw_custom.get(LAST_INTENT_KEY)
    if not blob:
        return None
    try:
        data = json.loads(blob)
        if not isinstance(data, dict):
            return None
        return AIIntent.model_validate(data)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def prior_sql(custom: dict[str, str] | AgentStatePayload | None) -> str | None:
    raw_custom = custom.custom if isinstance(custom, AgentStatePayload) else (custom or {})
    value = (raw_custom.get(LAST_SQL_KEY) or "").strip()
    return value or None


def prior_analysis_spec(custom: dict[str, str] | AgentStatePayload | None) -> str | None:
    raw_custom = custom.custom if isinstance(custom, AgentStatePayload) else (custom or {})
    value = (raw_custom.get(LAST_SPEC_KEY) or "").strip()
    return value or None


_FULL_QUESTION_START = re.compile(
    r"(?i)^(what|how|show|list|give|tell|find|get|which|who|when|where)\b"
)


def looks_like_follow_up(message: str, current: AIIntent) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    if _STRONG_FOLLOW_UP.search(text):
        return True
    words = text.split()
    # Short "by region" deltas only — not full questions that happen to include "by".
    if (
        len(words) <= 8
        and _SHORT_BY_DIM.search(text)
        and not _FULL_QUESTION_START.search(text)
    ):
        return True
    return (
        len(words) <= 8
        and not current.metrics
        and bool(_PRONOUN_RE.search(text))
        and not _FULL_QUESTION_START.search(text)
    )


def apply_follow_up_intent(
    prior: AIIntent,
    message: str,
    current: AIIntent,
    *,
    metadata: ResolvedMetadataContext | None = None,
    today: date | None = None,
) -> AIIntent:
    """Patch prior intent with follow-up deltas; preserve original metrics/subject."""
    today = today or date.today()
    text = (message or "").strip()
    updates: dict[str, Any] = {
        "requires_clarification": False,
        "clarification_question": None,
        "requires_data_access": True,
        "requires_metadata": True,
        "confidence": current.confidence or prior.confidence,
    }

    # Metrics / subject: keep prior unless the follow-up explicitly supplies new ones.
    metrics = list(current.metrics) if current.metrics else list(prior.metrics)
    subject = current.subject or prior.subject
    updates["metrics"] = metrics
    updates["subject"] = subject
    updates["sort"] = current.sort or prior.sort
    updates["requested_limit"] = current.requested_limit or prior.requested_limit
    updates["safe_limit"] = current.safe_limit or prior.safe_limit

    dimensions = list(prior.dimensions)
    filters = list(prior.filters)
    time_range = prior.time_range
    intent_type = prior.intent
    operation = prior.operation

    breakdown = _BREAKDOWN_RE.search(text)
    if breakdown:
        dim_name = _clean_dimension_token(breakdown.group(1))
        resolved = _ground_dimension(dim_name, metadata)
        if resolved is None:
            return prior.model_copy(
                update={
                    "requires_clarification": True,
                    "clarification_question": (
                        f"Which field should I break this down by? "
                        f"'{dim_name}' is not in the connected schema."
                    ),
                    "confidence": current.confidence,
                }
            )
        dimensions = _upsert_dimension(dimensions, resolved)
        intent_type = AIIntentType.RANKING if intent_type == AIIntentType.AGGREGATION else intent_type
        if intent_type in {AIIntentType.UNKNOWN, AIIntentType.SUMMARY, AIIntentType.DATA_LOOKUP}:
            intent_type = AIIntentType.ANALYTICAL_QUERY
        operation = AIOperationType.GROUP_BY

    if current.dimensions:
        for item in current.dimensions:
            grounded = _ground_dimension(item.name, metadata) or item.name
            dimensions = _upsert_dimension(dimensions, grounded)

    month_match = _ONLY_MONTH_RE.search(text)
    if month_match and not _LAST_YEAR_RE.search(text):
        month = _MONTHS[month_match.group(1).lower()]
        year = int(month_match.group(2)) if month_match.group(2) else _year_from_prior(prior, today)
        time_range = _month_range(year, month)
        # "only March" replaces the prior window rather than stacking filters.
        filters = [f for f in filters if f.field.lower() not in {"month", "period"}]

    if _LAST_YEAR_RE.search(text):
        time_range = _comparison_with_last_year(prior.time_range, today)
        intent_type = AIIntentType.COMPARISON
        operation = AIOperationType.COMPARE

    if current.time_range is not None and not month_match and not _LAST_YEAR_RE.search(text):
        time_range = current.time_range

    if current.filters:
        filters = _merge_filters(filters, list(current.filters))

    updates.update(
        {
            "intent": intent_type if intent_type is not AIIntentType.UNKNOWN else current.intent,
            "operation": operation or current.operation,
            "dimensions": dimensions,
            "filters": filters,
            "time_range": time_range,
        }
    )
    return prior.model_copy(update=updates)


def enrich_conversation_context(
    base: str | None,
    *,
    last_sql: str | None = None,
    last_analysis_spec: str | None = None,
) -> str | None:
    """Append prior SQL/spec so generation can modify the previous query."""
    parts: list[str] = []
    if base and base.strip():
        parts.append(base.strip())
    if last_analysis_spec and last_analysis_spec.strip():
        parts.append("Previous analysis specification:\n" + last_analysis_spec.strip())
    if last_sql and last_sql.strip():
        sql = " ".join(last_sql.split())
        if len(sql) > 800:
            sql = sql[:797] + "..."
        parts.append("Previous analysis SQL:\n" + sql)
    if not parts:
        return None
    parts.append(
        "Follow-up rule: modify only the filters, dimensions, metrics, or date range "
        "requested in the latest user message; preserve the rest of the prior query intent."
    )
    return "\n\n".join(parts)


def _clip(value: str, limit: int = _MAX_STORED_CHARS) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _clean_dimension_token(raw: str) -> str:
    token = " ".join(raw.strip().split())
    token = re.sub(r"(?i)\s+(please|thanks|now|instead)$", "", token).strip(" .,?")
    return token


def _ground_dimension(name: str, metadata: ResolvedMetadataContext | None) -> str | None:
    if not name:
        return None
    needle = name.strip().lower().replace(" ", "_")
    needle_words = name.strip().lower()
    if metadata is None:
        return name.strip()
    for column in metadata.columns:
        col = column.column_name.lower()
        if col == needle or col.replace("_", " ") == needle_words:
            return column.column_name
    for item in metadata.resolved_dimensions:
        if item.requested.lower() in {needle_words, needle} and item.resolved:
            return item.requested
    # Catalog present but no match → genuinely ambiguous.
    if metadata.columns or metadata.tables:
        return None
    return name.strip()


def _upsert_dimension(dimensions: list[AIDimension], name: str) -> list[AIDimension]:
    lowered = name.lower()
    if any(d.name.lower() == lowered for d in dimensions):
        return dimensions
    return [*dimensions, AIDimension(name=name)]


def _merge_filters(base: list, incoming: list) -> list:
    by_field = {f.field.lower(): f for f in base}
    for item in incoming:
        by_field[item.field.lower()] = item
    return list(by_field.values())


def _year_from_prior(prior: AIIntent, today: date) -> int:
    tr = prior.time_range
    if tr is not None:
        if tr.end_date is not None:
            return tr.end_date.year
        if tr.start_date is not None:
            return tr.start_date.year
    return today.year


def _month_range(year: int, month: int) -> AITimeRange:
    start = date(year, month, 1)
    if month == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)
    return AITimeRange(preset=TimeRangePreset.CUSTOM_RANGE, start_date=start, end_date=end)


def _shift_year(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(year=value.year + years, day=28)


def _comparison_with_last_year(
    prior_range: AITimeRange | None,
    today: date,
) -> AITimeRange:
    if prior_range is not None and prior_range.start_date and prior_range.end_date:
        return AITimeRange(
            preset=TimeRangePreset.CUSTOM_RANGE,
            start_date=_shift_year(prior_range.start_date, -1),
            end_date=prior_range.end_date,
        )
    if prior_range is not None and prior_range.preset is TimeRangePreset.THIS_YEAR:
        return AITimeRange(preset=TimeRangePreset.LAST_YEAR)
    # Default: last two calendar years through today.
    return AITimeRange(
        preset=TimeRangePreset.CUSTOM_RANGE,
        start_date=date(today.year - 1, 1, 1),
        end_date=today,
    )
