"""Schema-grounded SQL for planned investigation steps."""

from __future__ import annotations

import re
from datetime import date

from app.ai.analysis_pipeline import (
    _aggregation,
    _all_columns,
    _breakdown_column,
    _grain,
    _grounded_metric,
    _grounded_time_column,
    _safe_alias,
    _time_window,
)
from app.ai.analysis_profile import AnalysisProfile, QuestionKind
from app.ai.intent_types import AIIntent
from app.ai.metadata_types import (
    MetadataColumnCandidate,
    ResolvedMetadataContext,
    is_identifier_column,
    is_numeric_type,
)
from app.core.config import settings

_YOY_RE = re.compile(
    r"\b(year[- ]over[- ]year|yoy|vs\.?\s+last\s+year|compared\s+to\s+last\s+year)\b",
    re.IGNORECASE,
)
_FUNNEL_RE = re.compile(
    r"\b(funnel|conversion\s+rate|convert(ed|ion)?\s+to)\b",
    re.IGNORECASE,
)
_COHORT_RE = re.compile(
    r"\b(cohort|retention|first[- ]time|new\s+customers?)\b",
    re.IGNORECASE,
)
_TEXT_TYPE_RE = re.compile(r"char|text|string|enum|bool", re.IGNORECASE)


def investigation_context(
    message: str,
    metadata: ResolvedMetadataContext,
    *,
    intent: AIIntent | None,
    profile: AnalysisProfile,
    today: date | None = None,
) -> tuple[MetadataColumnCandidate, MetadataColumnCandidate, str, tuple[date, date]] | None:
    metric = _grounded_metric(metadata, intent) or _fallback_metric(message, metadata)
    if metric is None:
        return None
    same_table = [
        c
        for c in _all_columns(metadata)
        if (c.schema_name.lower(), c.table_name.lower())
        == (metric.schema_name.lower(), metric.table_name.lower())
    ]
    window = _time_window(message, intent, profile.kinds, today=today or date.today())
    time_col = _grounded_time_column(metadata, same_table)
    if window is None or time_col is None:
        return None
    table_sql = f'"{metric.schema_name}"."{metric.table_name}"'
    return metric, time_col, table_sql, window


def breakdown_dimensions(columns: list[MetadataColumnCandidate]) -> list[MetadataColumnCandidate]:
    dims: list[MetadataColumnCandidate] = []
    for column in columns:
        name = column.column_name.lower()
        if column.is_primary_key or name == "id":
            continue
        if name.endswith("_id") and name not in {"product_id"}:
            continue
        if _TEXT_TYPE_RE.search(column.data_type or "") or name in {"product_id", "region"}:
            dims.append(column)
    seen: set[str] = set()
    unique: list[MetadataColumnCandidate] = []
    for column in dims:
        key = column.column_name.lower()
        if key not in seen:
            seen.add(key)
            unique.append(column)
    return unique


def build_investigation_sqls(
    message: str,
    metadata: ResolvedMetadataContext,
    *,
    intent: AIIntent | None,
    profile: AnalysisProfile,
    max_steps: int,
    today: date | None = None,
) -> list[tuple[str, str, str]]:
    """Return (kind, description, sql) tuples in execution order."""
    ctx = investigation_context(message, metadata, intent=intent, profile=profile, today=today)
    if ctx is None:
        return []
    metric, time_col, table_sql, window = ctx
    same_table = [
        c
        for c in _all_columns(metadata)
        if (c.schema_name.lower(), c.table_name.lower())
        == (metric.schema_name.lower(), metric.table_name.lower())
    ]
    aggregation = _aggregation(intent, metric)
    metric_expr = "COUNT(*)" if aggregation == "COUNT" else f'{aggregation}("{metric.column_name}")'
    metric_alias = _safe_alias(metric.column_name)
    grain = _grain(message)
    start, end = window
    where = (
        f' WHERE "{time_col.column_name}" >= DATE \'{start.isoformat()}\''
        f' AND "{time_col.column_name}" < DATE \'{end.isoformat()}\''
    )
    limit = min(settings.AI_MAX_RESULT_LIMIT, 500)
    steps: list[tuple[str, str, str]] = []

    def add(kind: str, description: str, sql: str) -> None:
        if len(steps) >= max_steps:
            return
        steps.append((kind, description, sql))

    needs_investigation = bool(
        profile.kinds
        & {
            QuestionKind.DIAGNOSTIC,
            QuestionKind.COMPARISON,
            QuestionKind.ANOMALY,
            QuestionKind.TREND,
        }
    ) or _YOY_RE.search(message) or _FUNNEL_RE.search(message) or _COHORT_RE.search(message)
    if not needs_investigation:
        return []

    add(
        "overall_change",
        f"Overall {aggregation} of {metric.column_name} by {grain} for the comparison window",
        (
            f"SELECT date_trunc('{grain}', \"{time_col.column_name}\")::date AS period, "
            f'{metric_expr} AS "{metric_alias}" FROM {table_sql}{where} '
            f"GROUP BY 1 ORDER BY 1 LIMIT {limit}"
        ),
    )

    dims = breakdown_dimensions(same_table)
    if not dims:
        fallback = _breakdown_column(same_table)
        if fallback is not None:
            dims = [fallback]

    for dimension in dims:
        add(
            "breakdown",
            f"{aggregation} of {metric.column_name} by {grain} and {dimension.column_name}",
            (
                f"SELECT date_trunc('{grain}', \"{time_col.column_name}\")::date AS period, "
                f'"{dimension.column_name}", {metric_expr} AS "{metric_alias}" '
                f"FROM {table_sql}{where} GROUP BY 1, 2 ORDER BY 1, 2 LIMIT {limit}"
            ),
        )

    if _YOY_RE.search(message):
        year = end.year - 1
        yoy_start = date(year, start.month, min(start.day, 28))
        yoy_end = date(end.year, end.month, min(end.day, 28))
        add(
            "year_over_year",
            f"Same calendar window year-over-year {aggregation} of {metric.column_name}",
            (
                f"SELECT date_trunc('{grain}', \"{time_col.column_name}\")::date AS period, "
                f'{metric_expr} AS "{metric_alias}" FROM {table_sql} '
                f' WHERE "{time_col.column_name}" >= DATE \'{yoy_start.isoformat()}\' '
                f' AND "{time_col.column_name}" < DATE \'{yoy_end.isoformat()}\' '
                f"GROUP BY 1 ORDER BY 1 LIMIT {limit}"
            ),
        )

    orders_col = next((c for c in same_table if c.column_name.lower() == "orders"), None)
    conv_col = next((c for c in same_table if c.column_name.lower() == "conversions"), None)
    if _FUNNEL_RE.search(message) and orders_col and conv_col:
        add(
            "funnel",
            "Funnel totals: orders and conversions in the comparison window",
            (
                f'SELECT SUM("{orders_col.column_name}") AS total_orders, '
                f'SUM("{conv_col.column_name}") AS total_conversions, '
                f'SUM("{conv_col.column_name}")::numeric / NULLIF(SUM("{orders_col.column_name}"), 0) '
                f"AS conversion_rate FROM {table_sql}{where} LIMIT 1"
            ),
        )

    cohort_col = next(
        (
            c
            for c in same_table
            if c.column_name.lower() in {"customer_id", "user_id", "account_id"}
        ),
        None,
    )
    if _COHORT_RE.search(message) and cohort_col:
        add(
            "cohort",
            f"Monthly distinct {cohort_col.column_name} counts (cohort proxy)",
            (
                f"SELECT date_trunc('month', \"{time_col.column_name}\")::date AS period, "
                f'COUNT(DISTINCT "{cohort_col.column_name}") AS cohort_size '
                f"FROM {table_sql}{where} GROUP BY 1 ORDER BY 1 LIMIT {limit}"
            ),
        )

    if dims and len(steps) < max_steps:
        top_dim = dims[0]
        drill = next(
            (c for c in dims[1:] if c.column_name.lower() != top_dim.column_name.lower()),
            None,
        )
        if drill is not None:
            target_start = _shift_months(end, -1)
            drill_where = (
                f' WHERE "{time_col.column_name}" >= DATE \'{target_start.isoformat()}\''
                f' AND "{time_col.column_name}" < DATE \'{end.isoformat()}\''
            )
            add(
                "drill",
                f"Drill {metric.column_name} by {drill.column_name} within {top_dim.column_name} "
                f"for the latest period",
                (
                    f"SELECT \"{top_dim.column_name}\", \"{drill.column_name}\", "
                    f'{metric_expr} AS "{metric_alias}" FROM {table_sql}{drill_where} '
                    f"GROUP BY 1, 2 ORDER BY 3 DESC LIMIT {min(limit, 50)}"
                ),
            )

    return steps[:max_steps]


def _fallback_metric(
    message: str, metadata: ResolvedMetadataContext
) -> MetadataColumnCandidate | None:
    columns = [
        c
        for c in _all_columns(metadata)
        if is_numeric_type(c.data_type) and not is_identifier_column(c)
    ]
    if not columns:
        return None
    tokens = set(re.findall(r"[a-z]+", message.lower()))
    for column in columns:
        name = column.column_name.lower()
        if name in tokens or any(token in name for token in tokens if len(token) > 2):
            return column
    if len(metadata.tables) == 1:
        for preferred in ("revenue", "units", "orders", "conversions", "aov"):
            for column in columns:
                if column.column_name.lower() == preferred:
                    return column
        return columns[0]
    return None


def _shift_months(value: date, months: int) -> date:
    index = value.year * 12 + (value.month - 1) + months
    return date(index // 12, index % 12 + 1, 1)
