"""Rank dimension segments by measured contribution to a period-over-period change."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from app.ai.investigation.models import DimensionContribution
from app.ai.trend_analysis.series import parse_number, parse_period


def rank_contributors(
    *,
    columns: list[str],
    rows: list[list[Any]],
    dimension_column: str,
    measure_hints: tuple[str, ...] = ("revenue", "total", "value", "amount", "units"),
) -> list[DimensionContribution]:
    """Compare the last two time periods in a period × dimension result set."""
    if not columns or len(rows) < 2:
        return []
    period_idx = _period_index(columns)
    dim_idx = _column_index(columns, dimension_column)
    if period_idx is None or dim_idx is None:
        return []
    measure_idx = _measure_index(columns, measure_hints, skip={period_idx, dim_idx})
    if measure_idx is None:
        return []

    periods = sorted(
        {key for key in (_period_key(row[period_idx]) for row in rows) if key is not None}
    )
    if len(periods) < 2:
        return []
    current, prior = periods[-1], periods[-2]

    def bucket(period: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for row in rows:
            if _period_key(row[period_idx]) != period:
                continue
            label = str(row[dim_idx])
            value = parse_number(row[measure_idx])
            if value is None:
                continue
            out[label] = out.get(label, 0.0) + value
        return out

    current_map = bucket(current)
    prior_map = bucket(prior)
    total_delta = sum(current_map.values()) - sum(prior_map.values())
    ranked: list[DimensionContribution] = []
    for segment in sorted(set(current_map) | set(prior_map)):
        cur = current_map.get(segment, 0.0)
        prev = prior_map.get(segment, 0.0)
        delta = cur - prev
        share = (delta / total_delta * 100.0) if total_delta else None
        ranked.append(
            DimensionContribution(
                dimension=dimension_column,
                segment=segment,
                current_value=cur,
                prior_value=prev,
                delta=delta,
                share_of_change=share,
            )
        )
    ranked.sort(key=lambda item: abs(item.delta), reverse=True)
    return ranked


def format_contributions(items: list[DimensionContribution], *, limit: int = 5) -> str:
    if not items:
        return ""
    lines = [
        "Measured dimension contributions (correlation with the overall change, not proven causation):"
    ]
    for item in items[:limit]:
        share = (
            f", {item.share_of_change:.1f}% of total change"
            if item.share_of_change is not None
            else ""
        )
        lines.append(
            f"- {item.dimension}={item.segment}: {item.prior_value:.2f} -> "
            f"{item.current_value:.2f} (delta {item.delta:+.2f}{share})"
        )
    return "\n".join(lines)


def _period_index(columns: list[str]) -> int | None:
    for index, name in enumerate(columns):
        lowered = name.lower()
        if lowered in {"period", "month", "week", "day"} or "period" in lowered:
            return index
    for index, name in enumerate(columns):
        if any(token in name.lower() for token in ("date", "time", "month")):
            return index
    return None


def _column_index(columns: list[str], name: str) -> int | None:
    wanted = name.lower()
    for index, column in enumerate(columns):
        if column.lower() == wanted:
            return index
    return None


def _measure_index(
    columns: list[str], hints: tuple[str, ...], *, skip: set[int]
) -> int | None:
    for index, column in enumerate(columns):
        if index in skip:
            continue
        lowered = column.lower()
        if any(hint in lowered for hint in hints):
            return index
    for index, column in enumerate(columns):
        if index not in skip:
            return index
    return None


def _period_key(value: Any) -> str | None:
    if isinstance(value, (date, datetime)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    period = parse_period(value)
    return period.date().isoformat() if period else None
