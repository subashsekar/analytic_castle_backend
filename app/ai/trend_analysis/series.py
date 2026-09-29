"""Deterministic time-series shaping for the trend analysis agent.

The LLM narrates the trend; every number it narrates is computed here so that
direction, growth rates, and significance are reproducible and auditable.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

from app.ai.trend_analysis.models import PeriodComparison, TrendDirection, TrendSeries

# Calibration knobs: real series are noisy, these bands decide "moved" vs "flat".
SIGNIFICANT_CHANGE_PERCENT = 20.0
STABLE_BAND_PERCENT = 5.0
MIN_TREND_POINTS = 2
MAX_SERIES_POINTS = 500
MAX_PROMPT_COMPARISONS = 40

_PERIOD_NAME_HINTS = (
    "date",
    "time",
    "period",
    "month",
    "quarter",
    "week",
    "year",
    "day",
    "bucket",
)
_YEAR_RE = re.compile(r"^(19|20|21)\d{2}$")
_QUARTER_RE = re.compile(r"^(\d{4})[-/\s]?Q([1-4])$", re.IGNORECASE)
_WEEK_RE = re.compile(r"^(\d{4})[-/\s]?W(\d{1,2})$", re.IGNORECASE)
_EXTRA_FORMATS = ("%Y-%m", "%Y/%m", "%b %Y", "%B %Y", "%d-%m-%Y", "%m/%d/%Y")


def _drop_tz(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    # date_trunc periods are calendar labels (month/week start), not absolute
    # instants. Converting to UTC shifts the calendar day near timezone edges
    # (e.g. 2025-02-01T00:00+05:30 → 2025-01-31) and mislabels Feb/Mar.
    return value.replace(tzinfo=None)


def parse_period(value: Any) -> datetime | None:
    """Parse a cell into a comparable period, or None when it is not time-like."""
    if isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return _drop_tz(value)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, int) or (isinstance(value, float) and float(value).is_integer()):
        year = int(value)
        return datetime(year, 1, 1) if 1900 <= year <= 2199 else None
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    quarter = _QUARTER_RE.match(text)
    if quarter:
        return datetime(int(quarter.group(1)), 3 * int(quarter.group(2)) - 2, 1)
    week = _WEEK_RE.match(text)
    if week:
        try:
            return datetime.combine(date.fromisocalendar(int(week.group(1)), int(week.group(2)), 1), time())
        except ValueError:
            return None
    if _YEAR_RE.match(text):
        return datetime(int(text), 1, 1)
    try:
        return _drop_tz(datetime.fromisoformat(text))
    except ValueError:
        pass
    for fmt in _EXTRA_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def parse_number(value: Any) -> float | None:
    """Parse a cell into a float measure, or None when it is not numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        try:
            return float(value)
        except (ValueError, OverflowError):
            return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if number == number and abs(number) != float("inf") else None
    if isinstance(value, str):
        try:
            return float(value.strip().replace(",", ""))
        except ValueError:
            return None
    return None


def period_label(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def column_values(rows: list[list[Any]], index: int) -> list[Any]:
    return [row[index] for row in rows if index < len(row) and row[index] is not None]


def resolve_named_column(columns: list[str], name: str | None) -> int | None:
    if not name:
        return None
    lowered = name.strip().lower()
    for index, column in enumerate(columns):
        if column.strip().lower() == lowered:
            return index
    return None


def select_period_column(columns: list[str], rows: list[list[Any]]) -> int | None:
    candidates: list[tuple[bool, int]] = []
    for index, column in enumerate(columns):
        values = column_values(rows, index)
        if not values or any(parse_period(value) is None for value in values):
            continue
        hinted = any(hint in column.lower() for hint in _PERIOD_NAME_HINTS)
        candidates.append((not hinted, index))
    return min(candidates)[1] if candidates else None


def _select_value_column(columns: list[str], rows: list[list[Any]], period_index: int) -> int | None:
    for index, _column in enumerate(columns):
        if index == period_index:
            continue
        values = column_values(rows, index)
        if values and all(parse_number(value) is not None for value in values):
            return index
    return None


def _overall_direction(
    comparisons: list[PeriodComparison],
    *,
    total_change: float,
    growth_rate_percent: float | None,
    direction_changes: int,
    significance_threshold_percent: float,
    stable_band_percent: float,
) -> TrendDirection:
    moves = [item for item in comparisons if item.absolute_change != 0]
    if not moves:
        return TrendDirection.STABLE
    swing = growth_rate_percent is None or abs(growth_rate_percent) < significance_threshold_percent
    if direction_changes >= 2 and direction_changes * 2 >= len(moves) and swing:
        return TrendDirection.VOLATILE
    if growth_rate_percent is not None and abs(growth_rate_percent) < stable_band_percent:
        return TrendDirection.STABLE
    if total_change > 0:
        return TrendDirection.INCREASING
    if total_change < 0:
        return TrendDirection.DECREASING
    return TrendDirection.STABLE


def build_trend_series(
    columns: list[str],
    rows: list[list[Any]],
    *,
    period_column: str | None = None,
    value_column: str | None = None,
    significance_threshold_percent: float = SIGNIFICANT_CHANGE_PERCENT,
    stable_band_percent: float = STABLE_BAND_PERCENT,
) -> TrendSeries:
    """Build an ordered series with period-over-period metrics from query results.

    Always returns a series: when the data is not usable the direction is
    INSUFFICIENT_DATA and `notes` explains why.
    """
    notes: list[str] = []
    if not columns or not rows:
        return TrendSeries(notes=["Query returned no rows, so no trend could be computed."])

    period_index = resolve_named_column(columns, period_column)
    if period_column and period_index is None:
        notes.append(f"Requested period column '{period_column}' is not present in the results.")
    if period_index is None:
        period_index = select_period_column(columns, rows)
    if period_index is None:
        notes.append("No date, time, or period column was found, so the results are not a time series.")
        return TrendSeries(notes=notes)

    value_index = resolve_named_column(columns, value_column)
    if value_column and value_index is None:
        notes.append(f"Requested value column '{value_column}' is not present in the results.")
    if value_index is None or value_index == period_index:
        value_index = _select_value_column(columns, rows, period_index)
    if value_index is None:
        notes.append("No numeric measure column was found alongside the period column.")
        return TrendSeries(period_column=columns[period_index], notes=notes)

    names = {"period_column": columns[period_index], "value_column": columns[value_index]}

    # ponytail: duplicate periods are summed; pass value_column explicitly if the
    # measure is an average or ratio where summing is wrong.
    totals: dict[datetime, float] = {}
    labels: dict[datetime, str] = {}
    skipped = 0
    for row in rows:
        if len(row) <= max(period_index, value_index):
            skipped += 1
            continue
        moment = parse_period(row[period_index])
        number = parse_number(row[value_index])
        if moment is None or number is None:
            skipped += 1
            continue
        if moment in totals:
            totals[moment] += number
        else:
            totals[moment] = number
            labels[moment] = period_label(row[period_index])

    if skipped:
        notes.append(f"{skipped} row(s) were skipped because the period or value was missing or unparseable.")
    duplicates = len(rows) - skipped - len(totals)
    if duplicates > 0:
        notes.append(f"{duplicates} row(s) shared an existing period; values were summed per period.")

    reordered = list(totals) != sorted(totals)
    if reordered:
        notes.append("Rows were not in chronological order and were sorted before analysis.")

    ordered = sorted(totals)
    if len(ordered) > MAX_SERIES_POINTS:
        notes.append(f"Series truncated to the most recent {MAX_SERIES_POINTS} periods.")
        ordered = ordered[-MAX_SERIES_POINTS:]

    if len(ordered) < MIN_TREND_POINTS:
        notes.append(
            f"Only {len(ordered)} usable period(s) were found; at least {MIN_TREND_POINTS} are required for a trend."
        )
        return TrendSeries(
            **names,
            point_count=len(ordered),
            skipped_row_count=skipped,
            reordered=reordered,
            notes=notes,
        )

    comparisons: list[PeriodComparison] = []
    direction_changes = 0
    last_sign = 0
    for previous, current in zip(ordered, ordered[1:]):
        previous_value = totals[previous]
        value = totals[current]
        absolute_change = value - previous_value
        percent_change = (
            round(absolute_change / abs(previous_value) * 100, 4) if previous_value else None
        )
        if absolute_change > 0:
            direction = TrendDirection.INCREASING
            sign = 1
        elif absolute_change < 0:
            direction = TrendDirection.DECREASING
            sign = -1
        else:
            direction = TrendDirection.STABLE
            sign = 0
        if sign and last_sign and sign != last_sign:
            direction_changes += 1
        if sign:
            last_sign = sign
        significant = (
            abs(percent_change) >= significance_threshold_percent
            if percent_change is not None
            else absolute_change != 0
        )
        comparisons.append(
            PeriodComparison(
                previous_period=labels[previous],
                period=labels[current],
                previous_value=round(previous_value, 6),
                value=round(value, 6),
                absolute_change=round(absolute_change, 6),
                percent_change=percent_change,
                direction=direction,
                significant=significant,
            )
        )

    first_value = totals[ordered[0]]
    last_value = totals[ordered[-1]]
    total_change = last_value - first_value
    growth_rate_percent = round(total_change / abs(first_value) * 100, 4) if first_value else None
    stepwise = [item.percent_change for item in comparisons if item.percent_change is not None]
    average_period_change_percent = round(sum(stepwise) / len(stepwise), 4) if stepwise else None
    values = [totals[moment] for moment in ordered]

    return TrendSeries(
        **names,
        point_count=len(ordered),
        skipped_row_count=skipped,
        reordered=reordered,
        first_period=labels[ordered[0]],
        last_period=labels[ordered[-1]],
        first_value=round(first_value, 6),
        last_value=round(last_value, 6),
        minimum_value=round(min(values), 6),
        maximum_value=round(max(values), 6),
        direction=_overall_direction(
            comparisons,
            total_change=total_change,
            growth_rate_percent=growth_rate_percent,
            direction_changes=direction_changes,
            significance_threshold_percent=significance_threshold_percent,
            stable_band_percent=stable_band_percent,
        ),
        total_change=round(total_change, 6),
        growth_rate_percent=growth_rate_percent,
        average_period_change_percent=average_period_change_percent,
        direction_changes=direction_changes,
        comparisons=comparisons,
        significant_changes=[item for item in comparisons if item.significant],
        notes=notes,
    )


def series_prompt_payload(series: TrendSeries, *, max_comparisons: int = MAX_PROMPT_COMPARISONS) -> str:
    """Render the computed series as JSON evidence for the prompt, bounded in size."""
    data = series.model_dump(mode="json")
    comparisons = data.get("comparisons") or []
    if len(comparisons) > max_comparisons:
        head = max_comparisons // 2
        data["comparisons"] = comparisons[:head] + comparisons[head - max_comparisons :]
        data["comparisons_truncated"] = True
    data["significant_changes"] = (data.get("significant_changes") or [])[:max_comparisons]
    return json.dumps(data, indent=2, default=str)
