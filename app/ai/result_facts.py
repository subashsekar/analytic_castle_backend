"""Deterministic, measured facts computed from a SQL result.

Every statement here is arithmetic over the returned rows, so chat answers stay
correct and data-backed even when optional LLM agents are skipped or time out.
Column roles are inferred from values (and generic naming like *_id), never
from business-specific names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.ai.sql_execution.models import SQLExecutionResult
from app.ai.trend_analysis.series import parse_number, parse_period

_MAX_SERIES_LISTED = 12
_MAX_CONTRIBUTORS = 3
_PERIOD_HINTS = ("date", "time", "period", "month", "quarter", "week", "year", "day", "bucket")
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}  # fmt: skip
_MONTH_YEAR_RE = re.compile(
    r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?\s*,?\s*((?:19|20|21)\d{2})\b",
    re.IGNORECASE,
)
_MONTH_ONLY_RE = re.compile(
    r"\b(?:in|during|for|of)\s+("
    + "|".join(m for m in sorted(_MONTHS, key=len, reverse=True) if m != "may")
    + r")\b",
    re.IGNORECASE,
)
_QUARTER_RE = re.compile(r"\bq([1-4])\s*((?:19|20|21)\d{2})\b|\b((?:19|20|21)\d{2})\s*q([1-4])\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b((?:19|20|21)\d{2})\b")
# Averages, rates, and ratios cannot be summed across rows into totals or shares.
_NON_ADDITIVE_RE = re.compile(
    r"(^|_|\b)(avg|average|mean|median|rate|ratio|pct|percent|percentage|share|margin|per)(_|\b|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ResultFacts:
    """Measured statements about a result plus the shape they were derived from."""

    shape: str
    headline: str | None = None
    lines: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.shape == "empty"


def build_result_facts(result: SQLExecutionResult, *, message: str = "") -> ResultFacts:
    columns = list(result.columns)
    rows = [list(row) for row in result.rows]
    if not rows:
        return ResultFacts(
            shape="empty",
            headline="The query ran successfully but returned no rows for this request.",
            notes=["Check the period, filters, or spelling of the values you asked about."],
        )

    period_idx = _period_column(columns, rows)
    numeric_idx = [
        i for i in range(len(columns))
        if i != period_idx and _is_measure(columns[i], rows, i)
    ]  # fmt: skip
    category_idx = [
        i for i in range(len(columns)) if i != period_idx and i not in numeric_idx
    ]
    truncated_note = (
        ["Results were truncated by the row limit; totals cover only the returned rows."]
        if result.truncated
        else []
    )

    if len(rows) == 1 and period_idx is None:
        return _scalar_facts(columns, rows[0], truncated_note)
    if not numeric_idx:
        return ResultFacts(
            shape="table",
            headline=f"The query returned {result.row_count} row(s).",
            lines=[_row_preview(columns, rows)],
            notes=truncated_note,
        )

    metric = _pick_metric(columns, numeric_idx, message)
    if period_idx is not None:
        category = category_idx[0] if category_idx else None
        if category is not None and not _is_additive(columns[metric]):
            return ResultFacts(
                shape="table",
                headline=f"The query returned {result.row_count} row(s).",
                lines=[_row_preview(columns, rows, limit=8)],
                notes=truncated_note,
            )
        return _time_facts(columns, rows, period_idx, metric, category, message, truncated_note)
    if category_idx:
        return _categorical_facts(columns, rows, category_idx[0], metric, truncated_note)
    return ResultFacts(
        shape="table",
        headline=f"The query returned {result.row_count} row(s).",
        lines=[_row_preview(columns, rows)],
        notes=truncated_note,
    )


# ---------------------------------------------------------------- shapes


def _scalar_facts(columns: list[str], row: list[Any], notes: list[str]) -> ResultFacts:
    parts = [f"{_label(col)}: {_fmt_cell(value)}" for col, value in zip(columns[:8], row[:8])]
    headline = parts[0] if len(parts) == 1 else "; ".join(parts)
    return ResultFacts(shape="scalar", headline=headline, notes=notes)


def _categorical_facts(
    columns: list[str],
    rows: list[list[Any]],
    cat: int,
    metric: int,
    notes: list[str],
) -> ResultFacts:
    additive = _is_additive(columns[metric])
    totals: dict[str, float] = {}
    for row in rows:
        value = parse_number(row[metric])
        if value is None:
            continue
        key = _fmt_cell(row[cat])
        if additive:
            totals[key] = totals.get(key, 0.0) + value
        else:
            totals.setdefault(key, value)
    if not totals:
        return ResultFacts(shape="table", lines=[_row_preview(columns, rows)], notes=notes)
    ordered = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    grand = sum(totals.values())
    metric_label = _label(columns[metric])
    cat_label = _label(columns[cat])
    top_name, top_value = ordered[0]
    share = (
        f" ({_pct_share(top_value, grand)} of total)"
        if additive and len(ordered) > 1 and grand
        else ""
    )
    headline = f"Highest {metric_label} by {cat_label}: {top_name} with {_fmt_num(top_value)}{share}."
    summary = f"{len(ordered)} {cat_label} group(s)"
    if additive:
        summary += f"; total {metric_label} {_fmt_num(grand)}"
    lines = [
        summary + ".",
        "Ranking: "
        + ", ".join(f"{name}: {_fmt_num(value)}" for name, value in ordered[:_MAX_SERIES_LISTED])
        + (" ..." if len(ordered) > _MAX_SERIES_LISTED else ""),
    ]
    if len(ordered) > 1:
        low_name, low_value = ordered[-1]
        lines.append(f"Lowest: {low_name} with {_fmt_num(low_value)}.")
    return ResultFacts(shape="categorical", headline=headline, lines=lines, notes=notes)


def _time_facts(
    columns: list[str],
    rows: list[list[Any]],
    period_idx: int,
    metric: int,
    category: int | None,
    message: str,
    notes: list[str],
) -> ResultFacts:
    per_period: dict[datetime, float] = {}
    labels: dict[datetime, str] = {}
    per_cat: dict[datetime, dict[str, float]] = {}
    rollups: dict[datetime, float] = {}
    for row in rows:
        period = parse_period(row[period_idx])
        value = parse_number(row[metric])
        if period is None or value is None:
            continue
        labels.setdefault(period, _period_text(row[period_idx], period))
        if category is None:
            per_period[period] = per_period.get(period, 0.0) + value
            continue
        raw_cat = row[category]
        if raw_cat is None or (isinstance(raw_cat, str) and not str(raw_cat).strip()):
            rollups[period] = value
            continue
        key = _fmt_cell(raw_cat)
        bucket = per_cat.setdefault(period, {})
        bucket[key] = bucket.get(key, 0.0) + value

    if category is not None:
        preferred = _preferred_overall_category(per_cat)
        if rollups:
            per_period = dict(rollups)
        elif preferred is not None:
            per_period = {
                period: cats[preferred]
                for period, cats in per_cat.items()
                if preferred in cats
            }
        else:
            # Dimension breakdown (e.g. region×month): cite cell-level movers only.
            # Summing regions into a period total invents a number that is not a
            # result cell and fails evidence grounding.
            return _category_change_facts(
                columns, per_cat, labels, metric, category, message, notes
            )
    else:
        preferred = None

    if not per_period:
        return ResultFacts(shape="table", lines=[_row_preview(columns, rows)], notes=notes)

    periods = sorted(per_period)
    metric_label = _label(columns[metric])
    additive = _is_additive(columns[metric])
    lines: list[str] = []
    listed = [f"{labels[p]}: {_fmt_num(per_period[p])}" for p in periods]
    if len(listed) > _MAX_SERIES_LISTED:
        listed = listed[:3] + ["..."] + listed[-3:]
    series_text = ", ".join(listed)
    lines.append(f"{metric_label} by period ({len(periods)} periods): {series_text}.")

    if len(periods) == 1:
        only = periods[0]
        headline = f"{metric_label} for {labels[only]} was {_fmt_num(per_period[only])}."
        notes = notes + ["Only one period is present, so no period-over-period change can be measured."]
        return ResultFacts(shape="time_series", headline=headline, lines=lines, notes=notes)

    target = _target_period(message, periods) or periods[-1]
    index = periods.index(target)
    headline: str
    if index == 0:
        nxt = periods[1]
        headline = (
            f"{metric_label} for {labels[target]} was {_fmt_num(per_period[target])}; "
            f"the next period ({labels[nxt]}) was {_fmt_num(per_period[nxt])}."
        )
        previous = None
    else:
        previous = periods[index - 1]
        current_value = per_period[target]
        previous_value = per_period[previous]
        delta = current_value - previous_value
        headline = (
            f"{metric_label} for {labels[target]} was {_fmt_num(current_value)}, "
            f"{_direction(delta)} {_fmt_num(abs(delta))}"
            + (f" ({_fmt_pct(delta / previous_value * 100)})" if previous_value else "")
            + f" from {labels[previous]} ({_fmt_num(previous_value)})."
        )

    high = max(periods, key=lambda p: per_period[p])
    low = min(periods, key=lambda p: per_period[p])
    first_value, last_value = per_period[periods[0]], per_period[periods[-1]]
    lines.append(
        f"Highest period: {labels[high]} ({_fmt_num(per_period[high])}); "
        f"lowest period: {labels[low]} ({_fmt_num(per_period[low])})."
    )
    overall = last_value - first_value
    lines.append(
        f"From {labels[periods[0]]} to {labels[periods[-1]]}: {_direction(overall)} "
        f"{_fmt_num(abs(overall))}"
        + (f" ({_fmt_pct(overall / first_value * 100)})" if first_value else "")
        + (f"; total across periods {_fmt_num(sum(per_period.values()))}" if additive else "")
        + "."
    )
    if category is not None and previous is not None and (rollups or preferred is not None):
        # Multi-section / rollup query: contributor lines only when we also have
        # per-category leaf rows beyond the chosen overall grain.
        leaf = {
            period: {k: v for k, v in cats.items() if preferred is None or k != preferred}
            for period, cats in per_cat.items()
        }
        if any(leaf.values()):
            lines.extend(
                _contributors(
                    leaf.get(previous, {}),
                    leaf.get(target, {}),
                    category_label=_label(columns[category]),
                    previous_label=labels[previous],
                    target_label=labels[target],
                )
            )
    elif category is not None and previous is not None:
        lines.extend(
            _contributors(
                per_cat.get(previous, {}),
                per_cat.get(target, {}),
                category_label=_label(columns[category]),
                previous_label=labels[previous],
                target_label=labels[target],
            )
        )
    elif category is not None:
        lines.extend(_category_split(per_cat.get(target, {}), columns[category], metric_label))
    return ResultFacts(shape="time_series", headline=headline, lines=lines, notes=notes)


_OVERALL_CATEGORY_NAMES = frozenset(
    {
        "overall",
        "total",
        "all",
        "summary",
        "monthly_comparison",
        "overall_change",
        "period_total",
        "grand_total",
    }
)


def _preferred_overall_category(per_cat: dict[datetime, dict[str, float]]) -> str | None:
    names: set[str] = set()
    for cats in per_cat.values():
        names.update(cats)
    by_key = {name.lower().replace(" ", "_"): name for name in names}
    for candidate in _OVERALL_CATEGORY_NAMES:
        if candidate in by_key:
            return by_key[candidate]
    return None


def _category_change_facts(
    columns: list[str],
    per_cat: dict[datetime, dict[str, float]],
    labels: dict[datetime, str],
    metric: int,
    category: int,
    message: str,
    notes: list[str],
) -> ResultFacts:
    """Period×dimension facts using only cell-level values (no invented totals)."""
    periods = sorted(per_cat)
    metric_label = _label(columns[metric])
    cat_label = _label(columns[category])
    if len(periods) < 2:
        only = periods[0] if periods else None
        if only is None:
            return ResultFacts(shape="table", notes=notes)
        lines = _category_split(per_cat.get(only, {}), columns[category], metric_label)
        return ResultFacts(
            shape="time_series",
            headline=f"{metric_label} by {cat_label} for {labels[only]}.",
            lines=lines,
            notes=notes
            + ["Only one period is present, so no period-over-period change can be measured."],
        )

    target = _target_period(message, periods) or periods[-1]
    index = periods.index(target)
    previous = periods[index - 1] if index > 0 else periods[0]
    if previous == target and len(periods) > 1:
        previous = periods[0] if target != periods[0] else periods[1]
    before = per_cat.get(previous, {})
    after = per_cat.get(target, {})
    keys = set(before) | set(after)
    deltas = {key: after.get(key, 0.0) - before.get(key, 0.0) for key in keys}
    if not deltas:
        return ResultFacts(shape="table", notes=notes)
    mover, delta = max(deltas.items(), key=lambda item: abs(item[1]))
    before_value, after_value = before.get(mover, 0.0), after.get(mover, 0.0)
    headline = (
        f"Largest {cat_label} mover {labels[previous]} to {labels[target]}: {mover} "
        f"from {_fmt_num(before_value)} to {_fmt_num(after_value)}, "
        f"{_direction(delta)} {_fmt_num(abs(delta))}"
        + (f" ({_fmt_pct(delta / before_value * 100)})" if before_value else "")
        + "."
    )
    lines = [
        f"{metric_label} by {cat_label} ({labels[previous]} to {labels[target]}): "
        + ", ".join(
            f"{key} {_fmt_num(before.get(key, 0.0))} → {_fmt_num(after.get(key, 0.0))}"
            for key, _ in sorted(deltas.items(), key=lambda item: abs(item[1]), reverse=True)[
                :_MAX_SERIES_LISTED
            ]
        )
        + ".",
    ]
    lines.extend(
        _contributors(
            before,
            after,
            category_label=cat_label,
            previous_label=labels[previous],
            target_label=labels[target],
        )
    )
    return ResultFacts(shape="time_series", headline=headline, lines=lines, notes=notes)


def _contributors(
    before: dict[str, float],
    after: dict[str, float],
    *,
    category_label: str,
    previous_label: str,
    target_label: str,
) -> list[str]:
    keys = set(before) | set(after)
    deltas = {key: after.get(key, 0.0) - before.get(key, 0.0) for key in keys}
    total_delta = sum(deltas.values())
    if not deltas or total_delta == 0:
        return []
    same_direction = sorted(
        (item for item in deltas.items() if (item[1] < 0) == (total_delta < 0) and item[1] != 0),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    offsetting = sorted(
        (item for item in deltas.items() if (item[1] < 0) != (total_delta < 0) and item[1] != 0),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    lines: list[str] = []
    if same_direction:
        parts = []
        for key, delta in same_direction[:_MAX_CONTRIBUTORS]:
            base = before.get(key, 0.0)
            pct = f", {_fmt_pct(delta / base * 100)}" if base else ""
            share = _pct_share(delta, total_delta)
            parts.append(f"{key} {_signed(delta)}{pct} ({share} of the change)")
        lines.append(
            f"Largest contributors by {category_label} ({previous_label} to {target_label}): "
            + "; ".join(parts)
            + "."
        )
    if offsetting:
        key, delta = offsetting[0]
        lines.append(f"Partly offset by {key} ({_signed(delta)}).")
    return lines


def _category_split(values: dict[str, float], column: str, metric_label: str) -> list[str]:
    if not values:
        return []
    ordered = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
    return [
        f"{metric_label} by {_label(column)}: "
        + ", ".join(f"{k} {_fmt_num(v)}" for k, v in ordered[:_MAX_SERIES_LISTED])
        + "."
    ]


# ---------------------------------------------------------------- inference


def _period_column(columns: list[str], rows: list[list[Any]]) -> int | None:
    candidates: list[tuple[bool, int]] = []
    for index, column in enumerate(columns):
        values = [row[index] for row in rows if index < len(row) and row[index] is not None]
        if not values:
            continue
        hinted = any(hint in column.lower() for hint in _PERIOD_HINTS)
        numeric_only = all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
        )
        if numeric_only and not hinted:
            continue
        if any(isinstance(v, str) and parse_number(v) is not None and not hinted for v in values):
            continue
        if all(parse_period(v) is not None for v in values):
            candidates.append((not hinted, index))
    return min(candidates)[1] if candidates else None


def _is_measure(column: str, rows: list[list[Any]], index: int) -> bool:
    name = column.strip().lower()
    if name == "id" or name.endswith("_id") or name.endswith(" id"):
        return False
    values = [row[index] for row in rows if index < len(row) and row[index] is not None]
    return bool(values) and all(parse_number(v) is not None for v in values)


def _is_additive(column: str) -> bool:
    return _NON_ADDITIVE_RE.search(column.strip()) is None


def _pick_metric(columns: list[str], numeric_idx: list[int], message: str) -> int:
    words = set(re.findall(r"[a-z0-9]+", (message or "").lower()))
    scored = [
        (len(set(re.findall(r"[a-z0-9]+", columns[index].lower())) & words), -position, index)
        for position, index in enumerate(numeric_idx)
    ]
    overlap, _, index = max(scored)
    return index if overlap > 0 else numeric_idx[0]


def _target_period(message: str, periods: list[datetime]) -> datetime | None:
    text = message or ""
    quarter = _QUARTER_RE.search(text)
    if quarter:
        q = int(quarter.group(1) or quarter.group(4))
        year = int(quarter.group(2) or quarter.group(3))
        months = {3 * q - 2, 3 * q - 1, 3 * q}
        matches = [p for p in periods if p.year == year and p.month in months]
        if matches:
            return matches[-1]
    month_year = _MONTH_YEAR_RE.search(text)
    if month_year:
        month = _MONTHS[month_year.group(1).lower()]
        year = int(month_year.group(2))
        for period in periods:
            if period.year == year and period.month == month:
                return period
    month_only = _MONTH_ONLY_RE.search(text)
    if month_only:
        month = _MONTHS[month_only.group(1).lower()]
        matches = [p for p in periods if p.month == month]
        if matches:
            return matches[-1]
    years = [int(y) for y in _YEAR_RE.findall(text)]
    if len(years) == 1:
        matches = [p for p in periods if p.year == years[0]]
        if matches and len({p.year for p in periods}) > 1:
            return matches[-1]
    return None


# ---------------------------------------------------------------- formatting


def _label(column: str) -> str:
    return column.replace("_", " ").strip() or column


def _period_text(raw: Any, parsed: datetime) -> str:
    if isinstance(raw, str) and not re.match(r"^\d{4}-\d{2}-\d{2}", raw.strip()):
        return raw.strip()
    if parsed.day == 1 and parsed.hour == 0 and parsed.minute == 0:
        if parsed.month == 1 and isinstance(raw, (int, float)):
            return str(parsed.year)
        return parsed.strftime("%b %Y")
    return parsed.date().isoformat()


def _fmt_num(value: float) -> str:
    if abs(value - round(value)) < 1e-9 and abs(value) < 1e15:
        return f"{int(round(value)):,}"
    return f"{value:,.2f}"


def _fmt_cell(value: Any) -> str:
    if value is None:
        return "(none)"
    number = parse_number(value) if not isinstance(value, str) else None
    if number is not None:
        return _fmt_num(number)
    return str(value)


def _fmt_pct(value: float) -> str:
    return f"{value:+.1f}%"


def _pct_share(part: float, whole: float) -> str:
    if not whole:
        return "n/a"
    return f"{part / whole * 100:.0f}%"


def _signed(value: float) -> str:
    return ("+" if value > 0 else "-") + _fmt_num(abs(value))


def _direction(delta: float) -> str:
    if delta > 0:
        return "up"
    if delta < 0:
        return "down"
    return "unchanged by"


def _row_preview(columns: list[str], rows: list[list[Any]], limit: int = 5) -> str:
    head = ", ".join(columns[:8])
    body = " | ".join(
        ", ".join(_fmt_cell(cell) for cell in row[:8]) for row in rows[:limit]
    )
    more = " | ..." if len(rows) > limit else ""
    return f"Columns: {head}. Rows: {body}{more}"
