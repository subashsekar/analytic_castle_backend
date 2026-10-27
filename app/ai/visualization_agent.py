from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from app.ai.result_facts import ResultFacts
from app.ai.sql_execution.models import SQLExecutionResult
from app.schemas.charts import (
    BarChartSpec,
    LineChartSpec,
    PieChartSpec,
    ScatterChartSpec,
    TableChartSpec,
)
from app.ai.trend_analysis.series import (
    column_values,
    parse_number,
    parse_period,
    select_period_column,
)

_MAX_CHART_POINTS = 200
_MAX_TABLE_ROWS = 50
_MAX_TABLE_COLS = 10
_MAX_INPUT_ROWS_FOR_SPEC = 2000

_COMPOSITION_RE = re.compile(
    r"\b(share|shares|percentage|percent|mix|distribution|portion|composition|"
    r"contribution|% of|share of)\b",
    re.IGNORECASE,
)

_NON_ADDITIVE_RE = re.compile(
    r"\b(avg|average|mean|median|ratio|rate|per)\b",
    re.IGNORECASE,
)


def _non_additive_for_message(message: str) -> bool:
    """
    Used to decide whether it is safe to SUM duplicate categories/periods.

    ponytail: this is heuristic; if you later add metric-level additivity metadata,
    replace this with a proper check.
    """

    return bool(_NON_ADDITIVE_RE.search(message or ""))


def _hashable_category_key(value: Any) -> Any:
    # Use the raw cell value when possible; fall back to its string form for
    # unhashable objects, but keep labels based on the original value.
    try:
        hash(value)
        return value
    except Exception:  # noqa: BLE001
        return str(value)


def _validate_chart_spec(spec: dict[str, Any]) -> dict[str, Any]:
    spec_type = spec.get("type")
    try:
        model = {
            "line": LineChartSpec,
            "bar": BarChartSpec,
            "pie": PieChartSpec,
            "scatter": ScatterChartSpec,
            "table": TableChartSpec,
        }.get(spec_type)
        if model is None:
            # Unknown chart type: never leak an invalid payload downstream.
            # Fall back to a schema-valid, evidence-preserving table.
            return {
                "type": "table",
                "title": None,
                "columns": [],
                "rows": [],
                "truncated": False,
                "empty": True,
            }
        return model.model_validate(spec).model_dump(mode="json")
    except Exception:  # noqa: BLE001
        # Validation should never break the chart pipeline; fall back to a
        # conservative table.
        return {
            "type": "table",
            "title": None,
            "columns": [],
            "rows": [],
            "truncated": False,
            "empty": True,
        }


def _stringify_cell(value: Any) -> Any:
    # Keep primitives as-is (for JSON), but normalize whitespace-y strings.
    if isinstance(value, str):
        return " ".join(value.strip().split())
    return value


def _coerce_y(value: Any) -> float | None:
    # parse_number already handles strings with thousands separators.
    return parse_number(value)


def _numeric_column_indices(columns: list[str], rows: list[list[Any]]) -> list[int]:
    numeric: list[int] = []
    for idx, _col in enumerate(columns):
        values = column_values(rows, idx)
        if not values:
            continue
        parsed = [parse_number(v) for v in values]
        non_null = [v for v in parsed if v is not None]
        # Require at least one usable numeric and a minimal "enough" coverage.
        if non_null and len(non_null) / max(1, len(values)) >= 0.6:
            numeric.append(idx)
    return numeric


def _pick_period_and_value(
    columns: list[str], rows: list[list[Any]]
) -> tuple[int | None, int | None]:
    period_idx = select_period_column(columns, rows)
    if period_idx is None:
        return None, None
    numeric = [
        idx
        for idx in _numeric_column_indices(columns, rows)
        if idx != period_idx
    ]
    if not numeric:
        return period_idx, None
    # Prefer the first numeric column that has parseable numbers on rows with period.
    for idx in numeric:
        usable = 0
        total = 0
        for row in rows:
            if len(row) <= max(period_idx, idx):
                continue
            if parse_period(row[period_idx]) is None:
                continue
            total += 1
            if _coerce_y(row[idx]) is not None:
                usable += 1
        if total and usable / total >= 0.6:
            return period_idx, idx
    return period_idx, numeric[0]


def _pick_category_and_value(
    columns: list[str], rows: list[list[Any]], *, prefer_two_cols: bool
) -> tuple[int | None, int | None]:
    numeric = _numeric_column_indices(columns, rows)
    if not numeric:
        return None, None
    value_idx = numeric[0]
    if prefer_two_cols and len(columns) == 2:
        category_idx = 1 - value_idx
        if category_idx != value_idx:
            return category_idx, value_idx
    # Choose the first non-numeric column as category.
    for idx, _col in enumerate(columns):
        if idx == value_idx:
            continue
        values = column_values(rows, idx)
        if not values:
            continue
        # Category should not be numeric-mostly.
        parsed = [parse_number(v) for v in values]
        non_null = [v for v in parsed if v is not None]
        if non_null and len(non_null) / max(1, len(values)) < 0.4:
            return idx, value_idx
        if non_null and len(non_null) == 0:
            return idx, value_idx
    # Fallback: pick the first column that isn't the value.
    for idx in range(len(columns)):
        if idx != value_idx:
            return idx, value_idx
    return None, value_idx


def _truncate_rows(rows: list[list[Any]], max_rows: int) -> tuple[list[list[Any]], bool]:
    if len(rows) <= max_rows:
        return rows, False
    return rows[:max_rows], True


def _truncate_points(points: list[tuple[Any, float]], max_points: int) -> tuple[list[tuple[Any, float]], bool]:
    if len(points) <= max_points:
        return points, False
    # Prefer "most recent"/tail points when chronological; caller decides sort.
    return points[-max_points:], True


def build_chart_spec(
    *,
    message: str,
    query_result: SQLExecutionResult,
    facts: ResultFacts | None,
) -> dict[str, Any] | None:
    """
    Deterministic chart-spec builder.

    ponytail: column-role heuristics are intentionally simple; upgrade path is
    to reuse ResultFacts' internal role inference and add richer multi-series
    mapping when the frontend requires it.
    """
    # Always return *something* for successful queries, but if we can't form a
    # meaningful visualization, fall back to an evidence-preserving table.
    columns = list(query_result.columns)
    rows = [list(r) for r in (query_result.rows or [])]
    input_truncated = False
    if len(rows) > _MAX_INPUT_ROWS_FOR_SPEC:
        rows = rows[:_MAX_INPUT_ROWS_FOR_SPEC]
        input_truncated = True

    if not columns:
        return _validate_chart_spec(
            {
                "type": "table",
                "title": None,
                "columns": [],
                "rows": [],
                "truncated": False,
                "empty": True,
            }
        )

    if not rows or query_result.row_count == 0:
        return _validate_chart_spec(
            {
            "type": "table",
            "title": None,
            "columns": columns[:_MAX_TABLE_COLS],
            "rows": [],
            "truncated": False,
            "empty": True,
            }
        )

    facts_shape = facts.shape if facts is not None else None

    if facts_shape == "time_series":
        period_idx, value_idx = _pick_period_and_value(columns, rows)
        if period_idx is not None and value_idx is not None:
            # Build sortable period/value entries from actual cells.
            entries: list[tuple[datetime, Any, float]] = []
            for row in rows:
                if len(row) <= max(period_idx, value_idx):
                    continue
                px = row[period_idx]
                py = row[value_idx]
                period = parse_period(px)
                y = _coerce_y(py)
                if period is None or y is None:
                    continue
                x_label = px  # keep the original cell value as chart x-label
                entries.append((period, x_label, y))

            if entries:
                # Safe aggregation: only SUM duplicates when we don't detect
                # obvious non-additive phrasing (avg/ratio/rate/etc).
                allow_sum = not _non_additive_for_message(message)
                entries.sort(key=lambda t: t[0])

                if allow_sum:
                    grouped: dict[datetime, tuple[Any, float]] = {}
                    order: list[datetime] = []
                    for dt, x_label, y in entries:
                        if dt not in grouped:
                            grouped[dt] = (x_label, y)
                            order.append(dt)
                        else:
                            # Aggregate exactly once inside this function.
                            grouped[dt] = (grouped[dt][0], grouped[dt][1] + y)

                    points = [(grouped[dt][0], grouped[dt][1]) for dt in order]
                    points, truncated = _truncate_points(points, _MAX_CHART_POINTS)
                else:
                    points = [(x_label, y) for _dt, x_label, y in entries]
                    points, truncated = _truncate_points(points, _MAX_CHART_POINTS)

                if points:
                    x = [_stringify_cell(v) for v, _y in points]
                    y = [_y for _x, _y in points]
                    spec = {
                        "type": "line",
                        "title": None,
                        "x_axis": columns[period_idx] if period_idx < len(columns) else None,
                        "y_axis": columns[value_idx] if value_idx < len(columns) else None,
                        "x": x,
                        "y": y,
                        "truncated": truncated
                        or input_truncated
                        or bool(query_result.truncated),
                    }
                    return _validate_chart_spec(spec)

    if facts_shape == "categorical":
        prefer_pie = bool(_COMPOSITION_RE.search(message or ""))
        prefer_pie = prefer_pie and len(columns) == 2
        category_idx, value_idx = _pick_category_and_value(
            columns, rows, prefer_two_cols=(len(columns) == 2)
        )
        if category_idx is not None and value_idx is not None:
            # Pair up category/value only when both are usable.
            pairs: list[tuple[Any, float]] = []
            for row in rows:
                if len(row) <= max(category_idx, value_idx):
                    continue
                x_raw = row[category_idx]
                y_raw = row[value_idx]
                y = _coerce_y(y_raw)
                if y is None:
                    continue
                pairs.append((_stringify_cell(x_raw), y))
            if pairs:
                allow_sum = not _non_additive_for_message(message)

                # Pie is usually composition: duplicates should be safely SUMMED
                # into slices when the request implies shares/mix.
                if prefer_pie:
                    aggregated: dict[Any, float] = {}
                    first_label: dict[Any, Any] = {}
                    for label, y in pairs:
                        key = _hashable_category_key(label)
                        if key not in aggregated:
                            first_label[key] = label
                            aggregated[key] = y
                        else:
                            aggregated[key] = aggregated[key] + y

                    sorted_items = sorted(
                        aggregated.items(),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )
                    pie_points: list[tuple[Any, float]] = [
                        (first_label[key], value) for key, value in sorted_items
                    ]
                    pie_points, truncated = _truncate_points(
                        pie_points, _MAX_CHART_POINTS
                    )
                    spec = {
                        "type": "pie",
                        "title": None,
                        "x_axis": columns[category_idx],
                        "y_axis": columns[value_idx],
                        "labels": [lbl for lbl, _v in pie_points],
                        "values": [_v for _lbl, _v in pie_points],
                        "truncated": truncated
                        or input_truncated
                        or bool(query_result.truncated),
                    }
                    return _validate_chart_spec(spec)

                # Bar: SUM duplicates when it looks additive; otherwise keep
                # row-level categories (which may yield repeated x labels).
                if allow_sum:
                    aggregated = {}
                    first_order: list[Any] = []
                    for label, y in pairs:
                        key = _hashable_category_key(label)
                        if key not in aggregated:
                            aggregated[key] = [label, y]
                            first_order.append(key)
                        else:
                            # Aggregate exactly once.
                            aggregated[key][1] = aggregated[key][1] + y

                    bar_points = [(aggregated[key][0], aggregated[key][1]) for key in first_order]
                    # Sort bars by value descending for deterministic readability.
                    bar_points.sort(key=lambda t: t[1], reverse=True)
                    bar_points, truncated = _truncate_points(bar_points, _MAX_CHART_POINTS)
                else:
                    bar_points, truncated = _truncate_points(pairs, _MAX_CHART_POINTS)

                spec = {
                    "type": "bar",
                    "title": None,
                    "x_axis": columns[category_idx],
                    "y_axis": columns[value_idx],
                    "x": [_lbl for _lbl, _v in bar_points],
                    "y": [_v for _lbl, _v in bar_points],
                    "truncated": truncated
                    or input_truncated
                    or bool(query_result.truncated),
                }
                return _validate_chart_spec(spec)

    # For arbitrary "table" shapes, pick scatter if there are 2 numeric columns.
    numeric = _numeric_column_indices(columns, rows)
    if len(numeric) >= 2 and len(columns) <= 6 and (facts_shape in (None, "table", "scalar")):
        x_idx, y_idx = numeric[0], numeric[1]
        points: list[tuple[float, float]] = []
        for row in rows:
            if len(row) <= max(x_idx, y_idx):
                continue
            x_val = _coerce_y(row[x_idx])
            y_val = _coerce_y(row[y_idx])
            if x_val is None or y_val is None:
                continue
            points.append((x_val, y_val))
        if points:
            # Deterministic ordering for stable chart rendering.
            points.sort(key=lambda t: (t[0], t[1]))
            points, truncated = _truncate_points(points, _MAX_CHART_POINTS)
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            spec = {
                "type": "scatter",
                "title": None,
                "x_axis": columns[x_idx],
                "y_axis": columns[y_idx],
                "x": x,
                "y": y,
                "truncated": truncated or input_truncated or bool(query_result.truncated),
            }
            return _validate_chart_spec(spec)

    # Final fallback: table (evidence-preserving; never invents chart data).
    max_cols = min(_MAX_TABLE_COLS, len(columns))
    trimmed_rows, truncated = _truncate_rows(rows, _MAX_TABLE_ROWS)
    return _validate_chart_spec(
        {
        "type": "table",
        "title": None,
        "columns": columns[:max_cols],
        "rows": [r[:max_cols] for r in trimmed_rows],
        "truncated": truncated or input_truncated or bool(query_result.truncated),
        "empty": False,
        }
    )

