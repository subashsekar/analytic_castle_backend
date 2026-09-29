"""Deterministic anomaly detection over query results.

Detection and severity are computed here so the LLM only has to describe
findings it cannot invent. Period handling reuses the Phase 8.2 series helpers.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from typing import Any

from app.ai.anomaly_detection.models import (
    Anomaly,
    AnomalyScan,
    AnomalySeverity,
    AnomalyType,
    ColumnThreshold,
    severity_rank,
)
from app.ai.trend_analysis.models import TrendDirection
from app.ai.trend_analysis.series import (
    SIGNIFICANT_CHANGE_PERCENT,
    build_trend_series,
    column_values,
    parse_number,
    period_label,
    resolve_named_column,
    select_period_column,
)

# Calibration knobs: robust-score cut-offs and the smallest sample worth scoring.
MIN_STATISTICAL_POINTS = 5
OUTLIER_SCORE = 3.5
MEDIUM_OUTLIER_SCORE = 4.5
HIGH_OUTLIER_SCORE = 6.0
MEDIUM_CHANGE_PERCENT = 50.0
HIGH_CHANGE_PERCENT = 100.0
MEDIUM_BREACH_PERCENT = 20.0
HIGH_BREACH_PERCENT = 50.0
MAX_ANOMALIES = 50
MAX_SCAN_ROWS = 5_000
MAX_SERIES_COLUMNS = 5
# 0.6745 rescales the median absolute deviation onto the standard-deviation scale.
_MAD_SCALE = 0.6745


def _severity_from(value: float, medium: float, high: float) -> AnomalySeverity:
    if value >= high:
        return AnomalySeverity.HIGH
    if value >= medium:
        return AnomalySeverity.MEDIUM
    return AnomalySeverity.LOW


def _numeric_columns(
    columns: list[str],
    rows: list[list[Any]],
    period_index: int | None,
) -> dict[int, list[tuple[int, float]]]:
    found: dict[int, list[tuple[int, float]]] = {}
    for index in range(len(columns)):
        if index == period_index:
            continue
        values = column_values(rows, index)
        if not values or any(parse_number(value) is None for value in values):
            continue
        points: list[tuple[int, float]] = []
        for row_index, row in enumerate(rows):
            if index >= len(row):
                continue
            number = parse_number(row[index])
            if number is not None:
                points.append((row_index, number))
        if points:
            found[index] = points
    return found


def _detect_outliers(
    column: str,
    points: list[tuple[int, float]],
    labels: dict[int, str],
) -> tuple[list[Anomaly], dict[str, float]]:
    values = [value for _index, value in points]
    median = statistics.median(values)
    mean = statistics.fmean(values)
    deviations = [abs(value - median) for value in values]
    mad = statistics.median(deviations)
    stats = {
        "count": float(len(values)),
        "mean": round(mean, 6),
        "median": round(median, 6),
        "minimum": round(min(values), 6),
        "maximum": round(max(values), 6),
        "median_absolute_deviation": round(mad, 6),
        "standard_deviation": round(statistics.pstdev(values), 6),
    }

    if mad > 0:
        method = "modified_z_score"
        center, spread = median, mad / _MAD_SCALE
    else:
        spread = statistics.pstdev(values)
        if spread == 0:
            return [], stats
        method = "z_score"
        center = mean

    expected_low = round(center - OUTLIER_SCORE * spread, 6)
    expected_high = round(center + OUTLIER_SCORE * spread, 6)
    anomalies: list[Anomaly] = []
    for row_index, value in points:
        score = (value - center) / spread
        if abs(score) < OUTLIER_SCORE:
            continue
        period = labels.get(row_index)
        where = f" at {period}" if period else f" in row {row_index}"
        anomalies.append(
            Anomaly(
                column=column,
                anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                severity=_severity_from(abs(score), MEDIUM_OUTLIER_SCORE, HIGH_OUTLIER_SCORE),
                method=method,
                value=round(value, 6),
                row_index=row_index,
                period=period,
                expected_low=expected_low,
                expected_high=expected_high,
                score=round(score, 4),
                evidence=(
                    f"{column}={round(value, 6)}{where} scores {round(score, 2)} on the {method}, "
                    f"outside the expected range [{expected_low}, {expected_high}] "
                    f"(median {stats['median']}, n={len(values)})."
                ),
            )
        )
    return anomalies, stats


def _detect_threshold_breaches(
    column: str,
    points: list[tuple[int, float]],
    labels: dict[int, str],
    threshold: ColumnThreshold,
) -> list[Anomaly]:
    anomalies: list[Anomaly] = []
    for row_index, value in points:
        if threshold.maximum is not None and value > threshold.maximum:
            limit, side = threshold.maximum, "above the maximum"
        elif threshold.minimum is not None and value < threshold.minimum:
            limit, side = threshold.minimum, "below the minimum"
        else:
            continue
        scale = abs(limit) or abs(value) or 1.0
        breach_percent = abs(value - limit) / scale * 100
        period = labels.get(row_index)
        where = f" at {period}" if period else f" in row {row_index}"
        anomalies.append(
            Anomaly(
                column=column,
                anomaly_type=AnomalyType.THRESHOLD_BREACH,
                severity=_severity_from(breach_percent, MEDIUM_BREACH_PERCENT, HIGH_BREACH_PERCENT),
                method="threshold",
                value=round(value, 6),
                row_index=row_index,
                period=period,
                expected_low=threshold.minimum,
                expected_high=threshold.maximum,
                score=round(breach_percent, 4),
                evidence=(
                    f"{column}={round(value, 6)}{where} is {round(breach_percent, 2)}% {side} "
                    f"of {limit}."
                ),
            )
        )
    return anomalies


def _detect_unexpected_changes(
    columns: list[str],
    rows: list[list[Any]],
    column: str,
    period_column: str,
    change_threshold_percent: float,
) -> list[Anomaly]:
    series = build_trend_series(
        columns,
        rows,
        period_column=period_column,
        value_column=column,
        significance_threshold_percent=change_threshold_percent,
    )
    if series.direction is TrendDirection.INSUFFICIENT_DATA:
        return []

    anomalies: list[Anomaly] = []
    for change in series.significant_changes:
        magnitude = abs(change.percent_change) if change.percent_change is not None else HIGH_CHANGE_PERCENT
        movement = "rose" if change.absolute_change > 0 else "fell"
        percent_text = (
            f"{round(change.percent_change, 2)}%" if change.percent_change is not None else "from a zero baseline"
        )
        anomalies.append(
            Anomaly(
                column=column,
                anomaly_type=AnomalyType.UNEXPECTED_CHANGE,
                severity=_severity_from(magnitude, MEDIUM_CHANGE_PERCENT, HIGH_CHANGE_PERCENT),
                method="period_change",
                value=change.value,
                period=change.period,
                previous_value=change.previous_value,
                percent_change=change.percent_change,
                score=round(magnitude, 4),
                evidence=(
                    f"{column} {movement} {percent_text} from {change.previous_value} at "
                    f"{change.previous_period} to {change.value} at {change.period}, exceeding the "
                    f"{change_threshold_percent}% period-over-period threshold."
                ),
            )
        )
    return anomalies


def scan_for_anomalies(
    columns: list[str],
    rows: list[list[Any]],
    *,
    thresholds: Sequence[ColumnThreshold] | None = None,
    period_column: str | None = None,
    change_threshold_percent: float = SIGNIFICANT_CHANGE_PERCENT,
) -> AnomalyScan:
    """Scan query results for outliers, threshold breaches, and unexpected changes.

    Always returns a scan: when nothing could be scanned `analyzed` is False and
    `notes` explains why.
    """
    notes: list[str] = []
    if not columns or not rows:
        return AnomalyScan(notes=["Query returned no rows, so there was nothing to scan for anomalies."])

    scanned = rows
    if len(scanned) > MAX_SCAN_ROWS:
        scanned = scanned[:MAX_SCAN_ROWS]
        notes.append(f"Only the first {MAX_SCAN_ROWS} rows were scanned.")

    period_index = resolve_named_column(columns, period_column)
    if period_column and period_index is None:
        notes.append(f"Requested period column '{period_column}' is not present in the results.")
    if period_index is None:
        period_index = select_period_column(columns, scanned)
    labels: dict[int, str] = {}
    if period_index is not None:
        for row_index, row in enumerate(scanned):
            if period_index < len(row) and row[period_index] is not None:
                labels[row_index] = period_label(row[period_index])

    numeric = _numeric_columns(columns, scanned, period_index)
    if not numeric:
        notes.append("No numeric column was found, so the results are not suitable for anomaly detection.")
        return AnomalyScan(
            scanned_row_count=len(scanned),
            period_column=columns[period_index] if period_index is not None else None,
            notes=notes,
        )

    by_column = {threshold.column.strip().lower(): threshold for threshold in (thresholds or [])}
    unknown = sorted(name for name in by_column if resolve_named_column(columns, name) is None)
    if unknown:
        notes.append(f"Thresholds ignored for columns not present in the results: {', '.join(unknown)}.")

    anomalies: list[Anomaly] = []
    statistics_by_column: dict[str, dict[str, float]] = {}
    for index, points in numeric.items():
        column = columns[index]
        threshold = by_column.get(column.strip().lower())
        if threshold is not None:
            anomalies.extend(_detect_threshold_breaches(column, points, labels, threshold))
        if len(points) < MIN_STATISTICAL_POINTS:
            notes.append(
                f"Column '{column}' has only {len(points)} numeric value(s); "
                f"at least {MIN_STATISTICAL_POINTS} are required for statistical detection."
            )
            continue
        found, stats = _detect_outliers(column, points, labels)
        statistics_by_column[column] = stats
        anomalies.extend(found)

    if period_index is not None:
        for index in list(numeric)[:MAX_SERIES_COLUMNS]:
            anomalies.extend(
                _detect_unexpected_changes(
                    columns,
                    scanned,
                    columns[index],
                    columns[period_index],
                    change_threshold_percent,
                )
            )
    else:
        notes.append("No period column was found, so time-series anomaly detection was skipped.")

    anomalies.sort(key=lambda item: (-severity_rank(item.severity), -abs(item.score or 0.0)))
    if len(anomalies) > MAX_ANOMALIES:
        notes.append(f"{len(anomalies)} anomalies found; only the {MAX_ANOMALIES} most severe are reported.")
        anomalies = anomalies[:MAX_ANOMALIES]

    return AnomalyScan(
        analyzed=True,
        numeric_columns=[columns[index] for index in numeric],
        scanned_row_count=len(scanned),
        period_column=columns[period_index] if period_index is not None else None,
        column_statistics=statistics_by_column,
        anomalies=anomalies,
        notes=notes,
    )


def scan_prompt_payload(scan: AnomalyScan) -> str:
    """Render the scan as JSON evidence for the prompt."""
    return json.dumps(scan.model_dump(mode="json"), indent=2, default=str)
