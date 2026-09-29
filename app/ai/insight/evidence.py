"""Deterministic insight evidence built from query results and the Phase 8.1-8.4 analyses.

Key metrics are computed here, never by the model, so every metric an insight
cites can be checked against a number this module produced.
"""

from __future__ import annotations

import json

from app.ai.anomaly_detection.detection import scan_prompt_payload
from app.ai.anomaly_detection.models import AnomalyAnalysisResult
from app.ai.data_analyst.models import DataAnalysisResult
from app.ai.insight.models import KeyMetric
from app.ai.root_cause_analysis.models import RootCauseAnalysisResult
from app.ai.sql_execution.models import SQLExecutionResult
from app.ai.trend_analysis.models import TrendAnalysisResult
from app.ai.trend_analysis.series import (
    column_values,
    parse_number,
    select_period_column,
    series_prompt_payload,
)

MAX_KEY_METRICS = 10
MAX_EVIDENCE_ROWS = 100
MAX_ROOT_CAUSE_HYPOTHESES = 5


def identify_key_metrics(
    query_result: SQLExecutionResult | None,
    *,
    max_metrics: int = MAX_KEY_METRICS,
) -> list[KeyMetric]:
    """Summarize every fully numeric, non-period result column as a key metric."""
    if query_result is None or not query_result.columns or not query_result.rows:
        return []

    columns = list(query_result.columns)
    rows = [list(row) for row in query_result.rows]
    period_index = select_period_column(columns, rows)

    metrics: list[KeyMetric] = []
    for index, column in enumerate(columns):
        if index == period_index or len(metrics) == max_metrics:
            continue
        values = column_values(rows, index)
        numbers = [parse_number(value) for value in values]
        if not numbers or any(number is None for number in numbers):
            continue
        clean = [number for number in numbers if number is not None]
        metrics.append(
            KeyMetric(
                column=column,
                value_count=len(clean),
                total=round(sum(clean), 6),
                average=round(sum(clean) / len(clean), 6),
                minimum=round(min(clean), 6),
                maximum=round(max(clean), 6),
            )
        )
    return metrics


def has_evidence(
    query_result: SQLExecutionResult | None,
    analysis: DataAnalysisResult | None,
    trend: TrendAnalysisResult | None,
    anomalies: AnomalyAnalysisResult | None,
    root_cause: RootCauseAnalysisResult | None,
) -> bool:
    """Insights only run when there are actual results or a completed analysis to cite."""
    if query_result is not None and query_result.rows:
        return True
    return any(item is not None for item in (analysis, trend, anomalies, root_cause))


def groundable_names(
    query_result: SQLExecutionResult | None,
    key_metrics: list[KeyMetric],
) -> dict[str, str]:
    """Map lowercased metric and column names to their canonical spelling."""
    names: dict[str, str] = {}
    for column in list(query_result.columns) if query_result else []:
        names[column.strip().lower()] = column
    for metric in key_metrics:
        names[metric.column.strip().lower()] = metric.column
    return names


def build_evidence_payload(
    *,
    query_result: SQLExecutionResult | None,
    key_metrics: list[KeyMetric],
    analysis: DataAnalysisResult | None = None,
    trend: TrendAnalysisResult | None = None,
    anomalies: AnomalyAnalysisResult | None = None,
    root_cause: RootCauseAnalysisResult | None = None,
) -> str:
    """Render all available analysis evidence as bounded JSON for the prompt."""
    payload: dict[str, object] = {}

    if key_metrics:
        payload["key_metrics"] = [metric.model_dump(mode="json") for metric in key_metrics]

    if query_result is not None and query_result.rows:
        payload["query_rows"] = {
            "columns": list(query_result.columns),
            "rows": query_result.rows[:MAX_EVIDENCE_ROWS],
            "shown_row_count": min(len(query_result.rows), MAX_EVIDENCE_ROWS),
            "total_row_count": query_result.row_count,
            "truncated": query_result.truncated or len(query_result.rows) > MAX_EVIDENCE_ROWS,
        }

    if analysis is not None:
        payload["data_analysis"] = {
            "interpretation": analysis.interpretation,
            "summary": analysis.summary,
            "comparisons": analysis.comparisons,
            "conclusions": analysis.conclusions,
            "confidence_score": analysis.confidence_score.value,
        }

    if trend is not None:
        payload["trend"] = {
            "direction": trend.direction.value,
            "growth_rate_percent": trend.growth_rate_percent,
            "summary": trend.summary,
            "significant_changes": trend.significant_changes,
            "conclusions": trend.conclusions,
            "confidence_score": trend.confidence_score.value,
            "series": json.loads(series_prompt_payload(trend.series)),
        }

    if anomalies is not None:
        payload["anomalies"] = {
            "anomaly_count": anomalies.anomaly_count,
            "highest_severity": (
                anomalies.highest_severity.value if anomalies.highest_severity else None
            ),
            "summary": anomalies.summary,
            "conclusions": anomalies.conclusions,
            "confidence_score": anomalies.confidence_score.value,
            "scan": json.loads(scan_prompt_payload(anomalies.scan)),
        }

    if root_cause is not None:
        payload["root_cause"] = {
            "summary": root_cause.summary,
            "primary_cause": root_cause.primary_cause,
            "confidence_score": root_cause.confidence_score.value,
            "hypotheses": [
                {
                    "rank": item.rank,
                    "statement": item.statement,
                    "supporting_evidence": item.supporting_evidence,
                    "confidence_score": item.confidence_score.value,
                }
                for item in root_cause.hypotheses[:MAX_ROOT_CAUSE_HYPOTHESES]
            ],
            "notes": root_cause.notes,
        }

    return json.dumps(payload, indent=2, default=str)
