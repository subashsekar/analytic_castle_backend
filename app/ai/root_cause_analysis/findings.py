"""Deterministic finding evidence built from the Phase 8.2/8.3 analysis results."""

from __future__ import annotations

import json

from app.ai.anomaly_detection.detection import scan_prompt_payload
from app.ai.anomaly_detection.models import AnomalyAnalysisResult
from app.ai.trend_analysis.models import TrendAnalysisResult, TrendDirection
from app.ai.trend_analysis.series import series_prompt_payload

_EXPLAINABLE_DIRECTIONS = frozenset(
    {TrendDirection.INCREASING, TrendDirection.DECREASING, TrendDirection.VOLATILE}
)


def trend_is_explainable(trend: TrendAnalysisResult | None) -> bool:
    if trend is None:
        return False
    return (
        trend.direction in _EXPLAINABLE_DIRECTIONS
        or bool(trend.series.significant_changes)
    )


def has_findings(
    trend: TrendAnalysisResult | None,
    anomalies: AnomalyAnalysisResult | None,
) -> bool:
    """Root cause analysis only runs when there is a detected change to explain."""
    return trend_is_explainable(trend) or bool(anomalies and anomalies.anomaly_count)


def build_findings_payload(
    trend: TrendAnalysisResult | None,
    anomalies: AnomalyAnalysisResult | None,
) -> str:
    """Render the detected trend and anomalies as bounded JSON evidence for the prompt."""
    payload: dict[str, object] = {}
    if trend_is_explainable(trend) and trend is not None:
        payload["trend"] = {
            "direction": trend.direction.value,
            "growth_rate_percent": trend.growth_rate_percent,
            "summary": trend.summary,
            "significant_changes": trend.significant_changes,
            "confidence_score": trend.confidence_score.value,
            "series": json.loads(series_prompt_payload(trend.series)),
        }
    if anomalies is not None and anomalies.anomaly_count:
        payload["anomalies"] = {
            "anomaly_count": anomalies.anomaly_count,
            "highest_severity": (
                anomalies.highest_severity.value if anomalies.highest_severity else None
            ),
            "summary": anomalies.summary,
            "confidence_score": anomalies.confidence_score.value,
            "scan": json.loads(scan_prompt_payload(anomalies.scan)),
        }
    return json.dumps(payload, indent=2, default=str)
