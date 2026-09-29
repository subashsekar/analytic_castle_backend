"""Recommendation evidence: the Phase 8.1-8.4 analysis payload plus the 8.5 insights.

The analysis sections are built by the insight agent's evidence module so a
recommendation can only cite numbers that module already computed.
"""

from __future__ import annotations

import json

from app.ai.anomaly_detection.models import AnomalyAnalysisResult
from app.ai.data_analyst.models import DataAnalysisResult
from app.ai.insight.evidence import build_evidence_payload as build_analysis_payload
from app.ai.insight.evidence import groundable_names as analysis_groundable_names
from app.ai.insight.evidence import has_evidence as has_analysis_evidence
from app.ai.insight.models import InsightAnalysisResult, KeyMetric
from app.ai.root_cause_analysis.models import RootCauseAnalysisResult
from app.ai.sql_execution.models import SQLExecutionResult
from app.ai.trend_analysis.models import TrendAnalysisResult

MAX_INSIGHTS = 10


def has_evidence(
    query_result: SQLExecutionResult | None,
    analysis: DataAnalysisResult | None,
    trend: TrendAnalysisResult | None,
    anomalies: AnomalyAnalysisResult | None,
    root_cause: RootCauseAnalysisResult | None,
    insights: InsightAnalysisResult | None,
) -> bool:
    """Recommendations only run when there is a finding or result to act on."""
    if insights is not None and insights.insights:
        return True
    return has_analysis_evidence(query_result, analysis, trend, anomalies, root_cause)


def groundable_names(
    query_result: SQLExecutionResult | None,
    key_metrics: list[KeyMetric],
    insights: InsightAnalysisResult | None,
) -> dict[str, str]:
    """Map lowercased metric, column, and insight titles to their canonical spelling."""
    names = analysis_groundable_names(query_result, key_metrics)
    for item in insights.insights if insights else []:
        names[item.title.strip().lower()] = item.title
        if item.metric:
            names.setdefault(item.metric.strip().lower(), item.metric)
    return names


def build_evidence_payload(
    *,
    query_result: SQLExecutionResult | None,
    key_metrics: list[KeyMetric],
    analysis: DataAnalysisResult | None = None,
    trend: TrendAnalysisResult | None = None,
    anomalies: AnomalyAnalysisResult | None = None,
    root_cause: RootCauseAnalysisResult | None = None,
    insights: InsightAnalysisResult | None = None,
) -> str:
    """Render all available analysis and insight evidence as bounded JSON for the prompt."""
    payload = json.loads(
        build_analysis_payload(
            query_result=query_result,
            key_metrics=key_metrics,
            analysis=analysis,
            trend=trend,
            anomalies=anomalies,
            root_cause=root_cause,
        )
    )

    if insights is not None:
        payload["business_insights"] = {
            "summary": insights.summary,
            "top_insight": insights.top_insight,
            "confidence_score": insights.confidence_score.value,
            "data_gaps": insights.data_gaps,
            "insights": [
                {
                    "rank": item.rank,
                    "title": item.title,
                    "insight": item.insight,
                    "metric": item.metric,
                    "business_impact": item.business_impact,
                    "supporting_evidence": item.supporting_evidence,
                    "priority": item.priority.value,
                    "confidence_score": item.confidence_score.value,
                }
                for item in insights.insights[:MAX_INSIGHTS]
            ],
        }

    return json.dumps(payload, indent=2, default=str)
