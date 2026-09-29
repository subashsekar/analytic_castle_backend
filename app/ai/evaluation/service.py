"""Agent evaluation service: runs the deterministic checks and builds the report.

Evaluation reads already-validated agent outputs, so it makes no LLM call, opens
no database session, and adds no authorization surface of its own; the caller
has already been authorized by the agent that produced the outputs.
"""

from __future__ import annotations

import logging
from uuid import UUID

from app.ai.anomaly_detection.models import AnomalyAnalysisResult
from app.ai.data_analyst.models import DataAnalysisResult
from app.ai.evaluation.checks import (
    evaluate_anomalies,
    evaluate_data_analyst,
    evaluate_insight,
    evaluate_recommendation,
    evaluate_root_cause,
    evaluate_trend,
)
from app.ai.evaluation.models import (
    AgentEvaluation,
    CheckStatus,
    EvaluatedAgent,
    EvaluationMethod,
    EvaluationMetrics,
    EvaluationReport,
)
from app.ai.insight.evidence import identify_key_metrics
from app.ai.insight.models import InsightAnalysisResult
from app.ai.recommendation.models import RecommendationResult
from app.ai.root_cause_analysis.models import RootCauseAnalysisResult
from app.ai.sql_execution.models import SQLExecutionResult
from app.ai.trend_analysis.models import TrendAnalysisResult

logger = logging.getLogger(__name__)


def evaluate_agents(
    *,
    session_id: UUID | None = None,
    query_result: SQLExecutionResult | None = None,
    analysis: DataAnalysisResult | None = None,
    trend: TrendAnalysisResult | None = None,
    anomalies: AnomalyAnalysisResult | None = None,
    root_cause: RootCauseAnalysisResult | None = None,
    insights: InsightAnalysisResult | None = None,
    recommendations: RecommendationResult | None = None,
) -> EvaluationReport:
    """Evaluate whichever Phase 8 agent outputs were supplied for one analysis run."""
    key_metrics = (
        list(insights.key_metrics) if insights is not None else identify_key_metrics(query_result)
    )

    evaluations: list[AgentEvaluation] = []
    not_evaluated: list[EvaluatedAgent] = []

    if analysis is not None:
        evaluations.append(
            AgentEvaluation.from_checks(
                EvaluatedAgent.DATA_ANALYST,
                evaluate_data_analyst(
                    analysis,
                    query_result=query_result,
                    key_metrics=key_metrics,
                ),
            )
        )
    else:
        not_evaluated.append(EvaluatedAgent.DATA_ANALYST)

    if trend is not None:
        evaluations.append(
            AgentEvaluation.from_checks(
                EvaluatedAgent.TREND_ANALYSIS,
                evaluate_trend(trend, query_result=query_result),
            )
        )
    else:
        not_evaluated.append(EvaluatedAgent.TREND_ANALYSIS)

    if anomalies is not None:
        evaluations.append(
            AgentEvaluation.from_checks(
                EvaluatedAgent.ANOMALY_DETECTION,
                evaluate_anomalies(anomalies, query_result=query_result),
            )
        )
    else:
        not_evaluated.append(EvaluatedAgent.ANOMALY_DETECTION)

    if root_cause is not None:
        evaluations.append(
            AgentEvaluation.from_checks(
                EvaluatedAgent.ROOT_CAUSE_ANALYSIS,
                evaluate_root_cause(
                    root_cause,
                    query_result=query_result,
                    key_metrics=key_metrics,
                ),
            )
        )
    else:
        not_evaluated.append(EvaluatedAgent.ROOT_CAUSE_ANALYSIS)

    if insights is not None:
        evaluations.append(
            AgentEvaluation.from_checks(
                EvaluatedAgent.INSIGHT,
                evaluate_insight(insights, query_result=query_result),
            )
        )
    else:
        not_evaluated.append(EvaluatedAgent.INSIGHT)

    if recommendations is not None:
        evaluations.append(
            AgentEvaluation.from_checks(
                EvaluatedAgent.RECOMMENDATION,
                evaluate_recommendation(
                    recommendations,
                    query_result=query_result,
                    insights=insights,
                ),
            )
        )
    else:
        not_evaluated.append(EvaluatedAgent.RECOMMENDATION)

    notes: list[str] = []
    if not evaluations:
        notes.append("No agent outputs were supplied, so nothing could be evaluated.")
    if query_result is None:
        notes.append(
            "No query result was supplied, so cited numbers were checked only against the "
            "evidence carried inside the agent outputs."
        )

    report = EvaluationReport(
        session_id=session_id,
        agents=evaluations,
        agents_not_evaluated=not_evaluated,
        metrics=_metrics(evaluations, query_result=query_result),
        verified_correct=bool(evaluations) and all(item.verified_correct for item in evaluations),
        notes=notes,
    )

    logger.info(
        "evaluation.completed",
        extra={
            "session_id": str(session_id) if session_id else None,
            "agents_evaluated": report.metrics.agents_evaluated,
            "verified_failures": report.metrics.verified_failures,
            "estimated_warnings": report.metrics.estimated_warnings,
            "verified_correct": report.verified_correct,
        },
    )
    return report


def _metrics(
    evaluations: list[AgentEvaluation],
    *,
    query_result: SQLExecutionResult | None,
) -> EvaluationMetrics:
    checks = [check for item in evaluations for check in item.checks]
    verified = [check for check in checks if check.method is EvaluationMethod.VERIFIED]
    estimated = [check for check in checks if check.method is EvaluationMethod.ESTIMATED]
    verified_decided = [check for check in verified if check.status is not CheckStatus.SKIPPED]
    verified_passes = sum(1 for check in verified_decided if check.status is CheckStatus.PASS)
    decided = verified_decided + [
        check for check in estimated if check.status is not CheckStatus.SKIPPED
    ]
    passes = sum(1 for check in decided if check.status is CheckStatus.PASS)

    return EvaluationMetrics(
        agents_evaluated=len(evaluations),
        checks_run=len(checks),
        verified_checks=len(verified),
        verified_failures=sum(1 for check in verified if check.status is CheckStatus.FAIL),
        estimated_checks=len(estimated),
        estimated_warnings=sum(1 for check in estimated if check.status is CheckStatus.WARN),
        skipped_checks=sum(1 for check in checks if check.status is CheckStatus.SKIPPED),
        verified_pass_rate=(
            round(verified_passes / len(verified_decided), 4) if verified_decided else 0.0
        ),
        estimated_quality_score=round(passes / len(decided), 4) if decided else 0.0,
        query_duration_ms=query_result.duration_ms if query_result is not None else None,
    )
