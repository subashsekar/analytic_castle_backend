"""Recommendation agent service turning insights into prioritized business actions."""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy.orm import Session

from app.ai.anomaly_detection.models import AnomalyAnalysisResult
from app.ai.data_analyst.models import AnalysisConfidence, DataAnalysisResult
from app.ai.insight.evidence import identify_key_metrics
from app.ai.insight.models import InsightAnalysisResult
from app.ai.llm import (
    AsyncLLMClient,
    LLMClientConfig,
    LLMRequest,
    llm_client_config_from_settings,
)
from app.ai.llm.content import extract_json_object
from app.ai.prompt import (
    PromptRegistry,
    bundle_to_llm_messages,
    structured_output_from_model,
)
from app.ai.recommendation.errors import (
    RecommendationAuthorizationError,
    RecommendationConfigurationError,
    RecommendationLLMError,
    RecommendationValidationError,
)
from app.ai.recommendation.evidence import (
    build_evidence_payload,
    groundable_names,
    has_evidence,
)
from app.ai.recommendation.models import (
    LLMRecommendation,
    LLMRecommendationOutput,
    Recommendation,
    RecommendationResult,
    derive_priority,
    level_rank,
)
from app.ai.recommendation.prompts import (
    RECOMMENDATION_BUNDLE_VERSION,
    RECOMMENDATION_SYSTEM_PROMPT_ID,
    RECOMMENDATION_USER_TEMPLATE_ID,
    RecommendationVariables,
    build_recommendation_prompt_registry,
)
from app.ai.root_cause_analysis.models import RootCauseAnalysisResult, confidence_rank
from app.ai.sql_execution.models import SQLExecutionResult
from app.ai.state import AgentStateService, MergePayloadTransition
from app.ai.state.models import AnalysisSessionSnapshot
from app.ai.trend_analysis.models import TrendAnalysisResult
from app.db.models import DataSource

logger = logging.getLogger(__name__)

_NO_EVIDENCE_SUMMARY = (
    "No query results, analysis, or insights were supplied, so no action could be recommended."
)


class RecommendationAgent:
    """Recommend evidence-backed business actions, prioritized by impact and feasibility."""

    def __init__(
        self,
        session: Session,
        *,
        llm_client: AsyncLLMClient | None = None,
        llm_config: LLMClientConfig | None = None,
        prompt_registry: PromptRegistry | None = None,
        state_service: AgentStateService | None = None,
    ) -> None:
        self._session = session
        self._state_service = state_service or AgentStateService(session)
        self._prompt_registry = prompt_registry or build_recommendation_prompt_registry()
        if llm_client is not None:
            self._llm_client = llm_client
        else:
            config = llm_config or llm_client_config_from_settings()
            if not config.api_key:
                raise RecommendationConfigurationError("LLM API key is not configured")
            self._llm_client = AsyncLLMClient(config)

    async def recommend(
        self,
        *,
        session_id: UUID,
        workspace_id: UUID,
        user_id: UUID,
        message: str,
        sql: str | None = None,
        query_result: SQLExecutionResult | None = None,
        analysis: DataAnalysisResult | None = None,
        trend: TrendAnalysisResult | None = None,
        anomalies: AnomalyAnalysisResult | None = None,
        root_cause: RootCauseAnalysisResult | None = None,
        insights: InsightAnalysisResult | None = None,
        expected_agent_version: int | None = None,
    ) -> RecommendationResult:
        # 1. Authorize access
        snapshot = self._state_service.get_session(
            session_id,
            workspace_id=workspace_id,
            user_id=user_id,
        )
        self._ensure_data_source_access(snapshot, workspace_id)

        # 2. Metrics come from the same deterministic computation the insights were grounded on
        key_metrics = identify_key_metrics(query_result)

        # 3. Nothing found means nothing to act on, and no LLM call
        if not has_evidence(query_result, analysis, trend, anomalies, root_cause, insights):
            logger.info("recommendation.no_evidence", extra={"session_id": str(session_id)})
            result = _no_evidence_result()
            self._record_state(
                session_id,
                workspace_id=workspace_id,
                user_id=user_id,
                snapshot=snapshot,
                result=result,
                action="RECOMMENDATION_SKIPPED",
                expected_agent_version=expected_agent_version,
            )
            return result

        llm_output = await self._request_recommendations(
            message=message,
            sql=sql,
            query_result=query_result,
            evidence=build_evidence_payload(
                query_result=query_result,
                key_metrics=key_metrics,
                analysis=analysis,
                trend=trend,
                anomalies=anomalies,
                root_cause=root_cause,
                insights=insights,
            ),
        )

        # 4. Drop anything citing evidence we never supplied, then prioritize
        notes: list[str] = []
        grounded = _grounded_recommendations(
            llm_output.recommendations,
            names=groundable_names(query_result, key_metrics, insights),
            notes=notes,
        )
        if not grounded:
            raise RecommendationValidationError(
                "No recommendation was grounded in the supplied query results, analysis, or insights"
            )
        recommendations = _prioritize(grounded)
        top = recommendations[0]

        result = RecommendationResult(
            summary=llm_output.summary,
            top_recommendation=top.recommendation,
            recommendations=recommendations,
            data_gaps=llm_output.data_gaps,
            confidence_score=top.confidence_score,
            confidence_reasoning=top.confidence_reasoning,
            notes=notes,
        )

        self._record_state(
            session_id,
            workspace_id=workspace_id,
            user_id=user_id,
            snapshot=snapshot,
            result=result,
            action="RECOMMENDATION_COMPLETED",
            expected_agent_version=expected_agent_version,
        )
        return result

    async def _request_recommendations(
        self,
        *,
        message: str,
        sql: str | None,
        query_result: SQLExecutionResult | None,
        evidence: str,
    ) -> LLMRecommendationOutput:
        bundle = self._prompt_registry.build_bundle(
            bundle_version=RECOMMENDATION_BUNDLE_VERSION,
            system_prompt_id=RECOMMENDATION_SYSTEM_PROMPT_ID,
            user_template_id=RECOMMENDATION_USER_TEMPLATE_ID,
            user_variables=RecommendationVariables(
                message=message,
                sql=sql or "None",
                columns=(
                    ", ".join(query_result.columns)
                    if query_result and query_result.columns
                    else "[]"
                ),
                row_count=str(query_result.row_count) if query_result else "0",
                truncated="true" if query_result and query_result.truncated else "false",
                evidence=evidence,
            ),
            structured_output=structured_output_from_model(LLMRecommendationOutput),
        )
        request = LLMRequest(
            model=self._llm_client.config.model,
            messages=bundle_to_llm_messages(bundle),
            structured_output=bundle.structured_output,
        )

        try:
            response = await self._llm_client.complete(request)
        except Exception as exc:
            raise RecommendationLLMError(str(exc)) from exc

        if not response.content:
            raise RecommendationLLMError("Recommendation response was empty")

        try:
            payload = extract_json_object(response.content)
            return LLMRecommendationOutput.model_validate(payload)
        except Exception as exc:
            raise RecommendationValidationError(
                f"Recommendation response failed validation: {exc}"
            ) from exc

    def _record_state(
        self,
        session_id: UUID,
        *,
        workspace_id: UUID,
        user_id: UUID,
        snapshot: AnalysisSessionSnapshot,
        result: RecommendationResult,
        action: str,
        expected_agent_version: int | None,
    ) -> None:
        if expected_agent_version is None:
            return
        custom = dict(snapshot.agent_state.payload.custom) if snapshot.agent_state else {}
        custom["recommendation_action"] = action
        custom["recommendation_count"] = str(len(result.recommendations))
        custom["recommendation_confidence_score"] = result.confidence_score.value
        self._state_service.update_agent_state(
            session_id,
            MergePayloadTransition(updates={"custom": custom}),
            workspace_id=workspace_id,
            user_id=user_id,
            expected_version=expected_agent_version,
        )

    def _ensure_data_source_access(
        self,
        snapshot: AnalysisSessionSnapshot,
        workspace_id: UUID,
    ) -> None:
        if snapshot.data_source_id is None:
            return
        data_source = self._session.get(DataSource, snapshot.data_source_id)
        if data_source is None or data_source.workspace_id != workspace_id:
            raise RecommendationAuthorizationError("Data source is not accessible")


def _grounded_recommendations(
    recommendations: list[LLMRecommendation],
    *,
    names: dict[str, str],
    notes: list[str],
) -> list[LLMRecommendation]:
    """Keep only recommendations whose cited metric, column, or insight we supplied."""
    kept: list[LLMRecommendation] = []
    for item in recommendations:
        canonical = names.get(item.evidence_reference.strip().lower())
        if canonical is None:
            notes.append(
                f"Recommendation {item.title!r} was dropped: it cites "
                f"{item.evidence_reference!r}, which is not present in the query results, "
                "analysis, or insight evidence."
            )
            continue
        kept.append(item.model_copy(update={"evidence_reference": canonical}))
    return kept


def _prioritize(recommendations: list[LLMRecommendation]) -> list[Recommendation]:
    """Rank by impact, then feasibility, then confidence, with LLM order as the tiebreak."""
    ordered = sorted(
        enumerate(recommendations),
        key=lambda pair: (
            -level_rank(pair[1].impact),
            -level_rank(pair[1].feasibility),
            -confidence_rank(pair[1].confidence_score),
            pair[0],
        ),
    )
    return [
        Recommendation(
            rank=rank,
            title=item.title,
            recommendation=item.recommendation,
            evidence_reference=item.evidence_reference,
            supporting_evidence=item.supporting_evidence,
            expected_outcome=item.expected_outcome,
            assumptions=item.assumptions,
            risks=item.risks,
            impact=item.impact,
            feasibility=item.feasibility,
            priority=derive_priority(item.impact, item.feasibility),
            confidence_score=item.confidence_score,
            confidence_reasoning=item.confidence_reasoning,
        )
        for rank, (_, item) in enumerate(ordered, start=1)
    ]


def _no_evidence_result() -> RecommendationResult:
    return RecommendationResult(
        summary=_NO_EVIDENCE_SUMMARY,
        top_recommendation=None,
        recommendations=[],
        confidence_score=AnalysisConfidence.LOW,
        confidence_reasoning=(
            "No query results, upstream analysis, or insights were supplied, so no action "
            "could be justified."
        ),
    )
