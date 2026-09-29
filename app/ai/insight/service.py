"""Insight agent service turning analyzed data into prioritized business insights."""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy.orm import Session

from app.ai.anomaly_detection.models import AnomalyAnalysisResult
from app.ai.data_analyst.models import AnalysisConfidence, DataAnalysisResult
from app.ai.insight.errors import (
    InsightAuthorizationError,
    InsightConfigurationError,
    InsightLLMError,
    InsightValidationError,
)
from app.ai.insight.evidence import (
    build_evidence_payload,
    groundable_names,
    has_evidence,
    identify_key_metrics,
)
from app.ai.insight.models import (
    BusinessInsight,
    InsightAnalysisResult,
    LLMBusinessInsight,
    LLMInsightOutput,
    priority_rank,
)
from app.ai.insight.prompts import (
    INSIGHT_BUNDLE_VERSION,
    INSIGHT_SYSTEM_PROMPT_ID,
    INSIGHT_USER_TEMPLATE_ID,
    InsightVariables,
    build_insight_prompt_registry,
)
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
from app.ai.root_cause_analysis.models import RootCauseAnalysisResult, confidence_rank
from app.ai.sql_execution.models import SQLExecutionResult
from app.ai.state import AgentStateService, MergePayloadTransition
from app.ai.state.models import AnalysisSessionSnapshot
from app.ai.trend_analysis.models import TrendAnalysisResult
from app.db.models import DataSource

logger = logging.getLogger(__name__)

_NO_EVIDENCE_SUMMARY = (
    "No query results or analysis were supplied, so no business insight could be derived."
)


class InsightAgent:
    """Derive prioritized, evidence-backed business insights from analyzed data."""

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
        self._prompt_registry = prompt_registry or build_insight_prompt_registry()
        if llm_client is not None:
            self._llm_client = llm_client
        else:
            config = llm_config or llm_client_config_from_settings()
            if not config.api_key:
                raise InsightConfigurationError("LLM API key is not configured")
            self._llm_client = AsyncLLMClient(config)

    async def generate(
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
        expected_agent_version: int | None = None,
    ) -> InsightAnalysisResult:
        # 1. Authorize access
        snapshot = self._state_service.get_session(
            session_id,
            workspace_id=workspace_id,
            user_id=user_id,
        )
        self._ensure_data_source_access(snapshot, workspace_id)

        # 2. Key metrics are computed here, not narrated by the model
        key_metrics = identify_key_metrics(query_result)

        # 3. Nothing analyzed means nothing to conclude, and no LLM call
        if not has_evidence(query_result, analysis, trend, anomalies, root_cause):
            logger.info("insight.no_evidence", extra={"session_id": str(session_id)})
            result = _no_evidence_result()
            self._record_state(
                session_id,
                workspace_id=workspace_id,
                user_id=user_id,
                snapshot=snapshot,
                result=result,
                action="INSIGHT_SKIPPED",
                expected_agent_version=expected_agent_version,
            )
            return result

        llm_output = await self._request_insights(
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
            ),
        )

        # 4. Drop anything citing a metric we never computed, then prioritize
        notes: list[str] = []
        grounded = _grounded_insights(
            llm_output.insights,
            names=groundable_names(query_result, key_metrics),
            notes=notes,
        )
        if not grounded:
            raise InsightValidationError(
                "No insight was grounded in the supplied query results or analysis evidence"
            )
        insights = _prioritize(grounded)
        top = insights[0]

        result = InsightAnalysisResult(
            summary=llm_output.summary,
            top_insight=top.insight,
            insights=insights,
            key_metrics=key_metrics,
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
            action="INSIGHT_COMPLETED",
            expected_agent_version=expected_agent_version,
        )
        return result

    async def _request_insights(
        self,
        *,
        message: str,
        sql: str | None,
        query_result: SQLExecutionResult | None,
        evidence: str,
    ) -> LLMInsightOutput:
        bundle = self._prompt_registry.build_bundle(
            bundle_version=INSIGHT_BUNDLE_VERSION,
            system_prompt_id=INSIGHT_SYSTEM_PROMPT_ID,
            user_template_id=INSIGHT_USER_TEMPLATE_ID,
            user_variables=InsightVariables(
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
            structured_output=structured_output_from_model(LLMInsightOutput),
        )
        request = LLMRequest(
            model=self._llm_client.config.model,
            messages=bundle_to_llm_messages(bundle),
            structured_output=bundle.structured_output,
        )

        try:
            response = await self._llm_client.complete(request)
        except Exception as exc:
            raise InsightLLMError(str(exc)) from exc

        if not response.content:
            raise InsightLLMError("Insight response was empty")

        try:
            payload = extract_json_object(response.content)
            return LLMInsightOutput.model_validate(payload)
        except Exception as exc:
            raise InsightValidationError(
                f"Insight response failed validation: {exc}"
            ) from exc

    def _record_state(
        self,
        session_id: UUID,
        *,
        workspace_id: UUID,
        user_id: UUID,
        snapshot: AnalysisSessionSnapshot,
        result: InsightAnalysisResult,
        action: str,
        expected_agent_version: int | None,
    ) -> None:
        if expected_agent_version is None:
            return
        custom = dict(snapshot.agent_state.payload.custom) if snapshot.agent_state else {}
        custom["insight_action"] = action
        custom["insight_count"] = str(len(result.insights))
        custom["insight_confidence_score"] = result.confidence_score.value
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
            raise InsightAuthorizationError("Data source is not accessible")


def _grounded_insights(
    insights: list[LLMBusinessInsight],
    *,
    names: dict[str, str],
    notes: list[str],
) -> list[LLMBusinessInsight]:
    """Keep only insights whose cited metric exists in the evidence we supplied."""
    kept: list[LLMBusinessInsight] = []
    for item in insights:
        if item.metric is None:
            kept.append(item)
            continue
        canonical = names.get(item.metric.strip().lower())
        if canonical is None:
            notes.append(
                f"Insight {item.title!r} was dropped: it cites metric {item.metric!r}, "
                "which is not present in the query results or analysis evidence."
            )
            continue
        kept.append(item.model_copy(update={"metric": canonical}))
    return kept


def _prioritize(insights: list[LLMBusinessInsight]) -> list[BusinessInsight]:
    ordered = sorted(
        enumerate(insights),
        key=lambda pair: (
            -priority_rank(pair[1].priority),
            -confidence_rank(pair[1].confidence_score),
            pair[0],
        ),
    )
    return [
        BusinessInsight(
            rank=rank,
            title=item.title,
            insight=item.insight,
            metric=item.metric,
            business_impact=item.business_impact,
            supporting_evidence=item.supporting_evidence,
            priority=item.priority,
            confidence_score=item.confidence_score,
            confidence_reasoning=item.confidence_reasoning,
        )
        for rank, (_, item) in enumerate(ordered, start=1)
    ]


def _no_evidence_result() -> InsightAnalysisResult:
    return InsightAnalysisResult(
        summary=_NO_EVIDENCE_SUMMARY,
        top_insight=None,
        insights=[],
        confidence_score=AnalysisConfidence.LOW,
        confidence_reasoning=(
            "No query results or upstream analysis were supplied, so nothing could be concluded."
        ),
    )
