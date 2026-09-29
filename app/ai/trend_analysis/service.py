"""Trend analysis agent service for time-series query results."""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy.orm import Session

from app.ai.data_analyst.models import AnalysisConfidence
from app.ai.llm import AsyncLLMClient, LLMClientConfig, llm_client_config_from_settings, LLMRequest
from app.ai.llm.content import extract_json_object
from app.ai.prompt import PromptRegistry, bundle_to_llm_messages, structured_output_from_model
from app.ai.sql_execution.models import SQLExecutionResult
from app.ai.state import AgentStateService, MergePayloadTransition
from app.ai.state.models import AnalysisSessionSnapshot
from app.ai.trend_analysis.errors import (
    TrendAnalysisAuthorizationError,
    TrendAnalysisConfigurationError,
    TrendAnalysisLLMError,
    TrendAnalysisValidationError,
)
from app.ai.trend_analysis.models import (
    LLMTrendAnalysisOutput,
    TrendAnalysisResult,
    TrendDirection,
    TrendSeries,
)
from app.ai.trend_analysis.prompts import (
    TREND_ANALYSIS_BUNDLE_VERSION,
    TREND_ANALYSIS_SYSTEM_PROMPT_ID,
    TREND_ANALYSIS_USER_TEMPLATE_ID,
    TrendAnalysisVariables,
    build_trend_analysis_prompt_registry,
)
from app.ai.trend_analysis.series import build_trend_series, series_prompt_payload
from app.db.models import DataSource

logger = logging.getLogger(__name__)

_NO_TREND_SUMMARY = "No trend could be computed from these query results."


class TrendAnalysisAgent:
    """Detect growth, decline, and significant changes in time-series query results."""

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
        self._prompt_registry = prompt_registry or build_trend_analysis_prompt_registry()
        if llm_client is not None:
            self._llm_client = llm_client
        else:
            config = llm_config or llm_client_config_from_settings()
            if not config.api_key:
                raise TrendAnalysisConfigurationError("LLM API key is not configured")
            self._llm_client = AsyncLLMClient(config)

    async def analyze(
        self,
        *,
        session_id: UUID,
        workspace_id: UUID,
        user_id: UUID,
        message: str,
        sql: str | None = None,
        query_result: SQLExecutionResult | None = None,
        period_column: str | None = None,
        value_column: str | None = None,
        expected_agent_version: int | None = None,
    ) -> TrendAnalysisResult:
        # 1. Authorize access
        snapshot = self._state_service.get_session(
            session_id,
            workspace_id=workspace_id,
            user_id=user_id,
        )
        self._ensure_data_source_access(snapshot, workspace_id)

        # 2. Compute the series deterministically before involving the LLM
        series = build_trend_series(
            list(query_result.columns) if query_result else [],
            list(query_result.rows) if query_result else [],
            period_column=period_column,
            value_column=value_column,
        )

        # 3. Unusable data is answered without an LLM round trip
        if series.direction is TrendDirection.INSUFFICIENT_DATA:
            logger.info(
                "trend_analysis.insufficient_data",
                extra={"session_id": str(session_id), "point_count": series.point_count},
            )
            result = _insufficient_data_result(series)
            self._record_state(
                session_id,
                workspace_id=workspace_id,
                user_id=user_id,
                snapshot=snapshot,
                result=result,
                action="TREND_ANALYSIS_SKIPPED",
                expected_agent_version=expected_agent_version,
            )
            return result

        # 4. Build prompt bundle over the computed evidence
        bundle = self._prompt_registry.build_bundle(
            bundle_version=TREND_ANALYSIS_BUNDLE_VERSION,
            system_prompt_id=TREND_ANALYSIS_SYSTEM_PROMPT_ID,
            user_template_id=TREND_ANALYSIS_USER_TEMPLATE_ID,
            user_variables=TrendAnalysisVariables(
                message=message,
                sql=sql or "None",
                columns=", ".join(query_result.columns) if query_result and query_result.columns else "[]",
                row_count=str(query_result.row_count) if query_result else "0",
                truncated="true" if query_result and query_result.truncated else "false",
                trend_metrics=series_prompt_payload(series),
            ),
            structured_output=structured_output_from_model(LLMTrendAnalysisOutput),
        )

        request = LLMRequest(
            model=self._llm_client.config.model,
            messages=bundle_to_llm_messages(bundle),
            structured_output=bundle.structured_output,
        )

        # 5. Call LLM
        try:
            response = await self._llm_client.complete(request)
        except Exception as exc:
            raise TrendAnalysisLLMError(str(exc)) from exc

        if not response.content:
            raise TrendAnalysisLLMError("Trend analysis response was empty")

        # 6. Parse and validate response
        try:
            payload = extract_json_object(response.content)
            llm_output = LLMTrendAnalysisOutput.model_validate(payload)
        except Exception as exc:
            raise TrendAnalysisValidationError(f"Trend analysis response failed validation: {exc}") from exc

        # 7. Computed metrics win over anything the model claimed
        result = TrendAnalysisResult(
            direction=series.direction,
            growth_rate_percent=series.growth_rate_percent,
            series=series,
            summary=llm_output.summary,
            direction_explanation=llm_output.direction_explanation,
            period_comparisons=llm_output.period_comparisons,
            significant_changes=llm_output.significant_changes,
            conclusions=llm_output.conclusions,
            confidence_score=llm_output.confidence_score,
            confidence_reasoning=llm_output.confidence_reasoning,
        )

        self._record_state(
            session_id,
            workspace_id=workspace_id,
            user_id=user_id,
            snapshot=snapshot,
            result=result,
            action="TREND_ANALYSIS_COMPLETED",
            expected_agent_version=expected_agent_version,
        )
        return result

    def _record_state(
        self,
        session_id: UUID,
        *,
        workspace_id: UUID,
        user_id: UUID,
        snapshot: AnalysisSessionSnapshot,
        result: TrendAnalysisResult,
        action: str,
        expected_agent_version: int | None,
    ) -> None:
        if expected_agent_version is None:
            return
        custom = dict(snapshot.agent_state.payload.custom) if snapshot.agent_state else {}
        custom["trend_action"] = action
        custom["trend_direction"] = result.direction.value
        custom["trend_confidence_score"] = result.confidence_score.value
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
            raise TrendAnalysisAuthorizationError("Data source is not accessible")


def _insufficient_data_result(series: TrendSeries) -> TrendAnalysisResult:
    reason = " ".join(series.notes) or "The query results do not contain a usable time series."
    return TrendAnalysisResult(
        direction=TrendDirection.INSUFFICIENT_DATA,
        growth_rate_percent=None,
        series=series,
        summary=f"{_NO_TREND_SUMMARY} {reason}",
        direction_explanation="No trend direction could be determined.",
        period_comparisons="No period-over-period comparison was possible.",
        significant_changes="No significant trend change could be detected.",
        conclusions=[],
        confidence_score=AnalysisConfidence.LOW,
        confidence_reasoning=reason,
    )
