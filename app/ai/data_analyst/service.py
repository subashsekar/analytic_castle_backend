"""Data analyst agent service for query result analysis."""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy.orm import Session

from app.ai.llm import AsyncLLMClient, LLMClientConfig, llm_client_config_from_settings, LLMRequest
from app.ai.prompt import PromptRegistry, bundle_to_llm_messages, structured_output_from_model
from app.ai.state import AgentStateService
from app.ai.state.models import AnalysisSessionSnapshot
from app.db.models import DataSource
from app.ai.sql_execution.models import SQLExecutionResult
from app.ai.data_analyst.errors import (
    DataAnalystAuthorizationError,
    DataAnalystConfigurationError,
    DataAnalystLLMError,
    DataAnalystValidationError,
)
from app.ai.data_analyst.models import DataAnalysisResult, LLMDataAnalystOutput
from app.ai.data_analyst.prompts import (
    DATA_ANALYST_BUNDLE_VERSION,
    DATA_ANALYST_SYSTEM_PROMPT_ID,
    DATA_ANALYST_USER_TEMPLATE_ID,
    DataAnalystVariables,
    build_data_analyst_prompt_registry,
)

logger = logging.getLogger(__name__)


class DataAnalystAgent:
    """Analyze query results and produce structured analysis."""

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
        self._prompt_registry = prompt_registry or build_data_analyst_prompt_registry()
        if llm_client is not None:
            self._llm_client = llm_client
        else:
            config = llm_config or llm_client_config_from_settings()
            if not config.api_key:
                raise DataAnalystConfigurationError("LLM API key is not configured")
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
        expected_agent_version: int | None = None,
    ) -> DataAnalysisResult:
        # 1. Authorize access
        snapshot = self._state_service.get_session(
            session_id,
            workspace_id=workspace_id,
            user_id=user_id,
        )
        self._ensure_data_source_access(snapshot, workspace_id)

        # 2. Format query results for the prompt
        columns_str = "None"
        rows_str = "None"
        row_count_str = "0"
        truncated_str = "false"
        sql_str = sql or "None"

        if query_result is not None:
            columns_str = ", ".join(query_result.columns) if query_result.columns else "[]"
            rows_str = str(query_result.rows) if query_result.rows else "[]"
            row_count_str = str(query_result.row_count)
            truncated_str = "true" if query_result.truncated else "false"

        # 3. Build prompt bundle
        bundle = self._prompt_registry.build_bundle(
            bundle_version=DATA_ANALYST_BUNDLE_VERSION,
            system_prompt_id=DATA_ANALYST_SYSTEM_PROMPT_ID,
            user_template_id=DATA_ANALYST_USER_TEMPLATE_ID,
            user_variables=DataAnalystVariables(
                message=message,
                sql=sql_str,
                columns=columns_str,
                rows=rows_str,
                row_count=row_count_str,
                truncated=truncated_str,
            ),
            structured_output=structured_output_from_model(LLMDataAnalystOutput),
        )

        request = LLMRequest(
            model=self._llm_client.config.model,
            messages=bundle_to_llm_messages(bundle),
            structured_output=bundle.structured_output,
        )

        # 4. Call LLM
        try:
            response = await self._llm_client.complete(request)
        except Exception as exc:
            raise DataAnalystLLMError(str(exc)) from exc

        if not response.content:
            raise DataAnalystLLMError("Data analyst response was empty")

        # 5. Parse and validate response
        try:
            from app.ai.llm.content import extract_json_object
            payload = extract_json_object(response.content)
            llm_output = LLMDataAnalystOutput.model_validate(payload)
        except Exception as exc:
            raise DataAnalystValidationError(f"Data analyst response failed validation: {exc}") from exc

        # 6. Build final result
        result = DataAnalysisResult(
            interpretation=llm_output.interpretation,
            summary=llm_output.summary,
            comparisons=llm_output.comparisons,
            conclusions=llm_output.conclusions,
            confidence_score=llm_output.confidence_score,
            confidence_reasoning=llm_output.confidence_reasoning,
        )

        # 7. Optionally persist or update agent state if version is provided
        if expected_agent_version is not None:
            payload_custom = dict(snapshot.agent_state.payload.custom) if snapshot.agent_state else {}
            payload_custom["analyst_action"] = "ANALYSIS_COMPLETED"
            payload_custom["confidence_score"] = result.confidence_score.value
            
            from app.ai.state import MergePayloadTransition
            self._state_service.update_agent_state(
                session_id,
                MergePayloadTransition(updates={"custom": payload_custom}),
                workspace_id=workspace_id,
                user_id=user_id,
                expected_version=expected_agent_version,
            )

        return result

    def _ensure_data_source_access(
        self,
        snapshot: AnalysisSessionSnapshot,
        workspace_id: UUID,
    ) -> None:
        if snapshot.data_source_id is None:
            return
        data_source = self._session.get(DataSource, snapshot.data_source_id)
        if data_source is None or data_source.workspace_id != workspace_id:
            raise DataAnalystAuthorizationError("Data source is not accessible")
