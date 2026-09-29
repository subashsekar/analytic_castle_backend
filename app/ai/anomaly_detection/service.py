"""Anomaly detection agent service for query results."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy.orm import Session

from app.ai.anomaly_detection.detection import scan_for_anomalies, scan_prompt_payload
from app.ai.anomaly_detection.errors import (
    AnomalyDetectionAuthorizationError,
    AnomalyDetectionConfigurationError,
    AnomalyDetectionLLMError,
    AnomalyDetectionValidationError,
)
from app.ai.anomaly_detection.models import (
    AnomalyAnalysisResult,
    AnomalyScan,
    ColumnThreshold,
    LLMAnomalyDetectionOutput,
)
from app.ai.anomaly_detection.prompts import (
    ANOMALY_DETECTION_BUNDLE_VERSION,
    ANOMALY_DETECTION_SYSTEM_PROMPT_ID,
    ANOMALY_DETECTION_USER_TEMPLATE_ID,
    AnomalyDetectionVariables,
    build_anomaly_detection_prompt_registry,
)
from app.ai.data_analyst.models import AnalysisConfidence
from app.ai.llm import AsyncLLMClient, LLMClientConfig, llm_client_config_from_settings, LLMRequest
from app.ai.llm.content import extract_json_object
from app.ai.prompt import PromptRegistry, bundle_to_llm_messages, structured_output_from_model
from app.ai.sql_execution.models import SQLExecutionResult
from app.ai.state import AgentStateService, MergePayloadTransition
from app.ai.state.models import AnalysisSessionSnapshot
from app.db.models import DataSource

logger = logging.getLogger(__name__)


class AnomalyDetectionAgent:
    """Detect outliers, unexpected changes, and threshold breaches in query results."""

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
        self._prompt_registry = prompt_registry or build_anomaly_detection_prompt_registry()
        if llm_client is not None:
            self._llm_client = llm_client
        else:
            config = llm_config or llm_client_config_from_settings()
            if not config.api_key:
                raise AnomalyDetectionConfigurationError("LLM API key is not configured")
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
        thresholds: Sequence[ColumnThreshold] | None = None,
        period_column: str | None = None,
        expected_agent_version: int | None = None,
    ) -> AnomalyAnalysisResult:
        # 1. Authorize access
        snapshot = self._state_service.get_session(
            session_id,
            workspace_id=workspace_id,
            user_id=user_id,
        )
        self._ensure_data_source_access(snapshot, workspace_id)

        # 2. Detect deterministically before involving the LLM
        scan = scan_for_anomalies(
            list(query_result.columns) if query_result else [],
            list(query_result.rows) if query_result else [],
            thresholds=thresholds,
            period_column=period_column,
        )

        # 3. Unusable data and clean scans are answered without an LLM round trip
        if not scan.analyzed or not scan.anomalies:
            logger.info(
                "anomaly_detection.no_llm_call",
                extra={
                    "session_id": str(session_id),
                    "analyzed": scan.analyzed,
                    "scanned_row_count": scan.scanned_row_count,
                },
            )
            result = _deterministic_result(scan)
            self._record_state(
                session_id,
                workspace_id=workspace_id,
                user_id=user_id,
                snapshot=snapshot,
                result=result,
                action="ANOMALY_SCAN_COMPLETED" if scan.analyzed else "ANOMALY_SCAN_SKIPPED",
                expected_agent_version=expected_agent_version,
            )
            return result

        # 4. Build prompt bundle over the computed evidence
        bundle = self._prompt_registry.build_bundle(
            bundle_version=ANOMALY_DETECTION_BUNDLE_VERSION,
            system_prompt_id=ANOMALY_DETECTION_SYSTEM_PROMPT_ID,
            user_template_id=ANOMALY_DETECTION_USER_TEMPLATE_ID,
            user_variables=AnomalyDetectionVariables(
                message=message,
                sql=sql or "None",
                columns=", ".join(query_result.columns) if query_result and query_result.columns else "[]",
                row_count=str(query_result.row_count) if query_result else "0",
                truncated="true" if query_result and query_result.truncated else "false",
                scan_results=scan_prompt_payload(scan),
            ),
            structured_output=structured_output_from_model(LLMAnomalyDetectionOutput),
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
            raise AnomalyDetectionLLMError(str(exc)) from exc

        if not response.content:
            raise AnomalyDetectionLLMError("Anomaly detection response was empty")

        # 6. Parse and validate response
        try:
            payload = extract_json_object(response.content)
            llm_output = LLMAnomalyDetectionOutput.model_validate(payload)
        except Exception as exc:
            raise AnomalyDetectionValidationError(
                f"Anomaly detection response failed validation: {exc}"
            ) from exc

        # 7. Detected counts and severities win over anything the model claimed
        result = AnomalyAnalysisResult(
            anomaly_count=len(scan.anomalies),
            highest_severity=scan.highest_severity,
            scan=scan,
            summary=llm_output.summary,
            outliers=llm_output.outliers,
            unexpected_changes=llm_output.unexpected_changes,
            threshold_breaches=llm_output.threshold_breaches,
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
            action="ANOMALY_SCAN_COMPLETED",
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
        result: AnomalyAnalysisResult,
        action: str,
        expected_agent_version: int | None,
    ) -> None:
        if expected_agent_version is None:
            return
        custom = dict(snapshot.agent_state.payload.custom) if snapshot.agent_state else {}
        custom["anomaly_action"] = action
        custom["anomaly_count"] = str(result.anomaly_count)
        custom["anomaly_highest_severity"] = (
            result.highest_severity.value if result.highest_severity else "NONE"
        )
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
            raise AnomalyDetectionAuthorizationError("Data source is not accessible")


def _deterministic_result(scan: AnomalyScan) -> AnomalyAnalysisResult:
    """Answer unscannable data and clean scans without spending an LLM call."""
    notes = " ".join(scan.notes)
    if not scan.analyzed:
        reason = notes or "The query results are not suitable for anomaly detection."
        summary = f"No anomaly scan could be run. {reason}"
        confidence = AnalysisConfidence.LOW
    else:
        scanned = ", ".join(scan.numeric_columns) or "no columns"
        summary = (
            f"No anomalies were detected across {scan.scanned_row_count} row(s) "
            f"in {scanned} under the applied outlier, change, and threshold methods."
        )
        reason = notes or f"{scan.scanned_row_count} row(s) were scanned without limitation."
        confidence = AnalysisConfidence.MEDIUM

    none_found = "No anomalies of this type were detected."
    return AnomalyAnalysisResult(
        anomaly_count=0,
        highest_severity=None,
        scan=scan,
        summary=summary,
        outliers=none_found,
        unexpected_changes=none_found,
        threshold_breaches=none_found,
        conclusions=[],
        confidence_score=confidence,
        confidence_reasoning=reason,
    )
