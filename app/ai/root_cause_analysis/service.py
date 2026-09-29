"""Root cause analysis agent service for detected trends and anomalies."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy.orm import Session

from app.ai.anomaly_detection.models import AnomalyAnalysisResult
from app.ai.data_analyst.models import AnalysisConfidence
from app.ai.llm import (
    AsyncLLMClient,
    LLMClientConfig,
    LLMRequest,
    llm_client_config_from_settings,
)
from app.ai.llm.content import extract_json_object
from app.ai.metadata_types import ResolvedMetadataContext
from app.ai.prompt import (
    PromptRegistry,
    bundle_to_llm_messages,
    structured_output_from_model,
)
from app.ai.root_cause_analysis.errors import (
    RootCauseAnalysisAuthorizationError,
    RootCauseAnalysisConfigurationError,
    RootCauseAnalysisLLMError,
    RootCauseAnalysisValidationError,
)
from app.ai.root_cause_analysis.evidence import (
    evidence_prompt_payload,
    gather_evidence,
)
from app.ai.root_cause_analysis.findings import build_findings_payload, has_findings
from app.ai.root_cause_analysis.models import (
    LLMRootCauseOutput,
    RootCauseAnalysisResult,
    RootCauseEvidence,
    RootCauseHypothesis,
    confidence_rank,
)
from app.ai.root_cause_analysis.prompts import (
    ROOT_CAUSE_ANALYSIS_BUNDLE_VERSION,
    ROOT_CAUSE_ANALYSIS_SYSTEM_PROMPT_ID,
    ROOT_CAUSE_ANALYSIS_USER_TEMPLATE_ID,
    RootCauseAnalysisVariables,
    build_root_cause_analysis_prompt_registry,
)
from app.ai.sql_execution.service import SQLExecutionService
from app.ai.sql_execution.models import SQLExecutionResult
from app.ai.sql_generation.service import SQLGenerationService
from app.ai.state import AgentStateService, MergePayloadTransition
from app.ai.state.models import AnalysisSessionSnapshot
from app.ai.trend_analysis.models import TrendAnalysisResult
from app.db.models import DataSource

logger = logging.getLogger(__name__)

MAX_INVESTIGATION_QUERIES = 3

_NO_FINDING_SUMMARY = (
    "No trend change or anomaly was detected, so there is no finding to explain."
)


class RootCauseAnalysisAgent:
    """Explain detected trends and anomalies with ranked, evidence-backed hypotheses."""

    def __init__(
        self,
        session: Session,
        *,
        llm_client: AsyncLLMClient | None = None,
        llm_config: LLMClientConfig | None = None,
        prompt_registry: PromptRegistry | None = None,
        state_service: AgentStateService | None = None,
        sql_generation_service: SQLGenerationService | None = None,
        sql_execution_service: SQLExecutionService | None = None,
    ) -> None:
        self._session = session
        self._state_service = state_service or AgentStateService(session)
        self._prompt_registry = (
            prompt_registry or build_root_cause_analysis_prompt_registry()
        )
        self._sql_generation_service = sql_generation_service
        self._sql_execution_service = sql_execution_service
        if llm_client is not None:
            self._llm_client = llm_client
        else:
            config = llm_config or llm_client_config_from_settings()
            if not config.api_key:
                raise RootCauseAnalysisConfigurationError("LLM API key is not configured")
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
        trend: TrendAnalysisResult | None = None,
        anomalies: AnomalyAnalysisResult | None = None,
        organization_id: UUID | None = None,
        metadata: ResolvedMetadataContext | None = None,
        max_investigation_queries: int = MAX_INVESTIGATION_QUERIES,
        expected_agent_version: int | None = None,
    ) -> RootCauseAnalysisResult:
        # 1. Authorize access
        snapshot = self._state_service.get_session(
            session_id,
            workspace_id=workspace_id,
            user_id=user_id,
        )
        self._ensure_data_source_access(snapshot, workspace_id)

        # 2. Nothing detected means nothing to explain, and no LLM call
        if not has_findings(trend, anomalies):
            logger.info(
                "root_cause_analysis.no_finding",
                extra={"session_id": str(session_id)},
            )
            result = _no_finding_result()
            self._record_state(
                session_id,
                workspace_id=workspace_id,
                user_id=user_id,
                snapshot=snapshot,
                result=result,
                action="ROOT_CAUSE_ANALYSIS_SKIPPED",
                expected_agent_version=expected_agent_version,
            )
            return result

        findings = build_findings_payload(trend, anomalies)
        notes: list[str] = []

        # 3. Generate hypotheses from the evidence already in hand
        llm_output = await self._request_hypotheses(
            message=message,
            sql=sql,
            query_result=query_result,
            findings=findings,
        )

        # 4. Gather additional evidence when the model asked for it, then re-rank
        questions = [
            item.investigation_question
            for item in llm_output.hypotheses
            if item.investigation_question
        ][:max_investigation_queries]
        evidence: list[RootCauseEvidence] = []
        if questions:
            evidence = await self._investigate(
                questions,
                session_id=session_id,
                workspace_id=workspace_id,
                user_id=user_id,
                organization_id=organization_id,
                metadata=metadata,
                snapshot=snapshot,
                notes=notes,
            )
        if evidence:
            llm_output = await self._request_hypotheses(
                message=message,
                sql=sql,
                query_result=query_result,
                findings=findings,
                evidence=evidence_prompt_payload(evidence),
            )

        # 5. Ranking is deterministic: strongest confidence first, LLM order as tiebreak
        hypotheses = _rank_hypotheses(llm_output)
        top = hypotheses[0]
        result = RootCauseAnalysisResult(
            summary=llm_output.summary,
            primary_cause=top.statement,
            hypotheses=hypotheses,
            evidence=evidence,
            additional_queries_run=sum(1 for item in evidence if item.executed),
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
            action="ROOT_CAUSE_ANALYSIS_COMPLETED",
            expected_agent_version=expected_agent_version,
        )
        return result

    async def _request_hypotheses(
        self,
        *,
        message: str,
        sql: str | None,
        query_result: SQLExecutionResult | None,
        findings: str,
        evidence: str | None = None,
    ) -> LLMRootCauseOutput:
        variables = {
            "message": message,
            "sql": sql or "None",
            "columns": (
                ", ".join(query_result.columns)
                if query_result and query_result.columns
                else "[]"
            ),
            "row_count": str(query_result.row_count) if query_result else "0",
            "truncated": "true" if query_result and query_result.truncated else "false",
            "findings": findings,
        }
        if evidence is not None:
            variables["evidence"] = evidence

        bundle = self._prompt_registry.build_bundle(
            bundle_version=ROOT_CAUSE_ANALYSIS_BUNDLE_VERSION,
            system_prompt_id=ROOT_CAUSE_ANALYSIS_SYSTEM_PROMPT_ID,
            user_template_id=ROOT_CAUSE_ANALYSIS_USER_TEMPLATE_ID,
            user_variables=RootCauseAnalysisVariables(**variables),
            structured_output=structured_output_from_model(LLMRootCauseOutput),
        )
        request = LLMRequest(
            model=self._llm_client.config.model,
            messages=bundle_to_llm_messages(bundle),
            structured_output=bundle.structured_output,
        )

        try:
            response = await self._llm_client.complete(request)
        except Exception as exc:
            raise RootCauseAnalysisLLMError(str(exc)) from exc

        if not response.content:
            raise RootCauseAnalysisLLMError("Root cause analysis response was empty")

        try:
            payload = extract_json_object(response.content)
            return LLMRootCauseOutput.model_validate(payload)
        except Exception as exc:
            raise RootCauseAnalysisValidationError(
                f"Root cause analysis response failed validation: {exc}"
            ) from exc

    async def _investigate(
        self,
        questions: Sequence[str],
        *,
        session_id: UUID,
        workspace_id: UUID,
        user_id: UUID,
        organization_id: UUID | None,
        metadata: ResolvedMetadataContext | None,
        snapshot: AnalysisSessionSnapshot,
        notes: list[str],
    ) -> list[RootCauseEvidence]:
        data_source_id = snapshot.data_source_id
        if metadata is None or organization_id is None or data_source_id is None:
            notes.append(
                "Additional investigation queries were requested but not run: "
                "no authorized data source metadata was supplied."
            )
            return []
        if metadata.data_source_id != data_source_id:
            raise RootCauseAnalysisAuthorizationError(
                "Metadata context does not match the analysis session data source"
            )
        if snapshot.organization_id != organization_id:
            raise RootCauseAnalysisAuthorizationError(
                "Analysis session organization does not match"
            )

        generation = self._sql_generation_service or SQLGenerationService(
            self._session,
            llm_client=self._llm_client,
            state_service=self._state_service,
        )
        execution = self._sql_execution_service or SQLExecutionService(
            self._session,
            state_service=self._state_service,
        )
        evidence = await gather_evidence(
            generation=generation,
            execution=execution,
            questions=questions,
            workspace_id=workspace_id,
            organization_id=organization_id,
            user_id=user_id,
            data_source_id=data_source_id,
            metadata=metadata,
            session_id=session_id,
        )
        notes.extend(
            f"Investigation query for {item.question!r} produced no evidence: {item.note}"
            for item in evidence
            if not item.executed and item.note
        )
        return evidence

    def _record_state(
        self,
        session_id: UUID,
        *,
        workspace_id: UUID,
        user_id: UUID,
        snapshot: AnalysisSessionSnapshot,
        result: RootCauseAnalysisResult,
        action: str,
        expected_agent_version: int | None,
    ) -> None:
        if expected_agent_version is None:
            return
        custom = dict(snapshot.agent_state.payload.custom) if snapshot.agent_state else {}
        custom["root_cause_action"] = action
        custom["root_cause_hypothesis_count"] = str(len(result.hypotheses))
        custom["root_cause_confidence_score"] = result.confidence_score.value
        custom["root_cause_queries_run"] = str(result.additional_queries_run)
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
            raise RootCauseAnalysisAuthorizationError("Data source is not accessible")


def _rank_hypotheses(output: LLMRootCauseOutput) -> list[RootCauseHypothesis]:
    ordered = sorted(
        enumerate(output.hypotheses),
        key=lambda pair: (-confidence_rank(pair[1].confidence_score), pair[0]),
    )
    return [
        RootCauseHypothesis(
            rank=rank,
            statement=item.statement,
            contributing_factors=item.contributing_factors,
            supporting_evidence=item.supporting_evidence,
            contradicting_evidence=item.contradicting_evidence,
            confidence_score=item.confidence_score,
            confidence_reasoning=item.confidence_reasoning,
            investigation_question=item.investigation_question,
        )
        for rank, (_, item) in enumerate(ordered, start=1)
    ]


def _no_finding_result() -> RootCauseAnalysisResult:
    return RootCauseAnalysisResult(
        summary=_NO_FINDING_SUMMARY,
        primary_cause=None,
        hypotheses=[],
        confidence_score=AnalysisConfidence.LOW,
        confidence_reasoning=(
            "No detected trend change or anomaly was supplied, so no cause could be assessed."
        ),
    )
