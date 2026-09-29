"""Additional read-only evidence queries for root cause hypotheses.

Every query goes through the existing Phase 7 generation, validation, and MCP
execution pipeline, so authorization, read-only enforcement, and workspace
isolation are not re-implemented here.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from uuid import UUID

from app.ai.metadata_types import ResolvedMetadataContext
from app.ai.root_cause_analysis.models import RootCauseEvidence
from app.ai.sql_execution.errors import SQLExecutionError
from app.ai.sql_execution.models import SQLExecuteParams, SQLExecutionStatus
from app.ai.sql_execution.service import SQLExecutionService
from app.ai.sql_generation.errors import SQLGenerationError
from app.ai.sql_generation.models import SQLGenerateParams
from app.ai.sql_generation.service import SQLGenerationService

logger = logging.getLogger(__name__)

EVIDENCE_ROW_LIMIT = 100
MAX_EVIDENCE_PROMPT_ROWS = 20

_PLAN_SUMMARY = (
    "Root cause investigation: gather supporting or contradicting evidence for a "
    "candidate cause with a single read-only aggregate query."
)


async def gather_evidence(
    *,
    generation: SQLGenerationService,
    execution: SQLExecutionService,
    questions: Sequence[str],
    workspace_id: UUID,
    organization_id: UUID,
    user_id: UUID,
    data_source_id: UUID,
    metadata: ResolvedMetadataContext,
    session_id: UUID,
    row_limit: int = EVIDENCE_ROW_LIMIT,
) -> list[RootCauseEvidence]:
    """Run one investigation query per question, degrading to a note on failure."""
    gathered: list[RootCauseEvidence] = []
    for question in questions:
        try:
            generated = await generation.generate(
                SQLGenerateParams(
                    workspace_id=workspace_id,
                    organization_id=organization_id,
                    user_id=user_id,
                    data_source_id=data_source_id,
                    message=question,
                    metadata=metadata,
                    plan_summary=_PLAN_SUMMARY,
                    session_id=session_id,
                )
            )
        except SQLGenerationError as exc:
            gathered.append(RootCauseEvidence(question=question, note=str(exc)))
            continue

        outcome = generated.outcome
        if outcome.generated is None:
            gathered.append(
                RootCauseEvidence(
                    question=question,
                    note=outcome.clarification_question
                    or "No query could be generated for this question.",
                )
            )
            continue

        sql = outcome.generated.sql
        try:
            executed = await execution.execute(
                SQLExecuteParams(
                    workspace_id=workspace_id,
                    organization_id=organization_id,
                    user_id=user_id,
                    data_source_id=data_source_id,
                    sql=sql,
                    metadata=metadata,
                    limit=row_limit,
                    session_id=session_id,
                )
            )
        except SQLExecutionError as exc:
            gathered.append(
                RootCauseEvidence(question=question, sql=sql, note=str(exc))
            )
            continue

        result = executed.result
        if result.status is not SQLExecutionStatus.SUCCEEDED:
            gathered.append(
                RootCauseEvidence(
                    question=question,
                    sql=sql,
                    note=f"Query did not succeed (status {result.status.value}).",
                )
            )
            continue

        logger.info(
            "root_cause_analysis.evidence_query_executed",
            extra={
                "session_id": str(session_id),
                "data_source_id": str(data_source_id),
                "row_count": result.row_count,
            },
        )
        gathered.append(
            RootCauseEvidence(
                question=question,
                sql=sql,
                executed=True,
                columns=list(result.columns),
                rows=[list(row) for row in result.rows[:MAX_EVIDENCE_PROMPT_ROWS]],
                row_count=result.row_count,
                truncated=result.truncated
                or result.row_count > MAX_EVIDENCE_PROMPT_ROWS,
                note=None if result.row_count else "Query returned no rows.",
            )
        )
    return gathered


def evidence_prompt_payload(evidence: Sequence[RootCauseEvidence]) -> str:
    """Render gathered evidence as JSON for the prompt."""
    return json.dumps(
        [item.model_dump(mode="json") for item in evidence], indent=2, default=str
    )
