"""Execute a planned investigation via validation + MCP SQL execution."""

from __future__ import annotations

import logging
import time
from uuid import UUID

from sqlalchemy.orm import Session

from app.ai.investigation.contributors import format_contributions, rank_contributors
from app.ai.investigation.models import DimensionContribution, InvestigationRunResult
from app.ai.investigation.plan import build_investigation_plan, investigation_plan_summary
from app.ai.metadata_types import ResolvedMetadataContext
from app.ai.root_cause_analysis.models import RootCauseEvidence
from app.ai.sql_execution.errors import SQLExecutionError
from app.ai.sql_execution.models import SQLExecuteParams, SQLExecutionStatus
from app.ai.sql_execution.service import SQLExecutionService
from app.ai.sql_validation.validation import validate_generated_sql
from app.core.config import settings

logger = logging.getLogger(__name__)

MAX_PROMPT_ROWS = 20


async def run_planned_investigation(
    *,
    session: Session,
    execution: SQLExecutionService,
    message: str,
    metadata: ResolvedMetadataContext,
    workspace_id: UUID,
    organization_id: UUID,
    user_id: UUID,
    data_source_id: UUID,
    session_id: UUID,
    intent=None,
    profile=None,
    deadline: float | None = None,
) -> InvestigationRunResult:
    """Run schema-grounded investigation SQL; no LLM calls in this path."""
    del session
    steps = build_investigation_plan(
        message,
        metadata,
        intent=intent,
        profile=profile,
        max_steps=settings.AI_INVESTIGATION_MAX_QUERIES,
    )
    if not steps:
        return InvestigationRunResult(steps_planned=0)

    evidence: list[RootCauseEvidence] = []
    contributions: list[DimensionContribution] = []
    notes: list[str] = []
    executed_count = 0

    for step in steps:
        if deadline is not None and time.monotonic() >= deadline:
            notes.append("Planned investigation stopped: analysis time budget exhausted.")
            break
        validation = validate_generated_sql(step.sql, metadata)
        if not validation.is_valid:
            codes = ", ".join(item.code.value for item in validation.violations[:3])
            evidence.append(
                RootCauseEvidence(
                    question=step.question,
                    sql=step.sql,
                    note=f"Investigation SQL failed validation ({codes}).",
                )
            )
            continue
        try:
            outcome = await execution.execute(
                SQLExecuteParams(
                    workspace_id=workspace_id,
                    organization_id=organization_id,
                    user_id=user_id,
                    data_source_id=data_source_id,
                    sql=step.sql,
                    metadata=metadata,
                    limit=min(settings.AI_MAX_RESULT_LIMIT, 500),
                    session_id=session_id,
                )
            )
        except SQLExecutionError as exc:
            evidence.append(
                RootCauseEvidence(question=step.question, sql=step.sql, note=str(exc))
            )
            continue

        result = outcome.result
        if result.status is not SQLExecutionStatus.SUCCEEDED:
            evidence.append(
                RootCauseEvidence(
                    question=step.question,
                    sql=step.sql,
                    note=f"Investigation query status {result.status.value}.",
                )
            )
            continue

        executed_count += 1
        logger.info(
            "investigation step executed kind=%s rows=%s session_id=%s",
            step.kind.value,
            result.row_count,
            session_id,
        )
        evidence.append(
            RootCauseEvidence(
                question=step.question,
                sql=step.sql,
                executed=True,
                columns=list(result.columns),
                rows=[list(row) for row in result.rows[:MAX_PROMPT_ROWS]],
                row_count=result.row_count,
                truncated=result.truncated or result.row_count > MAX_PROMPT_ROWS,
                note=None if result.row_count else "Query returned no rows.",
            )
        )
        if step.kind.value == "breakdown" and result.columns:
            dim_col = _breakdown_dimension_column(step.description, result.columns)
            if dim_col:
                contributions.extend(
                    rank_contributors(
                        columns=list(result.columns),
                        rows=[list(row) for row in result.rows],
                        dimension_column=dim_col,
                    )
                )

    contributions.sort(key=lambda item: abs(item.delta), reverse=True)
    summary_parts = [investigation_plan_summary(steps)]
    contrib_text = format_contributions(contributions)
    if contrib_text:
        summary_parts.append(contrib_text)
    summary_parts.append(
        "Interpretation: segments that moved with the overall change may correlate with it; "
        "they are not proven causes without further evidence."
    )
    return InvestigationRunResult(
        evidence=evidence,
        contributions=contributions[:10],
        summary="\n\n".join(part for part in summary_parts if part),
        notes=notes,
        steps_planned=len(steps),
        steps_executed=executed_count,
    )


def _breakdown_dimension_column(description: str, columns: list[str]) -> str | None:
    lowered = description.lower()
    for column in columns:
        name = column.lower()
        if name in {"period", "month", "week", "day"} or "period" in name:
            continue
        if name in lowered:
            return column
    for column in columns:
        name = column.lower()
        if name not in {"period", "month", "week", "day"} and "period" not in name:
            return column
    return None
