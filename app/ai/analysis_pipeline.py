"""SQL execution + Phase 8 analysis chain for supervised chat.

Caller-owned: authorize via existing SQL/agent services, never invents rows,
and skips Phase 8 when execution did not succeed.

Flow: profile the question → generate SQL from a structured, schema-grounded
spec → validate → execute (one correction round, then an optional
catalog-only fallback) → deterministic facts → only the Phase 8 agents the
question needs (independent ones concurrently) → answer that separates
measured facts, hypotheses, and recommendations.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.ai.analysis_profile import (
    AnalysisProfile,
    QuestionKind,
    build_analysis_profile,
    build_analysis_spec,
)
from app.ai.anomaly_detection import AnomalyDetectionAgent
from app.ai.data_analyst import DataAnalystAgent
from app.ai.evaluation import evaluate_agents, evidence_numbers, ungrounded_numbers
from app.ai.insight import InsightAgent
from app.ai.intent_types import (
    AggregationType,
    AIIntent,
    AIRequestPlan,
    TimeRangePreset,
)
from app.ai.metadata_types import (
    MetadataColumnCandidate,
    ResolvedMetadataContext,
    is_identifier_column,
)
from app.ai.recommendation import RecommendationAgent
from app.ai.result_facts import ResultFacts, build_result_facts
from app.ai.root_cause_analysis import RootCauseAnalysisAgent
from app.ai.sql_correction import (
    SQLCorrectionService,
    SQLCorrectionStatus,
    SQLCorrectParams,
)
from app.ai.sql_execution import (
    SQLExecuteParams,
    SQLExecutionError,
    SQLExecutionService,
    SQLExecutionStatus,
    SQLExecutionValidationError,
)
from app.ai.sql_execution.models import SQLExecutionResult
from app.ai.sql_generation import SQLGenerateParams, SQLGenerationService
from app.ai.sql_validation import SQLValidateParams, SQLValidationService
from app.ai.trend_analysis import TrendAnalysisAgent
from app.ai.trend_analysis.models import TrendDirection
from app.core.config import settings
from app.mcp import build_postgres_mcp
from app.services.query_history import QueryHistoryService

logger = logging.getLogger(__name__)

_PREVIEW_MAX_ROWS = 5
_MAX_LISTED_ITEMS = 3
_MAX_SQL_IN_ANSWER = 240
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CHART_BY_SHAPE = {
    "time_series": "line",
    "categorical": "bar",
    "scalar": "kpi",
}


@dataclass(frozen=True)
class QueryPreview:
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    sample_rows: list[list[object]] = field(default_factory=list)


@dataclass(frozen=True)
class AnalysisPayload:
    """Structured Phase 8 outputs for the chat API (omit empty via exclude_none)."""

    session_id: UUID | None = None
    data_analyst: dict[str, Any] | None = None
    trend: dict[str, Any] | None = None
    anomaly: dict[str, Any] | None = None
    root_cause: dict[str, Any] | None = None
    insight: dict[str, Any] | None = None
    recommendation: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    sql: str | None = None
    query_preview: QueryPreview | None = None
    question_types: list[str] | None = None
    facts: list[str] | None = None
    notes: list[str] | None = None
    chart_hint: str | None = None
    question_understood: str | None = None
    date_range: str | None = None
    assumptions: list[str] | None = None
    performance: dict[str, Any] | None = None


@dataclass(frozen=True)
class AnalysisPipelineResult:
    answer: str
    payload: AnalysisPayload | None = None
    requires_clarification: bool = False
    clarification_question: str | None = None
    failed: bool = False


@dataclass(frozen=True)
class AnalysisPipelineParams:
    db: Session
    session_id: UUID
    workspace_id: UUID
    organization_id: UUID
    user_id: UUID
    data_source_id: UUID
    message: str
    metadata: ResolvedMetadataContext
    intent: AIIntent
    plan: AIRequestPlan
    data_source_name: str | None = None
    plan_summary: str | None = None
    expected_agent_version: int | None = None
    run_evaluation: bool = True
    conversation_context: str | None = None
    on_partial: Callable[[dict[str, Any]], Awaitable[None]] | None = None


def should_run_analysis(
    *,
    plan: AIRequestPlan,
    intent: AIIntent,
    metadata: ResolvedMetadataContext,
) -> bool:
    """True when chat should generate/execute SQL and run Phase 8.

    The orchestrator's clarification policy clears soft planner/intent questions
    before this gate once the catalog matched; any clarification still on the
    plan is genuine (no metric, unmatched or ambiguous metric) and blocks SQL.
    """
    del intent
    if plan.unsupported or not plan.requires_database or plan.requires_clarification:
        return False
    return has_sufficient_metadata(metadata)


def has_sufficient_metadata(metadata: ResolvedMetadataContext) -> bool:
    """Catalog already matched tables/columns (or resolved metrics) for SQL."""
    if not metadata.tables:
        return False
    if metadata.resolved_metrics and any(m.resolved for m in metadata.resolved_metrics):
        return True
    return bool(metadata.columns)


# --------------------------------------------------------------------- pipeline


@dataclass
class _SQLContext:
    params: AnalysisPipelineParams
    auth: dict
    limit: int | None
    plan_summary: str
    history: QueryHistoryService
    generation: SQLGenerationService
    validation: SQLValidationService
    execution: SQLExecutionService
    correction: SQLCorrectionService
    mcp_client: Any
    profile: AnalysisProfile


@dataclass(frozen=True)
class _SQLRun:
    sql: str | None = None
    result: SQLExecutionResult | None = None
    clarification: str | None = None
    failure: str | None = None
    notes: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()


async def run_analysis_pipeline(params: AnalysisPipelineParams) -> AnalysisPipelineResult:
    """Generate → validate → execute (+ correct) → facts → Phase 8 → answer."""
    started = time.perf_counter()
    stages: list[dict[str, Any]] = []
    profile = build_analysis_profile(params.message, intent=params.intent, plan=params.plan)
    from app.ai.investigation.plan import build_investigation_plan, investigation_plan_summary

    inv_steps = build_investigation_plan(
        params.message,
        params.metadata,
        intent=params.intent,
        profile=profile,
    )
    spec = build_analysis_spec(
        intent=params.intent,
        plan=params.plan,
        metadata=params.metadata,
        profile=profile,
    )
    inv_summary = investigation_plan_summary(inv_steps)
    if inv_summary:
        spec = f"{spec}\n\n{inv_summary}"
    mcp_registry, mcp_client = build_postgres_mcp(params.db)
    ctx = _SQLContext(
        params=params,
        auth=dict(
            workspace_id=params.workspace_id,
            organization_id=params.organization_id,
            user_id=params.user_id,
            data_source_id=params.data_source_id,
            session_id=params.session_id,
        ),
        limit=params.intent.safe_limit,
        plan_summary=spec,
        history=QueryHistoryService(params.db),
        generation=SQLGenerationService(params.db),
        validation=SQLValidationService(params.db),
        execution=SQLExecutionService(
            params.db, mcp_client=mcp_client, mcp_registry=mcp_registry
        ),
        correction=SQLCorrectionService(params.db),
        mcp_client=mcp_client,
        profile=profile,
    )
    logger.info(
        "chat analysis profile session_id=%s kinds=%s",
        params.session_id,
        ",".join(profile.labels()),
    )

    sql_started = time.perf_counter()
    run = await _resolve_sql(ctx)
    stages.append(
        {
            "stage": "sql",
            "duration_ms": round((time.perf_counter() - sql_started) * 1000, 1),
            "agents_skipped": not profile.uses_llm_agents,
        }
    )
    if run.clarification is not None:
        return AnalysisPipelineResult(
            answer=run.clarification,
            requires_clarification=True,
            clarification_question=run.clarification,
        )
    if run.result is None or run.sql is None:
        return AnalysisPipelineResult(
            answer=run.failure or "The query could not be executed against this data source.",
            failed=True,
        )

    facts = build_result_facts(run.result, message=params.message)
    early_payload = AnalysisPayload(
        session_id=params.session_id,
        sql=run.sql,
        query_preview=_preview(run.result),
        question_types=profile.labels(),
        facts=[line for line in [facts.headline, *facts.lines] if line] or None,
        chart_hint=_chart_hint(facts),
        question_understood=_question_understood(params.message, params.intent, profile),
        date_range=_date_range_label(params.intent),
        assumptions=list(run.assumptions) or None,
        notes=list(run.notes) or None,
        performance={"stages": list(stages)},
    )
    early_answer = _compose_answer(early_payload, run.result, facts=facts)
    if params.on_partial is not None:
        try:
            await params.on_partial(
                {
                    "stage": "facts",
                    "answer": early_answer,
                    "sql": run.sql,
                    "facts": early_payload.facts,
                    "chart_hint": early_payload.chart_hint,
                    "query_preview": {
                        "columns": early_payload.query_preview.columns,
                        "row_count": early_payload.query_preview.row_count,
                        "truncated": early_payload.query_preview.truncated,
                        "sample_rows": early_payload.query_preview.sample_rows,
                    }
                    if early_payload.query_preview
                    else None,
                    "performance": early_payload.performance,
                }
            )
        except Exception as exc:  # noqa: BLE001 — streaming must not fail the answer
            logger.warning("partial analysis callback failed error=%s", type(exc).__name__)

    phase8_started = time.perf_counter()
    payload = await _run_phase8(
        db=params.db,
        session_id=params.session_id,
        workspace_id=params.workspace_id,
        user_id=params.user_id,
        organization_id=params.organization_id,
        message=params.message,
        sql=run.sql,
        query_result=run.result,
        metadata=params.metadata,
        expected_agent_version=params.expected_agent_version,
        intent=params.intent,
        run_evaluation=params.run_evaluation,
        profile=profile,
        facts=facts,
        execution=ctx.execution,
        extra_notes=list(run.notes),
    )
    stages.append(
        {
            "stage": "phase8",
            "duration_ms": round((time.perf_counter() - phase8_started) * 1000, 1),
            "used_llm_agents": profile.uses_llm_agents,
        }
    )
    stages.append(
        {
            "stage": "total",
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    )
    assumptions = list(run.assumptions) or None
    payload = replace(
        payload,
        question_understood=_question_understood(params.message, params.intent, profile),
        date_range=_date_range_label(params.intent),
        assumptions=assumptions,
        chart_hint=payload.chart_hint or _chart_hint(facts),
        performance={"stages": stages},
    )
    answer = _compose_answer(payload, run.result, facts=facts)
    logger.info(
        "chat analysis performance session_id=%s stages=%s",
        params.session_id,
        stages,
    )
    return AnalysisPipelineResult(answer=answer, payload=payload)


async def _resolve_sql(ctx: _SQLContext) -> _SQLRun:
    params = ctx.params
    # Prefer single-grain catalog SQL for period comparisons/trends. LLM drafts
    # often use GROUPING SETS / multi-section unions that double-count in facts
    # and break series-shaped result matching.
    if settings.AI_CHAT_DETERMINISTIC_SQL_FALLBACK and _prefer_deterministic_sql(ctx):
        preferred = await _rescue_with_deterministic_sql(ctx)
        if preferred is not None:
            return _SQLRun(
                sql=preferred.sql,
                result=preferred.result,
                assumptions=preferred.assumptions,
                notes=(
                    "Used a catalog-grounded period query so totals stay single-grain "
                    "and directly comparable.",
                ),
            )
    try:
        outcome = await _generate(ctx, params.message)
    except Exception as exc:  # noqa: BLE001 — surface as safe chat failure
        logger.warning(
            "sql generation failed session_id=%s error=%s",
            params.session_id,
            type(exc).__name__,
        )
        return await _fallback_or_fail(
            ctx, "I could not generate a safe query for that request."
        )

    # One soft retry when the catalog clearly matches and the model only asked
    # for defaults (period/breakdown); saves a clarification round-trip.
    if outcome.requires_clarification and _soft_clarification_retry_allowed(ctx):
        try:
            outcome = await _generate(
                ctx,
                f"{params.message}\n\n(If a detail is missing, assume: the most recent "
                "complete period present in the data, the previous period of the same "
                "length as comparison, and state the assumption.)",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "sql generation soft retry failed error=%s", type(exc).__name__
            )
    if outcome.requires_clarification:
        return _SQLRun(
            clarification=outcome.clarification_question
            or "Could you clarify what you would like to analyze?"
        )
    if outcome.generated is None:
        return await _fallback_or_fail(
            ctx, "I could not generate a safe query for that request."
        )
    return await _validate_execute_or_repair(
        ctx,
        outcome.generated.sql,
        assumptions=tuple(outcome.generated.assumptions or ()),
    )


async def _generate(ctx: _SQLContext, message: str):
    params = ctx.params
    generated = await ctx.generation.generate(
        SQLGenerateParams(
            **ctx.auth,
            message=message,
            metadata=params.metadata,
            intent=params.intent,
            plan_summary=ctx.plan_summary,
            data_source_name=params.data_source_name,
            conversation_context=params.conversation_context,
        )
    )
    return generated.outcome


def _prefer_deterministic_sql(ctx: _SQLContext) -> bool:
    kinds = ctx.profile.kinds
    if not kinds & {QuestionKind.COMPARISON, QuestionKind.TREND}:
        return False
    # Keep LLM path for diagnostic/anomaly/recommendation narratives that need
    # richer breakdown SQL; facts grounding handles region×period separately.
    if kinds & {
        QuestionKind.DIAGNOSTIC,
        QuestionKind.ANOMALY,
        QuestionKind.RECOMMENDATION,
        QuestionKind.INSIGHT,
    }:
        return False
    return True


def _soft_clarification_retry_allowed(ctx: _SQLContext) -> bool:
    metadata = ctx.params.metadata
    if not has_sufficient_metadata(metadata) or metadata.unresolved_concepts:
        return False
    return bool(
        ctx.profile.kinds
        & {
            QuestionKind.DIAGNOSTIC,
            QuestionKind.TREND,
            QuestionKind.COMPARISON,
            QuestionKind.ANOMALY,
        }
    )


async def _validate_execute_or_repair(
    ctx: _SQLContext,
    draft_sql: str,
    *,
    assumptions: tuple[str, ...] = (),
) -> _SQLRun:
    params = ctx.params
    validated = ctx.validation.validate(
        SQLValidateParams(**ctx.auth, sql=draft_sql, metadata=params.metadata)
    )
    violations: tuple = ()
    execution_error_code: str | None = None
    execution_error_message: str | None = None

    if validated.result.is_valid and validated.result.validated is not None:
        sql = validated.result.validated.sql
        try:
            executed = await ctx.execution.execute(
                SQLExecuteParams(**ctx.auth, sql=sql, metadata=params.metadata, limit=ctx.limit)
            )
        except SQLExecutionValidationError as exc:
            violations = tuple(exc.violations)
            _record_validation(ctx, draft_sql, validated.result)
        except SQLExecutionError as exc:
            execution_error_code = exc.code.value
            execution_error_message = str(exc)
        else:
            _record_execution(ctx, draft_sql, sql, executed.result)
            if executed.result.status is SQLExecutionStatus.SUCCEEDED:
                return _SQLRun(sql=sql, result=executed.result, assumptions=assumptions)
            execution_error_code = executed.result.status.value
    else:
        violations = tuple(validated.result.violations)
        _record_validation(ctx, draft_sql, validated.result)

    corr, exec_result = await _correct_and_maybe_execute(
        correction=ctx.correction,
        mcp_client=ctx.mcp_client,
        auth=ctx.auth,
        sql=draft_sql,
        metadata=params.metadata,
        message=params.message,
        intent=params.intent,
        plan_summary=ctx.plan_summary,
        data_source_name=params.data_source_name,
        violations=violations,
        execution_error_code=execution_error_code,
        execution_error_message=execution_error_message,
        limit=ctx.limit,
        history=ctx.history,
        user_id=params.user_id,
        workspace_id=params.workspace_id,
        organization_id=params.organization_id,
        data_source_id=params.data_source_id,
        max_attempts=(
            1
            if not ctx.profile.uses_llm_agents
            else settings.AI_SQL_CORRECTION_MAX_ATTEMPTS
        ),
    )
    if corr.outcome.requires_clarification:
        return _SQLRun(
            clarification=corr.outcome.clarification_question
            or "Could you clarify what you would like to analyze?",
            assumptions=assumptions,
        )
    if (
        corr.outcome.status is SQLCorrectionStatus.CORRECTED
        and exec_result is not None
        and exec_result.status is SQLExecutionStatus.SUCCEEDED
        and corr.outcome.validated is not None
    ):
        return _SQLRun(
            sql=corr.outcome.validated.sql,
            result=exec_result,
            assumptions=assumptions,
        )

    detail = _failure_detail(violations, corr.outcome)
    if execution_error_code and not detail:
        detail = f"Database error: {execution_error_code}."
    logger.warning(
        "sql correction did not yield executable SQL session_id=%s detail=%s",
        params.session_id,
        detail,
    )
    return await _fallback_or_fail(
        ctx,
        f"I could not produce a valid query for that request. {detail}".strip(),
    )


async def _fallback_or_fail(ctx: _SQLContext, failure: str) -> _SQLRun:
    """Catalog-only fallback when it can answer faithfully; otherwise explain and ask."""
    if settings.AI_CHAT_DETERMINISTIC_SQL_FALLBACK:
        rescued = await _rescue_with_deterministic_sql(ctx)
        if rescued is not None:
            return rescued
    hint = _clarification_hint(ctx.params.metadata)
    return _SQLRun(failure=f"{failure} {hint}".strip())


def _record_validation(ctx: _SQLContext, sql: str, validation_result) -> None:
    params = ctx.params
    ctx.history.record_from_validation_result(
        user_id=params.user_id,
        workspace_id=params.workspace_id,
        organization_id=params.organization_id,
        data_source_id=params.data_source_id,
        generated_sql=sql,
        validation=validation_result,
    )


def _record_execution(
    ctx: _SQLContext, generated_sql: str, validated_sql: str, result: SQLExecutionResult
) -> None:
    params = ctx.params
    ctx.history.record_from_execution_result(
        user_id=params.user_id,
        workspace_id=params.workspace_id,
        organization_id=params.organization_id,
        data_source_id=params.data_source_id,
        generated_sql=generated_sql,
        validated_sql=validated_sql,
        result=result,
    )


async def _correct_and_maybe_execute(
    *,
    correction: SQLCorrectionService,
    mcp_client,
    auth: dict,
    sql: str,
    metadata: ResolvedMetadataContext,
    message: str,
    intent: AIIntent,
    plan_summary: str | None,
    data_source_name: str | None,
    history: QueryHistoryService,
    user_id: UUID,
    workspace_id: UUID,
    organization_id: UUID,
    data_source_id: UUID,
    violations: tuple = (),
    execution_error_code: str | None = None,
    execution_error_message: str | None = None,
    limit: int | None,
    max_attempts: int | None = None,
):
    from app.ai.sql_correction.workflow import execute_corrected_sql

    corr_params = SQLCorrectParams(
        **auth,
        sql=sql,
        metadata=metadata,
        message=message,
        intent=intent,
        plan_summary=plan_summary,
        data_source_name=data_source_name,
        violations=violations,
        execution_error_code=execution_error_code,
        execution_error_message=execution_error_message,
        max_attempts=max_attempts or settings.AI_SQL_CORRECTION_MAX_ATTEMPTS,
    )
    corr_result = await correction.correct(corr_params)
    history.record_from_correction_outcome(
        user_id=user_id,
        workspace_id=workspace_id,
        organization_id=organization_id,
        data_source_id=data_source_id,
        generated_sql=sql,
        outcome=corr_result.outcome,
    )
    if (
        corr_result.outcome.status is not SQLCorrectionStatus.CORRECTED
        or corr_result.outcome.validated is None
    ):
        return corr_result, None

    try:
        exec_result = await execute_corrected_sql(
            client=mcp_client,
            params=corr_params,
            outcome=corr_result.outcome,
            limit=limit,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("corrected sql execute failed error=%s", type(exc).__name__)
        return corr_result, None

    history.record_from_execution_result(
        user_id=user_id,
        workspace_id=workspace_id,
        organization_id=organization_id,
        data_source_id=data_source_id,
        generated_sql=sql,
        validated_sql=corr_result.outcome.validated.sql,
        result=exec_result,
    )
    return corr_result, exec_result


# ------------------------------------------------------- deterministic fallback


async def _rescue_with_deterministic_sql(ctx: _SQLContext) -> _SQLRun | None:
    params = ctx.params
    built = _deterministic_query(
        params.message,
        params.metadata,
        intent=params.intent,
        profile=ctx.profile,
    )
    if built is None:
        return None
    sql, description = built
    validated = ctx.validation.validate(
        SQLValidateParams(**ctx.auth, sql=sql, metadata=params.metadata)
    )
    if not validated.result.is_valid or validated.result.validated is None:
        logger.warning(
            "deterministic fallback invalid codes=%s",
            [v.code.value for v in validated.result.violations[:5]],
        )
        return None
    safe_sql = validated.result.validated.sql
    try:
        executed = await ctx.execution.execute(
            SQLExecuteParams(**ctx.auth, sql=safe_sql, metadata=params.metadata, limit=ctx.limit)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("deterministic fallback execute failed error=%s", type(exc).__name__)
        return None
    if executed.result.status is not SQLExecutionStatus.SUCCEEDED:
        return None
    _record_execution(ctx, sql, safe_sql, executed.result)
    logger.info("chat answered via deterministic fallback session_id=%s", params.session_id)
    return _SQLRun(
        sql=safe_sql,
        result=executed.result,
        notes=(
            "The generated query could not be validated, so a simpler query built only "
            f"from catalog columns was used: {description}.",
        ),
    )


_NUMERIC_TYPE_RE = re.compile(r"num|int|dec|float|double|real|money", re.IGNORECASE)
_TIME_TYPE_RE = re.compile(r"date|time", re.IGNORECASE)
_TEXT_TYPE_RE = re.compile(r"char|text|string|enum|bool", re.IGNORECASE)
_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}  # fmt: skip
_GRAIN_CUES = (
    ("day", re.compile(r"\b(daily|per day|by day|each day)\b", re.IGNORECASE)),
    ("week", re.compile(r"\b(weekly|per week|by week|each week)\b", re.IGNORECASE)),
    ("quarter", re.compile(r"\b(quarterly|per quarter|by quarter|each quarter)\b", re.IGNORECASE)),
    ("year", re.compile(r"\b(yearly|annual|annually|per year|by year|each year)\b", re.IGNORECASE)),
)


def _deterministic_query(
    message: str,
    metadata: ResolvedMetadataContext,
    *,
    intent: AIIntent | None = None,
    profile: AnalysisProfile | None = None,
    today: date | None = None,
) -> tuple[str, str] | None:
    """Single-table aggregate built only from resolved catalog columns.

    Returns None rather than guessing whenever the requested metric, dimensions,
    or filters cannot be grounded in the catalog, so it never substitutes data.
    """
    if not metadata.tables:
        return None
    if intent is not None and intent.filters:
        return None  # filters cannot be applied faithfully without the LLM
    metric = _grounded_metric(metadata, intent)
    if metric is None:
        return None
    same_table = [
        c
        for c in _all_columns(metadata)
        if (c.schema_name.lower(), c.table_name.lower())
        == (metric.schema_name.lower(), metric.table_name.lower())
    ]

    requested_dims = [
        cand
        for item in metadata.resolved_dimensions
        if item.resolved and not item.ambiguous
        for cand in item.candidates[:1]
    ]
    if intent is not None and len(requested_dims) < len(intent.dimensions):
        return None
    if any(
        (d.schema_name.lower(), d.table_name.lower())
        != (metric.schema_name.lower(), metric.table_name.lower())
        for d in requested_dims
    ):
        return None
    dims = requested_dims[:2]
    kinds = profile.kinds if profile is not None else frozenset()
    if not dims and QuestionKind.DIAGNOSTIC in kinds:
        breakdown = _breakdown_column(same_table)
        if breakdown is not None:
            dims = [breakdown]

    window = _time_window(message, intent, kinds, today=today or date.today())
    wants_time = bool(
        window is not None
        or kinds & {QuestionKind.TREND, QuestionKind.DIAGNOSTIC, QuestionKind.ANOMALY}
    )
    time_col = _grounded_time_column(metadata, same_table) if wants_time else None
    if window is not None and time_col is None:
        return None

    aggregation = _aggregation(intent, metric)
    metric_expr = "COUNT(*)" if aggregation == "COUNT" else f'{aggregation}("{metric.column_name}")'
    metric_alias = _safe_alias(
        metric.column_name if aggregation in {"SUM", "COUNT"} else f"{aggregation.lower()}_{metric.column_name}"
    )
    table_sql = f'"{metric.schema_name}"."{metric.table_name}"'
    select_parts: list[str] = []
    grain = _grain(message)
    if time_col is not None:
        select_parts.append(f"date_trunc('{grain}', \"{time_col.column_name}\")::date AS period")
    for dim in dims:
        select_parts.append(f'"{dim.column_name}"')
    # Positional GROUP BY/ORDER BY: output aliases could collide with physical names.
    group_parts = [str(position) for position in range(1, len(select_parts) + 1)]
    select_parts.append(f'{metric_expr} AS "{metric_alias}"')

    where = ""
    if window is not None and time_col is not None:
        start, end = window
        where = (
            f' WHERE "{time_col.column_name}" >= DATE \'{start.isoformat()}\''
            f' AND "{time_col.column_name}" < DATE \'{end.isoformat()}\''
        )
    sql = f"SELECT {', '.join(select_parts)} FROM {table_sql}{where}"
    if group_parts:
        sql += f" GROUP BY {', '.join(group_parts)}"
        if time_col is not None:
            order = group_parts
        else:
            order = [f"{len(select_parts)} DESC"]
        sql += f" ORDER BY {', '.join(order)}"
    limit = settings.AI_MAX_RESULT_LIMIT
    if intent is not None and intent.safe_limit:
        limit = min(limit, intent.safe_limit)
    sql += f" LIMIT {limit}"

    description = f"{aggregation} of {metric.table_name}.{metric.column_name}"
    if time_col is not None:
        description += f" by {grain} of {time_col.column_name}"
    if dims:
        description += " and " + ", ".join(d.column_name for d in dims)
    if window is not None:
        description += f" from {window[0].isoformat()} to {(window[1] - timedelta(days=1)).isoformat()}"
    return sql, description


def _deterministic_change_sql(message: str, metadata: ResolvedMetadataContext) -> str | None:
    """Backward-compatible wrapper used by tests and diagnostics."""
    built = _deterministic_query(
        message,
        metadata,
        profile=build_analysis_profile(message),
    )
    return built[0] if built else None


def _all_columns(metadata: ResolvedMetadataContext) -> list[MetadataColumnCandidate]:
    seen: set[tuple[str, str, str]] = set()
    out: list[MetadataColumnCandidate] = []
    groups: list[list[MetadataColumnCandidate]] = [
        list(metadata.columns),
        list(metadata.resolved_time_columns),
    ]
    for concept_group in (
        metadata.resolved_metrics,
        metadata.resolved_dimensions,
        metadata.resolved_filters,
    ):
        for item in concept_group:
            groups.append(list(item.candidates))
    for group in groups:
        for column in group:
            key = (
                column.schema_name.lower(),
                column.table_name.lower(),
                column.column_name.lower(),
            )
            if key not in seen:
                seen.add(key)
                out.append(column)
    return out


def _grounded_metric(
    metadata: ResolvedMetadataContext, intent: AIIntent | None
) -> MetadataColumnCandidate | None:
    for item in metadata.resolved_metrics:
        if not item.resolved or item.ambiguous:
            continue
        numeric = [c for c in item.candidates if _NUMERIC_TYPE_RE.search(c.data_type or "")]
        if numeric:
            return numeric[0]
    if intent is None or not intent.metrics:
        return None
    columns = [c for c in _all_columns(metadata) if _NUMERIC_TYPE_RE.search(c.data_type or "")]
    for metric in intent.metrics:
        wanted = _tokens(metric.name)
        exact = [c for c in columns if _tokens(c.column_name) == wanted]
        if len(exact) == 1:
            return exact[0]
    return None


def _grounded_time_column(
    metadata: ResolvedMetadataContext, same_table: list[MetadataColumnCandidate]
) -> MetadataColumnCandidate | None:
    table_keys = {(c.schema_name.lower(), c.table_name.lower()) for c in same_table}
    resolved = [
        c
        for c in metadata.resolved_time_columns
        if (c.schema_name.lower(), c.table_name.lower()) in table_keys
    ]
    if len(resolved) == 1:
        return resolved[0]
    typed = [c for c in same_table if _TIME_TYPE_RE.search(c.data_type or "")]
    if len(typed) == 1:
        return typed[0]
    return None  # zero or several candidates: do not guess


def _breakdown_column(columns: list[MetadataColumnCandidate]) -> MetadataColumnCandidate | None:
    for column in columns:
        name = column.column_name.lower()
        if column.is_primary_key or name == "id" or name.endswith("_id"):
            continue
        if _TEXT_TYPE_RE.search(column.data_type or ""):
            return column
    return None


def _aggregation(intent: AIIntent | None, metric: MetadataColumnCandidate) -> str:
    if is_identifier_column(metric):
        return "COUNT"
    if intent is not None:
        for item in intent.metrics:
            agg = item.aggregation
            if agg in {AggregationType.SUM, AggregationType.AVG, AggregationType.MIN, AggregationType.MAX}:
                return agg.value
            if agg in {AggregationType.COUNT, AggregationType.COUNT_DISTINCT}:
                return "COUNT"
    return "SUM" if _NUMERIC_TYPE_RE.search(metric.data_type or "") else "COUNT"


def _grain(message: str) -> str:
    for grain, pattern in _GRAIN_CUES:
        if pattern.search(message or ""):
            return grain
    return "month"


def _time_window(
    message: str,
    intent: AIIntent | None,
    kinds: frozenset,
    *,
    today: date,
) -> tuple[date, date] | None:
    """Half-open [start, end) window from the intent or explicit dates in the message."""
    comparative = bool(kinds & {QuestionKind.DIAGNOSTIC, QuestionKind.COMPARISON})

    def _with_prior(start: date, end: date) -> tuple[date, date]:
        """Extend a comparative window back by one window length (at least a month)."""
        if not comparative:
            return start, end
        months = max(1, (end.year - start.year) * 12 + end.month - start.month)
        return _shift_months(start, -months), end

    if intent is not None and intent.time_range is not None:
        tr = intent.time_range
        if tr.start_date and tr.end_date:
            return _with_prior(tr.start_date, tr.end_date + timedelta(days=1))
        preset = _preset_window(tr.preset, today)
        if preset is not None:
            return _with_prior(*preset)

    text = (message or "").lower()
    year_match = re.search(r"\b((?:19|20|21)\d{2})\b", text)
    month = next(
        (
            number
            for name, number in _MONTH_NAMES.items()
            if re.search(rf"\b{name}\b", text) and (name != "may" or year_match)
        ),
        None,
    )
    quarter = re.search(r"\bq([1-4])\b", text)
    if year_match is None:
        return None
    year = int(year_match.group(1))
    if quarter:
        start = date(year, 3 * int(quarter.group(1)) - 2, 1)
        return _with_prior(start, _shift_months(start, 3))
    if month is not None:
        start = date(year, month, 1)
        return _with_prior(start, _shift_months(start, 1))
    return _with_prior(date(year, 1, 1), date(year + 1, 1, 1))


def _preset_window(preset: TimeRangePreset, today: date) -> tuple[date, date] | None:
    month_start = today.replace(day=1)
    quarter_start = date(today.year, 3 * ((today.month - 1) // 3) + 1, 1)
    windows = {
        TimeRangePreset.TODAY: (today, today + timedelta(days=1)),
        TimeRangePreset.YESTERDAY: (today - timedelta(days=1), today),
        TimeRangePreset.THIS_MONTH: (month_start, _shift_months(month_start, 1)),
        TimeRangePreset.LAST_MONTH: (_shift_months(month_start, -1), month_start),
        TimeRangePreset.THIS_QUARTER: (quarter_start, _shift_months(quarter_start, 3)),
        TimeRangePreset.LAST_QUARTER: (_shift_months(quarter_start, -3), quarter_start),
        TimeRangePreset.THIS_YEAR: (date(today.year, 1, 1), date(today.year + 1, 1, 1)),
        TimeRangePreset.LAST_YEAR: (date(today.year - 1, 1, 1), date(today.year, 1, 1)),
    }
    return windows.get(preset)


def _shift_months(value: date, months: int) -> date:
    index = value.year * 12 + (value.month - 1) + months
    return date(index // 12, index % 12 + 1, 1)


def _tokens(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", value.lower()))


def _safe_alias(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", value).strip("_").lower()
    return cleaned[:60] or "value"


def _clarification_hint(metadata: ResolvedMetadataContext) -> str:
    if metadata.unresolved_concepts:
        return (
            "These terms did not match any column in this data source: "
            + ", ".join(metadata.unresolved_concepts[:5])
            + ". Could you rephrase using the available fields?"
        )
    return (
        "Could you rephrase with the metric, the period, and any grouping you want "
        "(for example: total <metric> by month for <year>)?"
    )


# ---------------------------------------------------------------------- phase 8


async def _run_phase8(
    *,
    db: Session,
    session_id: UUID,
    workspace_id: UUID,
    user_id: UUID,
    organization_id: UUID,
    message: str,
    sql: str,
    query_result: SQLExecutionResult,
    metadata: ResolvedMetadataContext,
    expected_agent_version: int | None,
    run_evaluation: bool,
    intent: AIIntent | None = None,
    profile: AnalysisProfile | None = None,
    facts: ResultFacts | None = None,
    execution: SQLExecutionService | None = None,
    extra_notes: list[str] | None = None,
) -> AnalysisPayload:
    """Run only the agents the question needs; independent agents concurrently.

    Stage 1: data analyst, trend, anomaly (independent).
    Stage 2: root cause (needs trend/anomaly findings).
    Stage 3: insight and recommendation (consume earlier outputs).
    Every agent is optional: failures/timeouts become notes, never errors.
    """
    from app.ai.llm import AsyncLLMClient, llm_client_config_from_settings

    profile = profile or build_analysis_profile(message)
    facts = facts or build_result_facts(query_result, message=message)
    notes: list[str] = list(extra_notes or [])
    base = dict(
        session_id=session_id,
        sql=sql,
        query_preview=_preview(query_result),
        question_types=profile.labels(),
        facts=[line for line in [facts.headline, *facts.lines] if line] or None,
        chart_hint=_chart_hint(facts),
    )

    def _finish(**agents: Any) -> AnalysisPayload:
        evaluation = None
        has_agent_output = any(value is not None for value in agents.values())
        if run_evaluation and has_agent_output:
            try:
                evaluation = evaluate_agents(
                    session_id=session_id,
                    query_result=query_result,
                    analysis=agents.get("analyst"),
                    trend=agents.get("trend"),
                    anomalies=agents.get("anomaly"),
                    root_cause=agents.get("root_cause"),
                    insights=agents.get("insight"),
                    recommendations=agents.get("recommendation"),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("evaluation skipped error=%s", type(exc).__name__)
        dumped = {
            key: value.model_dump(mode="json") if value is not None else None
            for key, value in agents.items()
        }
        return AnalysisPayload(
            **base,
            data_analyst=dumped.get("analyst"),
            trend=dumped.get("trend"),
            anomaly=dumped.get("anomaly"),
            root_cause=dumped.get("root_cause"),
            insight=dumped.get("insight"),
            recommendation=dumped.get("recommendation"),
            evaluation=evaluation.model_dump(mode="json") if evaluation else None,
            notes=notes or None,
        )

    if facts.is_empty or not profile.uses_llm_agents:
        logger.info(
            "phase8 deterministic only session_id=%s empty=%s",
            session_id,
            facts.is_empty,
        )
        return _finish()

    base_cfg = llm_client_config_from_settings()
    if not base_cfg.api_key:
        notes.append("Narrative analysis is unavailable because no LLM is configured.")
        return _finish()

    prompt_rows = settings.AI_CHAT_PHASE8_PROMPT_ROWS
    prompt_result = query_result
    if len(query_result.rows) > prompt_rows:
        prompt_result = query_result.model_copy(
            update={"rows": list(query_result.rows[:prompt_rows]), "truncated": True}
        )

    agent_timeout = settings.AI_CHAT_PHASE8_AGENT_TIMEOUT_SECONDS
    rca_timeout = settings.AI_CHAT_PHASE8_RCA_TIMEOUT_SECONDS

    def _client(timeout: float) -> AsyncLLMClient:
        return AsyncLLMClient(
            replace(
                base_cfg,
                max_tokens=min(base_cfg.max_tokens, settings.AI_CHAT_PHASE8_MAX_TOKENS),
                max_retries=0,
                timeout=min(base_cfg.timeout, timeout),
            )
        )

    llm_client = _client(agent_timeout)
    common = dict(
        session_id=session_id,
        workspace_id=workspace_id,
        user_id=user_id,
        message=message,
        sql=sql,
        query_result=prompt_result,
        expected_agent_version=expected_agent_version,
    )
    deadline = time.monotonic() + settings.AI_CHAT_PHASE8_BUDGET_SECONDS
    readable = {
        "data_analyst": "Data analysis",
        "trend_analysis": "Trend analysis",
        "anomaly_detection": "Anomaly detection",
        "root_cause_analysis": "Root cause analysis",
        "insight": "Insight generation",
        "recommendation": "Recommendations",
    }

    async def _run(label: str, factory, timeout: float):
        remaining = deadline - time.monotonic()
        if remaining <= 1:
            logger.warning("phase8 budget exhausted; skipping %s", label)
            notes.append(f"{readable[label]} was skipped because the analysis time budget ran out.")
            return None
        try:
            return await asyncio.wait_for(factory(), timeout=min(timeout, remaining))
        except asyncio.TimeoutError:
            logger.warning("%s timed out", label)
            notes.append(f"{readable[label]} did not finish in time.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s skipped error=%s", label, type(exc).__name__)
            notes.append(f"{readable[label]} was unavailable for this question.")
        return None

    async def _none():
        return None

    stage1 = await asyncio.gather(
        _run(
            "data_analyst",
            lambda: DataAnalystAgent(db, llm_client=llm_client).analyze(**common),
            agent_timeout,
        )
        if profile.run_data_analyst
        else _none(),
        _run(
            "trend_analysis",
            lambda: TrendAnalysisAgent(db, llm_client=llm_client).analyze(**common),
            agent_timeout,
        )
        if profile.run_trend
        else _none(),
        _run(
            "anomaly_detection",
            lambda: AnomalyDetectionAgent(db, llm_client=llm_client).analyze(**common),
            agent_timeout,
        )
        if profile.run_anomaly
        else _none(),
    )
    analyst, trend, anomaly = stage1

    root_cause = None
    if profile.run_root_cause:
        from app.ai.investigation.runner import run_planned_investigation

        investigation = None
        if execution is not None and settings.AI_INVESTIGATION_MAX_QUERIES > 0:
            investigation = await run_planned_investigation(
                session=db,
                execution=execution,
                message=message,
                metadata=metadata,
                workspace_id=workspace_id,
                organization_id=organization_id,
                user_id=user_id,
                data_source_id=metadata.data_source_id,
                session_id=session_id,
                intent=intent,
                profile=profile,
                deadline=deadline,
            )
            if investigation.notes:
                notes.extend(investigation.notes)
        rca_kwargs: dict[str, Any] = {"llm_client": _client(rca_timeout)}
        if execution is not None:
            rca_kwargs["sql_execution_service"] = execution
        inv = investigation
        root_cause = await _run(
            "root_cause_analysis",
            lambda: RootCauseAnalysisAgent(db, **rca_kwargs).analyze(
                **common,
                trend=trend,
                anomalies=anomaly,
                organization_id=organization_id,
                metadata=metadata,
                max_investigation_queries=settings.AI_CHAT_PHASE8_RCA_MAX_INVESTIGATION_QUERIES,
                prefetched_evidence=inv.evidence if inv else None,
                investigation_summary=inv.summary if inv else None,
            ),
            rca_timeout,
        )

    insight, recommendation = await asyncio.gather(
        _run(
            "insight",
            lambda: InsightAgent(db, llm_client=llm_client).generate(
                session_id=session_id,
                workspace_id=workspace_id,
                user_id=user_id,
                message=message,
                sql=sql,
                query_result=prompt_result,
                analysis=analyst,
                trend=trend,
                anomalies=anomaly,
                root_cause=root_cause,
                expected_agent_version=expected_agent_version,
            ),
            agent_timeout,
        )
        if profile.run_insight
        else _none(),
        _run(
            "recommendation",
            lambda: RecommendationAgent(db, llm_client=llm_client).recommend(
                session_id=session_id,
                workspace_id=workspace_id,
                user_id=user_id,
                message=message,
                sql=sql,
                query_result=prompt_result,
                analysis=analyst,
                trend=trend,
                anomalies=anomaly,
                root_cause=root_cause,
                expected_agent_version=expected_agent_version,
            ),
            agent_timeout,
        )
        if profile.run_recommendation
        else _none(),
    )
    return _finish(
        analyst=analyst,
        trend=trend,
        anomaly=anomaly,
        root_cause=root_cause,
        insight=insight,
        recommendation=recommendation,
    )


def _wants_full_phase8(message: str) -> bool:
    """True when the question needs narrative agents beyond deterministic facts."""
    return build_analysis_profile(message).uses_llm_agents


def _preview(result: SQLExecutionResult) -> QueryPreview:
    rows = list(result.rows[:_PREVIEW_MAX_ROWS])
    return QueryPreview(
        columns=list(result.columns),
        row_count=result.row_count,
        truncated=result.truncated or len(result.rows) > _PREVIEW_MAX_ROWS,
        sample_rows=rows,
    )


def _failure_detail(violations, outcome=None) -> str:
    """Compact, safe reason for chat — no credentials, no full SQL dumps."""
    parts: list[str] = []
    ids: list[str] = []
    if violations:
        codes = sorted({item.code.value for item in violations[:5]})
        ids = [item.identifier for item in violations[:5] if item.identifier]
        if codes:
            parts.append(f"Validation: {', '.join(codes)}.")
    if outcome is not None and getattr(outcome, "violations", None):
        for item in outcome.violations[:5]:
            identifier = getattr(item, "identifier", None)
            if identifier and identifier not in ids:
                ids.append(identifier)
    if ids:
        parts.append(f"Fields not found in this data source: {', '.join(ids)}.")
    if outcome is not None and getattr(outcome, "analysis", None) is not None:
        reason = (outcome.analysis.reason or "").strip()
        if reason:
            parts.append(reason[:200])
    return " ".join(parts)


# ----------------------------------------------------------------------- answer


def _compose_answer(
    payload: AnalysisPayload,
    result: SQLExecutionResult,
    *,
    facts: ResultFacts | None = None,
) -> str:
    """Answer = context + measured facts, then grounded agent findings."""
    facts = facts or build_result_facts(result)
    evidence = _answer_evidence(result, facts)
    weak_agents = _weak_grounding_agents(payload.evaluation)
    scrub_notes: list[str] = []
    sections: list[str] = []

    context = _answer_context(payload)
    if context:
        sections.append(context)

    if facts.headline:
        sections.append(facts.headline)
    if facts.lines:
        sections.append(_bullets("Measured from the query results:", facts.lines))

    if payload.data_analyst and "DATA_ANALYST" not in weak_agents:
        text = payload.data_analyst.get("interpretation") or payload.data_analyst.get("summary")
        grounded = _ground_narrative(str(text or ""), evidence)
        if grounded:
            sections.append(f"Analysis: {grounded}")
        elif text:
            scrub_notes.append("Dropped analysis text with numbers not present in the query results.")

    trend = payload.trend
    if (
        trend
        and "TREND_ANALYSIS" not in weak_agents
        and trend.get("direction") not in (None, TrendDirection.INSUFFICIENT_DATA.value)
    ):
        grounded = _ground_narrative(str(trend.get("summary") or ""), evidence)
        if grounded:
            sections.append(f"Trend: {grounded}")
        elif trend.get("summary"):
            scrub_notes.append("Dropped trend text with numbers not present in the query results.")

    anomaly = payload.anomaly
    if anomaly and "ANOMALY_DETECTION" not in weak_agents and (anomaly.get("anomaly_count") or 0) > 0:
        grounded = _ground_narrative(str(anomaly.get("summary") or ""), evidence)
        if grounded:
            sections.append(f"Anomalies: {grounded}")
        elif anomaly.get("summary"):
            scrub_notes.append("Dropped anomaly text with numbers not present in the query results.")

    root_cause = payload.root_cause
    if root_cause and "ROOT_CAUSE_ANALYSIS" not in weak_agents and root_cause.get("primary_cause"):
        hypotheses = root_cause.get("hypotheses") or []
        items = []
        for hypothesis in hypotheses[:_MAX_LISTED_ITEMS]:
            statement = _ground_narrative(str(hypothesis.get("statement") or ""), evidence)
            if not statement:
                continue
            confidence = str(hypothesis.get("confidence_score") or "").lower()
            evidence_text = _ground_narrative(
                str(hypothesis.get("supporting_evidence") or ""), evidence
            )
            line = statement + (f" ({confidence} confidence)" if confidence else "")
            if evidence_text:
                line += f" Evidence: {evidence_text}"
            items.append(line)
        if not items:
            primary = _ground_narrative(str(root_cause["primary_cause"]).strip(), evidence)
            if primary:
                items = [primary]
        if items:
            sections.append(_bullets("Likely causes (hypotheses, not confirmed facts):", items))
        else:
            scrub_notes.append(
                "Dropped root-cause claims with numbers not present in the query results."
            )

    insight = payload.insight
    if insight and "INSIGHT" not in weak_agents and insight.get("insights"):
        items = []
        for item in insight["insights"][:_MAX_LISTED_ITEMS]:
            body = _ground_narrative(str(item.get("insight") or ""), evidence)
            if not body:
                continue
            title = str(item.get("title") or "").strip()
            items.append(f"{title}: {body}".strip(": "))
        if items:
            sections.append(_bullets("Insights:", items))
        elif insight.get("insights"):
            scrub_notes.append("Dropped insights with numbers not present in the query results.")

    recommendation = payload.recommendation
    if (
        recommendation
        and "RECOMMENDATION" not in weak_agents
        and recommendation.get("recommendations")
    ):
        items = []
        for item in recommendation["recommendations"][:_MAX_LISTED_ITEMS]:
            body = _ground_narrative(str(item.get("recommendation") or ""), evidence)
            if not body:
                continue
            title = str(item.get("title") or "").strip()
            items.append(f"{title}: {body}".strip(": "))
        if items:
            sections.append(_bullets("Recommendations:", items))
        elif recommendation.get("recommendations"):
            scrub_notes.append(
                "Dropped recommendations with numbers not present in the query results."
            )

    if weak_agents:
        scrub_notes.append(
            "Some narrative sections were omitted because evaluation flagged unsupported claims."
        )

    chart = payload.chart_hint or _chart_hint(facts)
    if chart:
        sections.append(f"Chart hint: {chart}.")

    notes = list(facts.notes) + list(payload.notes or []) + scrub_notes
    if notes:
        sections.append(_bullets("Notes:", notes))
    if not sections:
        sections.append(_deterministic_row_summary(result))
    return "\n\n".join(sections)


def _answer_context(payload: AnalysisPayload) -> str | None:
    lines: list[str] = []
    if payload.question_understood:
        lines.append(f"Question understood: {payload.question_understood}")
    if payload.date_range:
        lines.append(f"Date range: {payload.date_range}")
    if payload.sql:
        sql = " ".join(payload.sql.split())
        if len(sql) > _MAX_SQL_IN_ANSWER:
            sql = sql[: _MAX_SQL_IN_ANSWER - 3] + "..."
        lines.append(f"SQL: {sql}")
    if payload.assumptions:
        lines.append("Assumptions: " + "; ".join(payload.assumptions[:5]))
    return "\n".join(lines) if lines else None


def _answer_evidence(result: SQLExecutionResult, facts: ResultFacts) -> list[float]:
    values = set(evidence_numbers(query_result=result))
    for text in (facts.headline, *facts.lines, *facts.notes):
        if text:
            values.update(
                float(token.replace(",", ""))
                for token in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
            )
    return sorted(values)


def _ground_narrative(text: str, evidence: list[float]) -> str | None:
    """Keep only sentences whose cited numbers appear in the query evidence."""
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return None
    if not ungrounded_numbers([cleaned], evidence):
        return cleaned
    kept = [
        sentence.strip()
        for sentence in _SENTENCE_SPLIT_RE.split(cleaned)
        if sentence.strip() and not ungrounded_numbers([sentence], evidence)
    ]
    return " ".join(kept) if kept else None


def _weak_grounding_agents(evaluation: dict[str, Any] | None) -> set[str]:
    """Agents with verified FAIL grounding checks from evaluate_agents."""
    if not evaluation:
        return set()
    weak: set[str] = set()
    for agent_eval in evaluation.get("agents") or []:
        agent = str(agent_eval.get("agent") or "")
        for check in agent_eval.get("checks") or []:
            if check.get("status") == "FAIL" and check.get("dimension") in {
                "HALLUCINATION",
                "GROUNDING",
            }:
                weak.add(agent)
                break
    return weak


def _chart_hint(facts: ResultFacts) -> str | None:
    if facts.shape == "time_series" and any("Only one period" in note for note in facts.notes):
        return "kpi"
    return _CHART_BY_SHAPE.get(facts.shape)


def _question_understood(
    message: str,
    intent: AIIntent | None,
    profile: AnalysisProfile,
) -> str:
    parts: list[str] = []
    if intent is not None:
        if intent.subject:
            parts.append(intent.subject)
        if intent.metrics:
            parts.append(
                "metrics "
                + ", ".join(
                    f"{m.name}" + (f" ({m.aggregation.value})" if m.aggregation.value != "NONE" else "")
                    for m in intent.metrics[:5]
                )
            )
        if intent.dimensions:
            parts.append("by " + ", ".join(d.name for d in intent.dimensions[:5]))
        if intent.filters:
            parts.append(
                "filters "
                + ", ".join(f"{f.field} {f.operator.value}" for f in intent.filters[:4])
            )
    kinds = ", ".join(profile.labels())
    if parts:
        return f"{'; '.join(parts)} ({kinds})"
    text = " ".join((message or "").split())
    return f"{text[:160]} ({kinds})" if text else kinds


def _date_range_label(intent: AIIntent | None) -> str | None:
    if intent is None or intent.time_range is None:
        return None
    tr = intent.time_range
    if tr.start_date and tr.end_date:
        return f"{tr.start_date.isoformat()} to {tr.end_date.isoformat()}"
    return tr.preset.value.replace("_", " ")


def _bullets(title: str, items: list[str]) -> str:
    return title + "\n" + "\n".join(f"- {item}" for item in items)


def _deterministic_row_summary(result: SQLExecutionResult) -> str:
    cols = ", ".join(result.columns[:8]) if result.columns else "no columns"
    lines = [f"Query returned {result.row_count} row(s) ({cols})."]
    for row in result.rows[:8]:
        cells = ", ".join(str(cell) for cell in row[:8])
        lines.append(cells)
    if result.row_count > 8 or result.truncated:
        lines.append("(preview truncated)")
    return " ".join(lines)
