"""Chat wiring: SQL pipeline + Phase 8 via supervised orchestrator."""

from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.orm import Session

from app.ai.analysis_pipeline import (
    AnalysisPayload,
    AnalysisPipelineResult,
    QueryPreview,
    has_sufficient_metadata,
    should_run_analysis,
)
from app.ai.data_analyst.models import AnalysisConfidence, DataAnalysisResult
from app.ai.intent_types import (
    AIConfidence,
    AIIntent,
    AIIntentType,
    AIRequestPlan,
)
from app.ai.metadata_types import (
    MetadataColumnCandidate,
    MetadataMatchReason,
    MetadataTableCandidate,
    ResolvedMetadataContext,
    empty_resolved_context,
)
from app.ai.orchestrator import AIAnalystOrchestrator
from app.ai.sql_execution.models import SQLExecutionResult, SQLExecutionStatus
from app.ai.supervisor.models import (
    ClassificationConfidence,
    RequestCategory,
    RequestClassification,
)
from app.ai.types import AI_CAPABILITY_CHAT, AIContext, AIRequest
from app.db.models import AnalysisSession, DataSourceType
from app.enums import AgentPhase, AnalysisSessionStatus
from tests.ai_fakes import FakeLLMProvider
from tests.conftest import run_async
from tests.planner_helpers import configure_analytical_provider
from tests.test_supervisor import _seed_data_source, _seed_workspace

pytest_plugins = ["tests.planner_helpers"]


def _context(
    *,
    user_id: uuid.UUID,
    workspace_id: uuid.UUID,
    organization_id: uuid.UUID,
    data_source_id: uuid.UUID,
) -> AIContext:
    return AIContext(
        user_id=user_id,
        workspace_id=workspace_id,
        organization_id=organization_id,
        data_source_id=data_source_id,
        data_source_name="Analytics DB",
        data_source_type=DataSourceType.POSTGRESQL.value,
        workspace_name="Analytics",
        workspace_role="OWNER",
        allowed_capabilities=frozenset({AI_CAPABILITY_CHAT}),
    )


def _table(data_source_id: uuid.UUID) -> MetadataTableCandidate:
    return MetadataTableCandidate(
        table_id=uuid.uuid4(),
        schema_name="public",
        table_name="sales",
        match_reason=MetadataMatchReason.EXACT,
        relevance_score=100,
    )


def _column(table: MetadataTableCandidate, name: str) -> MetadataColumnCandidate:
    return MetadataColumnCandidate(
        column_id=uuid.uuid4(),
        table_id=table.table_id,
        schema_name=table.schema_name,
        table_name=table.table_name,
        column_name=name,
        data_type="numeric",
        match_reason=MetadataMatchReason.EXACT,
        relevance_score=100,
    )


def _rich_metadata(data_source_id: uuid.UUID) -> ResolvedMetadataContext:
    table = _table(data_source_id)
    return ResolvedMetadataContext(
        data_source_id=data_source_id,
        tables=[table],
        columns=[_column(table, "revenue"), _column(table, "sale_date")],
        requires_clarification=False,
    )


@pytest.fixture(autouse=True)
def mock_supervisor_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _classify(**_kwargs: object) -> RequestClassification:
        return RequestClassification(
            category=RequestCategory.ANALYTICAL_QUERY,
            confidence=ClassificationConfidence.HIGH,
            requires_data_access=True,
        )

    monkeypatch.setattr("app.ai.supervisor.service.classify_request", _classify)


@pytest.fixture(autouse=True)
def mock_planner_for_supervised_chat(mock_planner_llm: None) -> None:
    """Ensure supervised orchestrator tests never call a live planner LLM."""


def test_should_run_analysis_requires_database_and_metadata() -> None:
    ds = uuid.uuid4()
    intent = AIIntent(
        intent=AIIntentType.ANALYTICAL_QUERY,
        confidence=AIConfidence.HIGH,
        requires_data_access=True,
    )
    plan = AIRequestPlan(intent=AIIntentType.ANALYTICAL_QUERY, requires_database=True)
    meta = _rich_metadata(ds)
    assert should_run_analysis(plan=plan, intent=intent, metadata=meta) is True

    # Clarification left on the plan after the policy is genuine and blocks SQL.
    clarify = plan.model_copy(
        update={"requires_clarification": True, "clarification_question": "Which metric?"}
    )
    assert should_run_analysis(plan=clarify, intent=intent, metadata=meta) is False

    empty = empty_resolved_context(ds)
    assert has_sufficient_metadata(empty) is False
    assert should_run_analysis(plan=plan, intent=intent, metadata=empty) is False


def _typed_column(
    table: MetadataTableCandidate, name: str, data_type: str
) -> MetadataColumnCandidate:
    return _column(table, name).model_copy(update={"data_type": data_type})


def _grounded_metadata(ds: uuid.UUID) -> ResolvedMetadataContext:
    from app.ai.metadata_types import ConceptColumnResolution

    table = _table(ds)
    metric = _typed_column(table, "revenue", "numeric")
    return ResolvedMetadataContext(
        data_source_id=ds,
        tables=[table],
        columns=[
            metric,
            _typed_column(table, "sale_date", "date"),
            _typed_column(table, "region", "string"),
            _typed_column(table, "product_id", "integer"),
        ],
        resolved_metrics=[
            ConceptColumnResolution(requested="revenue", resolved=True, candidates=[metric])
        ],
    )


def test_deterministic_query_is_grounded_and_valid() -> None:
    from app.ai.analysis_pipeline import _deterministic_change_sql
    from app.ai.sql_validation.validation import validate_generated_sql

    ds = uuid.uuid4()
    meta = _grounded_metadata(ds)
    sql = _deterministic_change_sql("Why did revenue fall in March 2025?", meta)
    assert sql is not None
    assert "sale_date" in sql and "region" in sql and "2025-02-01" in sql
    assert "2025-04-01" in sql
    assert validate_generated_sql(sql, meta).is_valid


def test_deterministic_query_refuses_to_guess() -> None:
    from app.ai.analysis_pipeline import _deterministic_query
    from app.ai.analysis_profile import build_analysis_profile
    from app.ai.intent_types import AIFilter, FilterOperator

    ds = uuid.uuid4()
    table = _table(ds)
    # No resolved metric: never pick an arbitrary numeric column.
    unresolved = ResolvedMetadataContext(
        data_source_id=ds,
        tables=[table],
        columns=[_typed_column(table, "revenue", "numeric")],
    )
    message = "Why did profit fall in March 2025?"
    assert (
        _deterministic_query(message, unresolved, profile=build_analysis_profile(message))
        is None
    )
    # Filters cannot be applied faithfully: no fallback.
    intent = AIIntent(
        intent=AIIntentType.AGGREGATION,
        filters=[AIFilter(field="region", operator=FilterOperator.EQUALS, value="North")],
    )
    assert _deterministic_query("revenue in North", _grounded_metadata(ds), intent=intent) is None


def test_genuine_clarification_kept_for_unmatched_metric() -> None:
    from app.ai.metadata_types import ConceptColumnResolution
    from app.ai.orchestrator import _apply_clarification_policy

    ds = uuid.uuid4()
    intent = AIIntent(intent=AIIntentType.AGGREGATION, requires_data_access=True)
    plan = AIRequestPlan(intent=AIIntentType.AGGREGATION, requires_database=True)
    meta = _rich_metadata(ds).model_copy(
        update={
            "resolved_metrics": [
                ConceptColumnResolution(requested="churn", resolved=False, candidates=[])
            ]
        }
    )
    _intent, plan2, _meta = _apply_clarification_policy(intent, plan, meta)
    assert plan2.requires_clarification is True
    assert "churn" in (plan2.clarification_question or "")
    assert should_run_analysis(plan=plan2, intent=_intent, metadata=_meta) is False

    vague = AIIntent(intent=AIIntentType.UNKNOWN, requires_clarification=True)
    _i, plan3, _m = _apply_clarification_policy(vague, plan, _rich_metadata(ds))
    assert plan3.requires_clarification is True


def test_clarification_policy_rebuilds_db_plan_when_catalog_matched() -> None:
    from app.ai.orchestrator import _apply_clarification_policy

    ds = uuid.uuid4()
    intent = AIIntent(
        intent=AIIntentType.ANALYTICAL_QUERY,
        confidence=AIConfidence.HIGH,
        requires_data_access=True,
        requires_clarification=True,
        clarification_question="Which metric?",
    )
    # Planner clarification stub — no requires_database (the live bug).
    plan = AIRequestPlan(
        intent=AIIntentType.ANALYTICAL_QUERY,
        requires_clarification=True,
        clarification_question=(
            "I need more detail about: Exact name of the revenue column "
            "in the sales table, Exact name of the date column."
        ),
    )
    meta = _rich_metadata(ds)
    intent2, plan2, meta2 = _apply_clarification_policy(intent, plan, meta)
    assert intent2.requires_clarification is False
    assert plan2.requires_clarification is False
    assert plan2.requires_database is True
    assert meta2.requires_clarification is False
    assert should_run_analysis(plan=plan2, intent=intent2, metadata=meta2) is True


def test_chat_happy_path_sql_and_phase8(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    context = _context(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        data_source_id=data_source.id,
    )
    provider = FakeLLMProvider()
    configure_analytical_provider(provider)

    meta = _rich_metadata(data_source.id)

    class _Resolver:
        def resolve(self, *_a: Any, **_k: Any) -> ResolvedMetadataContext:
            return meta

    payload = AnalysisPayload(
        session_id=uuid.uuid4(),
        data_analyst=DataAnalysisResult(
            interpretation="Monthly revenue for 2025 totals 42,000.",
            summary="Sum=42000",
            comparisons="N/A",
            conclusions=["January led."],
            confidence_score=AnalysisConfidence.HIGH,
            confidence_reasoning="Enough rows.",
        ).model_dump(mode="json"),
        insight={"top_insight": "Revenue peaked in January.", "summary": "ok"},
        recommendation={"top_recommendation": "Review March dip."},
        sql="SELECT date_trunc('month', sale_date), sum(revenue) FROM public.sales GROUP BY 1",
        query_preview=QueryPreview(
            columns=["month", "sum"],
            row_count=3,
            truncated=False,
            sample_rows=[["2025-01", 20000], ["2025-02", 15000], ["2025-03", 7000]],
        ),
    )

    async def _fake_pipeline(params: Any) -> AnalysisPipelineResult:
        return AnalysisPipelineResult(
            answer="Monthly revenue for 2025 totals 42,000. Revenue peaked in January.",
            payload=replace(payload, session_id=params.session_id),
        )

    monkeypatch.setattr(
        "app.ai.orchestrator.run_analysis_pipeline",
        _fake_pipeline,
    )

    orchestrator = AIAnalystOrchestrator(provider, resolver=_Resolver())
    result = run_async(
        orchestrator.chat(
            AIRequest(
                message="Sum monthly revenue from sales for 2025",
                data_source_id=data_source.id,
            ),
            context,
            db=db_session,
        )
    )

    assert "42,000" in result.response or "January" in result.response
    assert "No query was executed" not in result.response
    assert result.phase8_analysis is not None
    assert result.phase8_analysis.sql is not None
    assert result.phase8_analysis.data_analyst is not None
    session = (
        db_session.query(AnalysisSession)
        .filter(AnalysisSession.id == result.conversation_id)
        .one()
    )
    assert session.status is AnalysisSessionStatus.ACTIVE
    assert session.agent_state is not None
    assert session.agent_state.phase is AgentPhase.PLANNING


def test_chat_clarification_skips_pipeline(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    context = _context(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        data_source_id=data_source.id,
    )
    provider = FakeLLMProvider()
    configure_analytical_provider(provider)

    called = {"n": 0}

    async def _fake_pipeline(_params: Any) -> AnalysisPipelineResult:
        called["n"] += 1
        return AnalysisPipelineResult(answer="should not run", failed=True)

    monkeypatch.setattr("app.ai.orchestrator.run_analysis_pipeline", _fake_pipeline)

    # Planner will produce a normal plan; force insufficient metadata so SQL is skipped.
    class _Resolver:
        def resolve(self, *_a: Any, **_k: Any) -> ResolvedMetadataContext:
            return empty_resolved_context(data_source.id)

    orchestrator = AIAnalystOrchestrator(provider, resolver=_Resolver())
    result = run_async(
        orchestrator.chat(
            AIRequest(message="How many orders?", data_source_id=data_source.id),
            context,
            db=db_session,
        )
    )
    assert called["n"] == 0
    assert result.phase8_analysis is None
    assert "No query was executed" in result.response


def test_chat_sql_failure_skips_phase8(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    context = _context(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        data_source_id=data_source.id,
    )
    provider = FakeLLMProvider()
    configure_analytical_provider(provider)

    class _Resolver:
        def resolve(self, *_a: Any, **_k: Any) -> ResolvedMetadataContext:
            return _rich_metadata(data_source.id)

    async def _fail_pipeline(_params: Any) -> AnalysisPipelineResult:
        return AnalysisPipelineResult(
            answer="I could not produce a valid query for that request.",
            failed=True,
            payload=None,
        )

    monkeypatch.setattr("app.ai.orchestrator.run_analysis_pipeline", _fail_pipeline)
    orchestrator = AIAnalystOrchestrator(provider, resolver=_Resolver())
    result = run_async(
        orchestrator.chat(
            AIRequest(message="Sum revenue", data_source_id=data_source.id),
            context,
            db=db_session,
        )
    )
    assert result.phase8_analysis is None
    assert "could not produce a valid query" in result.response.lower()


def test_pipeline_partial_phase8_skips(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trend/anomaly skip paths still yield analyst answer when SQL succeeds."""
    from app.ai import analysis_pipeline as pipeline_mod

    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    from app.ai.state import AgentStateService, CreateSessionParams

    state = AgentStateService(db_session)
    snapshot = state.create_session(
        CreateSessionParams(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
            data_source_id=data_source.id,
        )
    )

    exec_result = SQLExecutionResult(
        status=SQLExecutionStatus.SUCCEEDED,
        columns=["month", "revenue"],
        rows=[["2025-01", 10]],
        row_count=1,
    )

    analyst = DataAnalysisResult(
        interpretation="One month of revenue is 10.",
        summary="n=1",
        comparisons="none",
        conclusions=["Sparse data."],
        confidence_score=AnalysisConfidence.LOW,
        confidence_reasoning="Single row.",
    )

    async def _analyze(**_k: Any) -> DataAnalysisResult:
        return analyst

    monkeypatch.setattr(
        pipeline_mod.DataAnalystAgent,
        "analyze",
        AsyncMock(side_effect=_analyze),
    )
    monkeypatch.setattr(
        pipeline_mod.TrendAnalysisAgent,
        "analyze",
        AsyncMock(side_effect=RuntimeError("skip trend")),
    )
    monkeypatch.setattr(
        pipeline_mod.AnomalyDetectionAgent,
        "analyze",
        AsyncMock(side_effect=RuntimeError("skip anomaly")),
    )
    monkeypatch.setattr(
        pipeline_mod.RootCauseAnalysisAgent,
        "analyze",
        AsyncMock(side_effect=RuntimeError("skip rca")),
    )
    monkeypatch.setattr(
        pipeline_mod.InsightAgent,
        "generate",
        AsyncMock(side_effect=RuntimeError("skip insight")),
    )
    monkeypatch.setattr(
        pipeline_mod.RecommendationAgent,
        "recommend",
        AsyncMock(side_effect=RuntimeError("skip rec")),
    )

    _force_llm_key(monkeypatch)
    payload = run_async(
        pipeline_mod._run_phase8(
            db=db_session,
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            organization_id=organization.id,
            message="Compare revenue by region",
            sql="SELECT 1",
            query_result=exec_result,
            metadata=_rich_metadata(data_source.id),
            expected_agent_version=None,
            run_evaluation=False,
        )
    )
    assert payload.data_analyst is not None
    # Comparison/breakdown questions do not run trend/insight agents at all.
    assert payload.trend is None
    assert payload.insight is None
    assert pipeline_mod.TrendAnalysisAgent.analyze.call_count == 0
    answer = pipeline_mod._compose_answer(payload, exec_result)
    assert "10" in answer
    assert "Analysis: One month of revenue is 10." in answer


def _force_llm_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace as dc_replace

    import app.ai.llm as llm_mod

    real = llm_mod.llm_client_config_from_settings

    def _config():
        cfg = real()
        return cfg if cfg.api_key else dc_replace(cfg, api_key="test-key")

    monkeypatch.setattr(llm_mod, "llm_client_config_from_settings", _config)


def test_simple_aggregation_uses_no_llm_agents(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.ai import analysis_pipeline as pipeline_mod

    for agent, method in (
        (pipeline_mod.DataAnalystAgent, "analyze"),
        (pipeline_mod.TrendAnalysisAgent, "analyze"),
        (pipeline_mod.AnomalyDetectionAgent, "analyze"),
        (pipeline_mod.RootCauseAnalysisAgent, "analyze"),
        (pipeline_mod.InsightAgent, "generate"),
        (pipeline_mod.RecommendationAgent, "recommend"),
    ):
        monkeypatch.setattr(agent, method, AsyncMock(side_effect=AssertionError("no LLM")))
    result = SQLExecutionResult(
        status=SQLExecutionStatus.SUCCEEDED,
        columns=["total_revenue"],
        rows=[[187650]],
        row_count=1,
    )
    payload = run_async(
        pipeline_mod._run_phase8(
            db=db_session,
            session_id=uuid.uuid4(),
            workspace_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            organization_id=uuid.uuid4(),
            message="What is the total revenue for 2025?",
            sql="SELECT 1",
            query_result=result,
            metadata=_rich_metadata(uuid.uuid4()),
            expected_agent_version=None,
            run_evaluation=False,
        )
    )
    answer = pipeline_mod._compose_answer(payload, result)
    assert answer.startswith("total revenue: 187,650")
    assert payload.question_types == ["AGGREGATE"]


def test_diagnostic_runs_agents_concurrently_and_separates_hypotheses(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    import time as time_mod
    from datetime import date

    from app.ai import analysis_pipeline as pipeline_mod
    from app.ai.root_cause_analysis.models import (
        RootCauseAnalysisResult,
        RootCauseHypothesis,
    )

    async def _slow_none(**_k: Any) -> None:
        await asyncio.sleep(0.4)
        return None

    rca = RootCauseAnalysisResult(
        summary="North drove the March decline.",
        primary_cause="North sales collapsed in March.",
        hypotheses=[
            RootCauseHypothesis(
                rank=1,
                statement="North sales collapsed in March.",
                contributing_factors=["North"],
                supporting_evidence="North fell 80% month over month.",
                confidence_score=AnalysisConfidence.MEDIUM,
                confidence_reasoning="Single breakdown.",
            )
        ],
        confidence_score=AnalysisConfidence.MEDIUM,
        confidence_reasoning="Single breakdown.",
    )
    monkeypatch.setattr(pipeline_mod.TrendAnalysisAgent, "analyze", AsyncMock(side_effect=_slow_none))
    monkeypatch.setattr(pipeline_mod.AnomalyDetectionAgent, "analyze", AsyncMock(side_effect=_slow_none))
    monkeypatch.setattr(pipeline_mod.RootCauseAnalysisAgent, "analyze", AsyncMock(return_value=rca))
    monkeypatch.setattr(
        pipeline_mod.InsightAgent, "generate", AsyncMock(side_effect=RuntimeError("down"))
    )
    data_analyst = AsyncMock(side_effect=AssertionError("not needed"))
    monkeypatch.setattr(pipeline_mod.DataAnalystAgent, "analyze", data_analyst)
    _force_llm_key(monkeypatch)

    rows = [
        [date(2025, 2, 1), "North", 6000],
        [date(2025, 2, 1), "South", 5000],
        [date(2025, 3, 1), "North", 1200],
        [date(2025, 3, 1), "South", 4800],
    ]
    result = SQLExecutionResult(
        status=SQLExecutionStatus.SUCCEEDED,
        columns=["period", "region", "revenue"],
        rows=rows,
        row_count=len(rows),
    )
    started = time_mod.perf_counter()
    payload = run_async(
        pipeline_mod._run_phase8(
            db=db_session,
            session_id=uuid.uuid4(),
            workspace_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            organization_id=uuid.uuid4(),
            message="Why did revenue fall in March 2025?",
            sql="SELECT 1",
            query_result=result,
            metadata=_rich_metadata(uuid.uuid4()),
            expected_agent_version=None,
            run_evaluation=False,
        )
    )
    elapsed = time_mod.perf_counter() - started
    assert elapsed < 0.75, "trend and anomaly should run concurrently"
    assert data_analyst.call_count == 0
    answer = pipeline_mod._compose_answer(payload, result)
    assert answer.startswith("revenue for Mar 2025 was 6,000, down 5,000")
    assert "Largest contributors by region" in answer
    assert "Likely causes (hypotheses, not confirmed facts):" in answer
    assert "North sales collapsed in March." in answer
    assert "Insight generation was unavailable" in answer
    assert "DIAGNOSTIC" in (payload.question_types or [])
