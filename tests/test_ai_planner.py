from __future__ import annotations

from app.ai.intent import normalize_intent
from app.ai.intent_types import (
    AggregationType,
    AIConfidence,
    AIDimension,
    AIFilter,
    AIIntent,
    AIIntentType,
    AIMetric,
    AIOperationType,
    AIPlanCapability,
    AIPlanOperation,
    AITimeRange,
    FilterOperator,
    LLMIntentDetection,
    TimeRangePreset,
)
from app.ai.planner import AIRequestPlanner, foundation_response
from app.ai.safety import unsupported_intent


def _plan(intent: AIIntent):
    return AIRequestPlanner().plan(intent)


def test_count_plan_requires_metadata_and_data() -> None:
    intent = normalize_intent(
        LLMIntentDetection(
            intent=AIIntentType.ANALYTICAL_QUERY,
            operation=AIOperationType.COUNT,
            subject="orders",
            time_range=AITimeRange(preset=TimeRangePreset.LAST_MONTH),
            requires_data_access=True,
            requires_metadata=True,
            confidence=AIConfidence.HIGH,
        )
    )
    plan = _plan(intent)
    assert plan.requires_clarification is False
    assert plan.requires_metadata is True
    assert plan.requires_database is True
    assert plan.requires_time_filter is True
    assert AIPlanOperation.COUNT in plan.operations
    assert AIPlanOperation.FILTER_TIME in plan.operations
    assert AIPlanCapability.METADATA in plan.required_capabilities
    assert AIPlanCapability.DATABASE in plan.required_capabilities
    assert AIPlanCapability.TIME_FILTER in plan.required_capabilities
    dumped = plan.model_dump()
    assert "sql" not in dumped
    assert foundation_response(intent, plan).endswith("No query was executed.")


def test_ranking_plan_includes_rank_operation() -> None:
    intent = normalize_intent(
        LLMIntentDetection(
            intent=AIIntentType.RANKING,
            operation=AIOperationType.RANK,
            subject="customers",
            metrics=[AIMetric(name="revenue", aggregation=AggregationType.SUM)],
            dimensions=[AIDimension(name="customer")],
            requested_limit=10,
            requires_data_access=True,
            requires_metadata=True,
            confidence=AIConfidence.HIGH,
        )
    )
    plan = _plan(intent)
    assert AIPlanOperation.RANK in plan.operations
    assert plan.requires_aggregation is True
    assert plan.unsupported is False


def test_schema_question_does_not_require_database() -> None:
    intent = normalize_intent(
        LLMIntentDetection(
            intent=AIIntentType.SCHEMA_QUESTION,
            subject="tables",
            requires_data_access=False,
            requires_metadata=True,
            confidence=AIConfidence.HIGH,
        )
    )
    plan = _plan(intent)
    assert plan.requires_metadata is True
    assert plan.requires_database is False
    assert AIPlanOperation.METADATA_LOOKUP in plan.operations
    assert AIPlanCapability.DATABASE not in plan.required_capabilities


def test_filtered_lookup_plan() -> None:
    intent = normalize_intent(
        LLMIntentDetection(
            intent=AIIntentType.DATA_LOOKUP,
            operation=AIOperationType.SELECT,
            subject="orders",
            filters=[
                AIFilter(field="country", operator=FilterOperator.EQUALS, value="India")
            ],
            requires_data_access=True,
            requires_metadata=True,
            confidence=AIConfidence.HIGH,
        )
    )
    plan = _plan(intent)
    assert AIPlanOperation.SELECT in plan.operations
    assert AIPlanOperation.FILTER in plan.operations
    assert plan.requires_sample_data is True
    assert plan.requires_database is True


def test_clarification_has_no_execution_operations() -> None:
    intent = normalize_intent(
        LLMIntentDetection(
            intent=AIIntentType.UNKNOWN,
            subject="sales",
            confidence=AIConfidence.LOW,
            requires_clarification=True,
            clarification_question="What would you like to know about sales?",
        )
    )
    plan = _plan(intent)
    assert plan.requires_clarification is True
    assert plan.operations == []
    assert plan.required_capabilities == []
    assert plan.unsupported is False
    assert "sales" in foundation_response(intent, plan)


def test_unsupported_has_no_execution_plan() -> None:
    intent = unsupported_intent("write_operation")
    plan = _plan(intent)
    assert plan.unsupported is True
    assert plan.operations == []
    assert plan.required_capabilities == []
    assert plan.requires_database is False
    assert plan.requires_metadata is False
    assert "read-only" in foundation_response(intent, plan).lower()
    assert "SELECT" not in foundation_response(intent, plan)


def test_aggregation_with_dimension_adds_group_by() -> None:
    intent = normalize_intent(
        LLMIntentDetection(
            intent=AIIntentType.AGGREGATION,
            operation=AIOperationType.AGGREGATE,
            subject="revenue",
            metrics=[AIMetric(name="revenue", aggregation=AggregationType.SUM)],
            dimensions=[AIDimension(name="month")],
            requires_data_access=True,
            requires_metadata=True,
            confidence=AIConfidence.HIGH,
        )
    )
    plan = _plan(intent)
    assert AIPlanOperation.AGGREGATE in plan.operations
    assert AIPlanOperation.GROUP_BY in plan.operations
    assert plan.requires_aggregation is True
