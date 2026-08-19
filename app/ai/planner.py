"""Deterministic request planner. Does not call an LLM or execute anything."""

from __future__ import annotations

from app.ai.intent_types import (
    AIIntent,
    AIIntentType,
    AIOperationType,
    AIPlanCapability,
    AIPlanOperation,
    AIRequestPlan,
)

_DATA_INTENTS = {
    AIIntentType.ANALYTICAL_QUERY,
    AIIntentType.DATA_LOOKUP,
    AIIntentType.AGGREGATION,
    AIIntentType.COMPARISON,
    AIIntentType.TREND_ANALYSIS,
    AIIntentType.RANKING,
    AIIntentType.SUMMARY,
}

_AGGREGATING_OPERATIONS = {
    AIOperationType.AGGREGATE,
    AIOperationType.COUNT,
    AIOperationType.RANK,
    AIOperationType.TREND,
    AIOperationType.SUMMARY,
}

_INTENT_OPERATIONS: dict[AIIntentType, tuple[AIPlanOperation, ...]] = {
    AIIntentType.SCHEMA_QUESTION: (AIPlanOperation.METADATA_LOOKUP,),
    AIIntentType.DATA_LOOKUP: (AIPlanOperation.SELECT,),
    AIIntentType.AGGREGATION: (AIPlanOperation.AGGREGATE,),
    AIIntentType.RANKING: (AIPlanOperation.RANK, AIPlanOperation.SORT),
    AIIntentType.COMPARISON: (AIPlanOperation.COMPARE,),
    AIIntentType.TREND_ANALYSIS: (AIPlanOperation.TREND, AIPlanOperation.GROUP_BY),
    AIIntentType.SUMMARY: (AIPlanOperation.SUMMARY,),
    AIIntentType.ANALYTICAL_QUERY: (AIPlanOperation.SELECT,),
}

_OPERATION_MAP: dict[AIOperationType, AIPlanOperation] = {
    AIOperationType.SELECT: AIPlanOperation.SELECT,
    AIOperationType.FILTER: AIPlanOperation.FILTER,
    AIOperationType.AGGREGATE: AIPlanOperation.AGGREGATE,
    AIOperationType.GROUP_BY: AIPlanOperation.GROUP_BY,
    AIOperationType.SORT: AIPlanOperation.SORT,
    AIOperationType.RANK: AIPlanOperation.RANK,
    AIOperationType.COMPARE: AIPlanOperation.COMPARE,
    AIOperationType.TREND: AIPlanOperation.TREND,
    AIOperationType.SUMMARY: AIPlanOperation.SUMMARY,
    AIOperationType.COUNT: AIPlanOperation.COUNT,
    AIOperationType.DISTINCT: AIPlanOperation.DISTINCT,
}

UNSUPPORTED_RESPONSE = (
    "This request is not supported. AnalyticCastle only plans read-only "
    "analytical questions."
)
NO_QUERY_EXECUTED = "No query was executed."


class AIRequestPlanner:
    """Turn a validated intent into a structured, non-executable plan."""

    def plan(self, intent: AIIntent) -> AIRequestPlan:
        if intent.intent is AIIntentType.UNSUPPORTED:
            return AIRequestPlan(
                intent=AIIntentType.UNSUPPORTED,
                unsupported=True,
                requires_clarification=False,
            )
        if intent.requires_clarification or intent.intent is AIIntentType.UNKNOWN:
            return AIRequestPlan(
                intent=intent.intent,
                requires_clarification=True,
                clarification_question=intent.clarification_question,
                requires_metadata=intent.requires_metadata,
            )

        operations = _operations_for(intent)
        requires_metadata = _requires_metadata(intent)
        requires_database = _requires_database(intent)
        requires_sample_data = intent.intent is AIIntentType.DATA_LOOKUP
        requires_aggregation = _requires_aggregation(intent)
        requires_time_filter = intent.time_range is not None
        requires_relationships = intent.requires_relationships
        capabilities = _capabilities(
            requires_metadata=requires_metadata,
            requires_database=requires_database,
            requires_sample_data=requires_sample_data,
            requires_aggregation=requires_aggregation,
            requires_time_filter=requires_time_filter,
            requires_relationships=requires_relationships,
        )
        return AIRequestPlan(
            intent=intent.intent,
            requires_clarification=False,
            clarification_question=None,
            operations=operations,
            required_capabilities=capabilities,
            requires_metadata=requires_metadata,
            requires_database=requires_database,
            requires_sample_data=requires_sample_data,
            requires_aggregation=requires_aggregation,
            requires_time_filter=requires_time_filter,
            requires_relationships=requires_relationships,
            unsupported=False,
        )


def foundation_response(intent: AIIntent, plan: AIRequestPlan) -> str:
    """User-facing planning message. Never includes SQL or query results."""
    if plan.unsupported:
        return UNSUPPORTED_RESPONSE
    if plan.requires_clarification:
        return (
            plan.clarification_question
            or intent.clarification_question
            or "I need more detail before I can plan this request."
        )
    label = intent.intent.value.replace("_", " ").lower()
    parts = [f"I understood this as a {label}."]
    if intent.operation is not None:
        parts.append(
            f"Planned operation: {intent.operation.value.replace('_', ' ').lower()}."
        )
    if plan.required_capabilities:
        names = ", ".join(item.value.lower() for item in plan.required_capabilities)
        parts.append(f"Required next capabilities: {names}.")
    parts.append(NO_QUERY_EXECUTED)
    return " ".join(parts)


def _operations_for(intent: AIIntent) -> list[AIPlanOperation]:
    ordered: list[AIPlanOperation] = []
    seen: set[AIPlanOperation] = set()

    def add(operation: AIPlanOperation) -> None:
        if operation not in seen:
            ordered.append(operation)
            seen.add(operation)

    if intent.operation is not None:
        add(_OPERATION_MAP[intent.operation])
    for item in _INTENT_OPERATIONS.get(intent.intent, ()):
        add(item)
    if intent.filters:
        add(AIPlanOperation.FILTER)
    if intent.time_range is not None:
        add(AIPlanOperation.FILTER_TIME)
    if intent.dimensions and intent.intent in {
        AIIntentType.AGGREGATION,
        AIIntentType.TREND_ANALYSIS,
        AIIntentType.ANALYTICAL_QUERY,
        AIIntentType.SUMMARY,
    }:
        add(AIPlanOperation.GROUP_BY)
    if intent.sort is not None:
        add(AIPlanOperation.SORT)
    if not ordered:
        add(AIPlanOperation.SELECT)
    return ordered


def _requires_metadata(intent: AIIntent) -> bool:
    if intent.requires_metadata:
        return True
    return (
        intent.intent in _DATA_INTENTS or intent.intent is AIIntentType.SCHEMA_QUESTION
    )


def _requires_database(intent: AIIntent) -> bool:
    if intent.intent is AIIntentType.SCHEMA_QUESTION:
        return False
    if intent.requires_data_access:
        return True
    return intent.intent in _DATA_INTENTS


def _requires_aggregation(intent: AIIntent) -> bool:
    if intent.intent in {
        AIIntentType.AGGREGATION,
        AIIntentType.RANKING,
        AIIntentType.TREND_ANALYSIS,
        AIIntentType.SUMMARY,
    }:
        return True
    if intent.operation in _AGGREGATING_OPERATIONS:
        return True
    return any(metric.aggregation.value != "NONE" for metric in intent.metrics)


def _capabilities(
    *,
    requires_metadata: bool,
    requires_database: bool,
    requires_sample_data: bool,
    requires_aggregation: bool,
    requires_time_filter: bool,
    requires_relationships: bool,
) -> list[AIPlanCapability]:
    capabilities: list[AIPlanCapability] = []
    if requires_metadata:
        capabilities.append(AIPlanCapability.METADATA)
    if requires_database:
        capabilities.append(AIPlanCapability.DATABASE)
    if requires_sample_data:
        capabilities.append(AIPlanCapability.SAMPLE_DATA)
    if requires_aggregation:
        capabilities.append(AIPlanCapability.AGGREGATION)
    if requires_time_filter:
        capabilities.append(AIPlanCapability.TIME_FILTER)
    if requires_relationships:
        capabilities.append(AIPlanCapability.RELATIONSHIPS)
    return capabilities
