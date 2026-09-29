from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ai.intent_types import (
    AggregationType,
    AIConfidence,
    AIIntent,
    AIIntentType,
    AIOperationType,
    AIPlanCapability,
    AIPlanOperation,
    AIRequestPlan,
    FilterOperator,
    SortDirection,
    TimeRangePreset,
)
from app.ai.metadata_types import ResolvedMetadataContext
from app.core.config import settings


def _strip_text(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


class AIChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=32_000)
    data_source_id: UUID
    conversation_id: UUID | None = None
    conversation_version: int | None = Field(default=None, ge=1)

    @field_validator("message", mode="before")
    @classmethod
    def strip_and_limit_message(cls, value: Any) -> Any:
        value = _strip_text(value)
        if not isinstance(value, str):
            return value
        max_chars = settings.AI_MAX_MESSAGE_CHARS
        if len(value) > max_chars:
            raise ValueError(
                f"Message exceeds maximum length of {max_chars} characters"
            )
        return value


class AIUsageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class AIMetricResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    aggregation: AggregationType = AggregationType.NONE


class AIDimensionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str


class AIFilterResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    operator: FilterOperator
    value: str | int | float | bool | None = None
    values: list[str | int | float | bool] | None = None
    start: str | int | float | bool | None = None
    end: str | int | float | bool | None = None


class AITimeRangeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset: TimeRangePreset
    start_date: str | None = None
    end_date: str | None = None


class AISortResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    direction: SortDirection = SortDirection.DESC


class AIIntentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: AIIntentType
    operation: AIOperationType | None = None
    subject: str | None = None
    metrics: list[AIMetricResponse] = Field(default_factory=list)
    dimensions: list[AIDimensionResponse] = Field(default_factory=list)
    filters: list[AIFilterResponse] = Field(default_factory=list)
    time_range: AITimeRangeResponse | None = None
    sort: AISortResponse | None = None
    requested_limit: int | None = None
    safe_limit: int | None = None
    exceeds_limit: bool = False
    requires_data_access: bool = False
    requires_metadata: bool = False
    confidence: AIConfidence = AIConfidence.MEDIUM
    requires_clarification: bool = False
    clarification_question: str | None = None


class AIPlanResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requires_clarification: bool = False
    clarification_question: str | None = None
    operations: list[AIPlanOperation] = Field(default_factory=list)
    required_capabilities: list[AIPlanCapability] = Field(default_factory=list)
    requires_metadata: bool = False
    requires_database: bool = False
    requires_sample_data: bool = False
    requires_aggregation: bool = False
    requires_time_filter: bool = False
    requires_relationships: bool = False
    unsupported: bool = False


class AIQueryPreviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: list[str] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    sample_rows: list[list[Any]] = Field(default_factory=list)


class AIPhase8AnalysisResponse(BaseModel):
    """Optional Phase 8 analysis panels for the AI Analyst frontend."""

    model_config = ConfigDict(extra="forbid")

    session_id: UUID | None = None
    data_analyst: dict[str, Any] | None = None
    trend: dict[str, Any] | None = None
    anomaly: dict[str, Any] | None = None
    root_cause: dict[str, Any] | None = None
    insight: dict[str, Any] | None = None
    recommendation: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    sql: str | None = None
    query_preview: AIQueryPreviewResponse | None = None
    question_types: list[str] | None = None
    facts: list[str] | None = None
    notes: list[str] | None = None


class AIChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    response: str
    model: str
    usage: AIUsageResponse | None = None
    intent: AIIntentResponse
    plan: AIPlanResponse
    metadata_context: ResolvedMetadataContext
    conversation_id: UUID | None = None
    conversation_version: int | None = Field(default=None, ge=1)
    analysis: AIPhase8AnalysisResponse | None = None


def intent_response(intent: AIIntent) -> AIIntentResponse:
    time_range = None
    if intent.time_range is not None:
        time_range = AITimeRangeResponse(
            preset=intent.time_range.preset,
            start_date=(
                intent.time_range.start_date.isoformat()
                if intent.time_range.start_date
                else None
            ),
            end_date=(
                intent.time_range.end_date.isoformat()
                if intent.time_range.end_date
                else None
            ),
        )
    return AIIntentResponse(
        type=intent.intent,
        operation=intent.operation,
        subject=intent.subject,
        metrics=[
            AIMetricResponse(name=item.name, aggregation=item.aggregation)
            for item in intent.metrics
        ],
        dimensions=[AIDimensionResponse(name=item.name) for item in intent.dimensions],
        filters=[
            AIFilterResponse(
                field=item.field,
                operator=item.operator,
                value=item.value,
                values=item.values,
                start=item.start,
                end=item.end,
            )
            for item in intent.filters
        ],
        time_range=time_range,
        sort=(
            AISortResponse(field=intent.sort.field, direction=intent.sort.direction)
            if intent.sort
            else None
        ),
        requested_limit=intent.requested_limit,
        safe_limit=intent.safe_limit,
        exceeds_limit=intent.exceeds_limit,
        requires_data_access=intent.requires_data_access,
        requires_metadata=intent.requires_metadata,
        confidence=intent.confidence,
        requires_clarification=intent.requires_clarification,
        clarification_question=intent.clarification_question,
    )


def plan_response(plan: AIRequestPlan) -> AIPlanResponse:
    return AIPlanResponse(
        requires_clarification=plan.requires_clarification,
        clarification_question=plan.clarification_question,
        operations=list(plan.operations),
        required_capabilities=list(plan.required_capabilities),
        requires_metadata=plan.requires_metadata,
        requires_database=plan.requires_database,
        requires_sample_data=plan.requires_sample_data,
        requires_aggregation=plan.requires_aggregation,
        requires_time_filter=plan.requires_time_filter,
        requires_relationships=plan.requires_relationships,
        unsupported=plan.unsupported,
    )


def phase8_analysis_response(payload: object | None) -> AIPhase8AnalysisResponse | None:
    if payload is None:
        return None
    from app.ai.analysis_pipeline import AnalysisPayload

    if not isinstance(payload, AnalysisPayload):
        return None
    preview = None
    if payload.query_preview is not None:
        preview = AIQueryPreviewResponse(
            columns=list(payload.query_preview.columns),
            row_count=payload.query_preview.row_count,
            truncated=payload.query_preview.truncated,
            sample_rows=list(payload.query_preview.sample_rows),
        )
    return AIPhase8AnalysisResponse(
        session_id=payload.session_id,
        data_analyst=payload.data_analyst,
        trend=payload.trend,
        anomaly=payload.anomaly,
        root_cause=payload.root_cause,
        insight=payload.insight,
        recommendation=payload.recommendation,
        evaluation=payload.evaluation,
        sql=payload.sql,
        query_preview=preview,
        question_types=payload.question_types,
        facts=payload.facts,
        notes=payload.notes,
    )
