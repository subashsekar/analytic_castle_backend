"""Typed intent, filter, metric, and plan models for Phase 5.2.

These structures describe what the user asked for. They are not SQL, catalog
identifiers, or query results.
"""

from __future__ import annotations

import enum
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.core.config import settings

_MAX_LLM_COLLECTION = 50
_MAX_CLARIFICATION_CHARS = 500
_MAX_UNSUPPORTED_REASON_CHARS = 128
FilterScalar = str | int | float | bool


class AIIntentType(str, enum.Enum):
    ANALYTICAL_QUERY = "ANALYTICAL_QUERY"
    SCHEMA_QUESTION = "SCHEMA_QUESTION"
    DATA_LOOKUP = "DATA_LOOKUP"
    AGGREGATION = "AGGREGATION"
    COMPARISON = "COMPARISON"
    TREND_ANALYSIS = "TREND_ANALYSIS"
    RANKING = "RANKING"
    SUMMARY = "SUMMARY"
    UNKNOWN = "UNKNOWN"
    UNSUPPORTED = "UNSUPPORTED"


class AIOperationType(str, enum.Enum):
    SELECT = "SELECT"
    FILTER = "FILTER"
    AGGREGATE = "AGGREGATE"
    GROUP_BY = "GROUP_BY"
    SORT = "SORT"
    RANK = "RANK"
    COMPARE = "COMPARE"
    TREND = "TREND"
    SUMMARY = "SUMMARY"
    COUNT = "COUNT"
    DISTINCT = "DISTINCT"


class AIPlanOperation(str, enum.Enum):
    SELECT = "SELECT"
    FILTER = "FILTER"
    FILTER_TIME = "FILTER_TIME"
    AGGREGATE = "AGGREGATE"
    GROUP_BY = "GROUP_BY"
    SORT = "SORT"
    RANK = "RANK"
    COMPARE = "COMPARE"
    TREND = "TREND"
    SUMMARY = "SUMMARY"
    COUNT = "COUNT"
    DISTINCT = "DISTINCT"
    METADATA_LOOKUP = "METADATA_LOOKUP"


class FilterOperator(str, enum.Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    IN = "in"
    BETWEEN = "between"


class TimeRangePreset(str, enum.Enum):
    TODAY = "today"
    YESTERDAY = "yesterday"
    THIS_WEEK = "this_week"
    LAST_WEEK = "last_week"
    THIS_MONTH = "this_month"
    LAST_MONTH = "last_month"
    THIS_QUARTER = "this_quarter"
    LAST_QUARTER = "last_quarter"
    THIS_YEAR = "this_year"
    LAST_YEAR = "last_year"
    CUSTOM_RANGE = "custom_range"


class AggregationType(str, enum.Enum):
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    COUNT = "COUNT"
    COUNT_DISTINCT = "COUNT_DISTINCT"
    NONE = "NONE"


class AIConfidence(str, enum.Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class AIPlanCapability(str, enum.Enum):
    METADATA = "METADATA"
    DATABASE = "DATABASE"
    SAMPLE_DATA = "SAMPLE_DATA"
    AGGREGATION = "AGGREGATION"
    TIME_FILTER = "TIME_FILTER"
    RELATIONSHIPS = "RELATIONSHIPS"


class SortDirection(str, enum.Enum):
    ASC = "asc"
    DESC = "desc"


def _normalize_enum_token(value: Any, *, upper: bool) -> Any:
    if not isinstance(value, str):
        return value
    token = value.strip().replace(" ", "_").replace("-", "_")
    if not token:
        return value
    return token.upper() if upper else token.lower()


def _looks_like_sql(value: str) -> bool:
    lowered = value.lower()
    if any(marker in value for marker in (";", "--", "/*", "*/", "::")):
        return True
    for token in (
        " select ",
        " insert ",
        " update ",
        " delete ",
        " drop ",
        " alter ",
        " truncate ",
        " union ",
        " grant ",
        " revoke ",
        " create ",
    ):
        if token in f" {lowered} ":
            return True
    return False


def validate_concept_name(value: str, *, field_name: str) -> str:
    cleaned = " ".join(value.strip().split())
    max_chars = settings.AI_MAX_CONCEPT_CHARS
    if not cleaned:
        raise ValueError(f"{field_name} is required")
    if len(cleaned) > max_chars:
        raise ValueError(f"{field_name} exceeds maximum length of {max_chars}")
    if _looks_like_sql(cleaned):
        raise ValueError(f"{field_name} must not contain SQL")
    return cleaned


class AIMetric(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1, max_length=512)
    aggregation: AggregationType = AggregationType.NONE

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return validate_concept_name(value, field_name="metric")

    @field_validator("aggregation", mode="before")
    @classmethod
    def normalize_aggregation(cls, value: Any) -> Any:
        return _normalize_enum_token(value, upper=True)


class AIDimension(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1, max_length=512)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return validate_concept_name(value, field_name="dimension")


class AIFilter(BaseModel):
    """Structured filter. Operators are controlled; values are not SQL."""

    model_config = ConfigDict(extra="ignore")

    field: str = Field(min_length=1, max_length=512)
    operator: FilterOperator
    value: FilterScalar | None = None
    values: list[FilterScalar] | None = None
    start: FilterScalar | None = None
    end: FilterScalar | None = None

    @field_validator("field")
    @classmethod
    def validate_field(cls, value: str) -> str:
        return validate_concept_name(value, field_name="filter field")

    @field_validator("operator", mode="before")
    @classmethod
    def normalize_operator(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        token = _normalize_enum_token(value, upper=False)
        aliases = {
            "eq": FilterOperator.EQUALS.value,
            "=": FilterOperator.EQUALS.value,
            "==": FilterOperator.EQUALS.value,
            "ne": FilterOperator.NOT_EQUALS.value,
            "neq": FilterOperator.NOT_EQUALS.value,
            "!=": FilterOperator.NOT_EQUALS.value,
            "<>": FilterOperator.NOT_EQUALS.value,
            "gt": FilterOperator.GREATER_THAN.value,
            ">": FilterOperator.GREATER_THAN.value,
            "lt": FilterOperator.LESS_THAN.value,
            "<": FilterOperator.LESS_THAN.value,
            "gte": FilterOperator.GREATER_THAN_OR_EQUAL.value,
            ">=": FilterOperator.GREATER_THAN_OR_EQUAL.value,
            "lte": FilterOperator.LESS_THAN_OR_EQUAL.value,
            "<=": FilterOperator.LESS_THAN_OR_EQUAL.value,
            "not_equal": FilterOperator.NOT_EQUALS.value,
            "like": FilterOperator.CONTAINS.value,
            "ilike": FilterOperator.CONTAINS.value,
        }
        if token in aliases:
            return aliases[token]
        return token

    @field_validator("value", "start", "end", mode="before")
    @classmethod
    def reject_non_scalars(cls, value: Any) -> Any:
        if isinstance(value, (dict, list)):
            raise TypeError("Filter scalars cannot be objects or lists")
        if isinstance(value, str) and _looks_like_sql(value):
            raise ValueError("Filter values must not contain SQL")
        return value

    @field_validator("values", mode="before")
    @classmethod
    def validate_values(cls, value: Any) -> Any:
        if value is None:
            return value
        if not isinstance(value, list):
            raise TypeError("Filter values must be a list")
        if len(value) > _MAX_LLM_COLLECTION:
            raise ValueError("Too many filter values")
        for item in value:
            if isinstance(item, (dict, list)):
                raise TypeError("Filter values must be scalars")
            if isinstance(item, str) and _looks_like_sql(item):
                raise ValueError("Filter values must not contain SQL")
        return value

    @model_validator(mode="before")
    @classmethod
    def promote_list_value(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        operator = data.get("operator")
        if isinstance(operator, str) and operator.strip().lower() == "in":
            raw_value = data.get("value")
            if data.get("values") is None and isinstance(raw_value, list):
                data = {**data, "values": raw_value, "value": None}
        return data

    @model_validator(mode="after")
    def validate_operator_payload(self) -> AIFilter:
        max_values = settings.AI_MAX_FILTER_VALUES
        if self.operator is FilterOperator.IN:
            if not self.values:
                raise ValueError("Operator 'in' requires values")
            if len(self.values) > max_values:
                raise ValueError("Too many filter values")
            return self
        if self.operator is FilterOperator.BETWEEN:
            if self.start is None or self.end is None:
                raise ValueError("Operator 'between' requires start and end")
            return self
        if self.value is None:
            raise ValueError(f"Operator '{self.operator.value}' requires a value")
        return self


class AITimeRange(BaseModel):
    model_config = ConfigDict(extra="ignore")

    preset: TimeRangePreset
    start_date: date | None = None
    end_date: date | None = None

    @field_validator("preset", mode="before")
    @classmethod
    def normalize_preset(cls, value: Any) -> Any:
        return _normalize_enum_token(value, upper=False)

    @model_validator(mode="after")
    def validate_custom_range(self) -> AITimeRange:
        if self.preset is TimeRangePreset.CUSTOM_RANGE:
            if self.start_date is None or self.end_date is None:
                raise ValueError("custom_range requires start_date and end_date")
            if self.start_date > self.end_date:
                raise ValueError("custom_range start_date cannot be after end_date")
            return self
        if self.start_date is not None or self.end_date is not None:
            raise ValueError("Named time ranges cannot include custom dates")
        return self


class AISort(BaseModel):
    model_config = ConfigDict(extra="ignore")

    field: str = Field(min_length=1, max_length=512)
    direction: SortDirection = SortDirection.DESC

    @field_validator("field")
    @classmethod
    def validate_field(cls, value: str) -> str:
        return validate_concept_name(value, field_name="sort field")

    @field_validator("direction", mode="before")
    @classmethod
    def normalize_direction(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        token = _normalize_enum_token(value, upper=False)
        aliases = {
            "ascending": SortDirection.ASC.value,
            "descending": SortDirection.DESC.value,
            "ascend": SortDirection.ASC.value,
            "descend": SortDirection.DESC.value,
        }
        if token in aliases:
            return aliases[token]
        return token


def coerce_llm_intent_payload(data: Any) -> Any:
    """Normalize common LLM shape mistakes before strict intent validation."""
    if not isinstance(data, dict):
        return data
    payload = dict(data)

    intent = payload.get("intent")
    if isinstance(intent, str):
        token = _normalize_enum_token(intent, upper=True)
        known = {item.value for item in AIIntentType}
        aliases = {
            "REPORT": AIIntentType.SUMMARY.value,
            "ANALYZE": AIIntentType.ANALYTICAL_QUERY.value,
            "ANALYSIS": AIIntentType.ANALYTICAL_QUERY.value,
            "QUERY": AIIntentType.ANALYTICAL_QUERY.value,
            "LOOKUP": AIIntentType.DATA_LOOKUP.value,
            "SCHEMA": AIIntentType.SCHEMA_QUESTION.value,
            "META": AIIntentType.SCHEMA_QUESTION.value,
            "METADATA": AIIntentType.SCHEMA_QUESTION.value,
        }
        if token in aliases:
            payload["intent"] = aliases[token]
        elif token not in known:
            payload["intent"] = AIIntentType.UNKNOWN.value
            payload["requires_clarification"] = True
            if not payload.get("clarification_question"):
                payload["clarification_question"] = (
                    "What would you like to know — a total, a count, "
                    "a ranking, or a breakdown?"
                )

    operation = payload.get("operation")
    if isinstance(operation, str):
        op_token = _normalize_enum_token(operation, upper=True)
        known_ops = {item.value for item in AIOperationType}
        if op_token not in known_ops:
            payload["operation"] = None
    elif operation == "":
        payload["operation"] = None

    clarification = payload.get("clarification_question")
    if isinstance(clarification, str) and _looks_like_sql(clarification):
        payload["clarification_question"] = (
            "What would you like to know — a total, a count, "
            "a ranking, or a breakdown?"
        )
        payload["requires_clarification"] = True

    metrics = payload.get("metrics")
    if isinstance(metrics, list):
        coerced_metrics: list[Any] = []
        for item in metrics[:_MAX_LLM_COLLECTION]:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    coerced_metrics.append(
                        {"name": cleaned, "aggregation": AggregationType.NONE.value}
                    )
            elif isinstance(item, (dict, AIMetric)):
                coerced_metrics.append(item)
        payload["metrics"] = coerced_metrics
    elif metrics is None:
        payload["metrics"] = []

    dimensions = payload.get("dimensions")
    if isinstance(dimensions, list):
        coerced_dimensions: list[Any] = []
        for item in dimensions[:_MAX_LLM_COLLECTION]:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    coerced_dimensions.append({"name": cleaned})
            elif isinstance(item, (dict, AIDimension)):
                coerced_dimensions.append(item)
        payload["dimensions"] = coerced_dimensions
    elif dimensions is None:
        payload["dimensions"] = []

    filters = payload.get("filters")
    if isinstance(filters, list):
        coerced_filters: list[Any] = []
        dropped = False
        for item in filters[:_MAX_LLM_COLLECTION]:
            if isinstance(item, AIFilter):
                coerced_filters.append(item)
                continue
            if not isinstance(item, dict):
                dropped = True
                continue
            try:
                AIFilter.model_validate(item)
            except (ValidationError, TypeError, ValueError):
                dropped = True
                continue
            coerced_filters.append(item)
        payload["filters"] = coerced_filters
        if dropped:
            payload["requires_clarification"] = True
            if not payload.get("clarification_question"):
                payload["clarification_question"] = (
                    "What filters or metrics should this report include?"
                )
    elif filters is None:
        payload["filters"] = []

    time_range = payload.get("time_range")
    if isinstance(time_range, (dict, AITimeRange)):
        try:
            if isinstance(time_range, AITimeRange):
                AITimeRange.model_validate(time_range.model_dump())
            else:
                AITimeRange.model_validate(time_range)
        except (ValidationError, TypeError, ValueError):
            payload["time_range"] = None
            payload["requires_clarification"] = True
    elif time_range in ("", None):
        payload["time_range"] = None

    sort = payload.get("sort")
    if isinstance(sort, (dict, AISort)):
        try:
            if isinstance(sort, AISort):
                AISort.model_validate(sort.model_dump())
            else:
                AISort.model_validate(sort)
        except (ValidationError, TypeError, ValueError):
            payload["sort"] = None
    elif sort in ("", None):
        payload["sort"] = None

    for flag in (
        "requires_data_access",
        "requires_metadata",
        "requires_relationships",
        "requires_clarification",
    ):
        value = payload.get(flag)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1"}:
                payload[flag] = True
            elif lowered in {"false", "no", "0"}:
                payload[flag] = False

    return payload


class LLMIntentDetection(BaseModel):
    """Structured LLM output for intent detection. Validated before planning."""

    model_config = ConfigDict(extra="ignore")

    intent: AIIntentType
    operation: AIOperationType | None = None
    subject: str | None = None
    metrics: list[AIMetric] = Field(
        default_factory=list, max_length=_MAX_LLM_COLLECTION
    )
    dimensions: list[AIDimension] = Field(
        default_factory=list, max_length=_MAX_LLM_COLLECTION
    )
    filters: list[AIFilter] = Field(
        default_factory=list, max_length=_MAX_LLM_COLLECTION
    )
    time_range: AITimeRange | None = None
    sort: AISort | None = None
    requested_limit: int | None = Field(default=None, ge=1, le=10_000_000)
    requires_data_access: bool = False
    requires_metadata: bool = False
    requires_relationships: bool = False
    confidence: AIConfidence = AIConfidence.MEDIUM
    requires_clarification: bool = False
    clarification_question: str | None = Field(
        default=None, max_length=_MAX_CLARIFICATION_CHARS
    )
    unsupported_reason: str | None = Field(
        default=None, max_length=_MAX_UNSUPPORTED_REASON_CHARS
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_payload(cls, data: Any) -> Any:
        return coerce_llm_intent_payload(data)

    @field_validator("intent", mode="before")
    @classmethod
    def normalize_intent(cls, value: Any) -> Any:
        return _normalize_enum_token(value, upper=True)

    @field_validator("operation", mode="before")
    @classmethod
    def normalize_operation(cls, value: Any) -> Any:
        if value is None or value == "":
            return None
        return _normalize_enum_token(value, upper=True)

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: Any) -> Any:
        return _normalize_enum_token(value, upper=True)

    @field_validator("subject")
    @classmethod
    def validate_subject(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        return validate_concept_name(cleaned, field_name="subject")

    @field_validator("clarification_question", "unsupported_reason")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(value.strip().split())
        if cleaned and _looks_like_sql(cleaned):
            raise ValueError("Text fields must not contain SQL")
        return cleaned or None


class AIIntent(BaseModel):
    """Normalized analytical intent. Planning concepts only; not executable."""

    model_config = ConfigDict(extra="forbid")

    intent: AIIntentType
    operation: AIOperationType | None = None
    subject: str | None = None
    metrics: list[AIMetric] = Field(default_factory=list)
    dimensions: list[AIDimension] = Field(default_factory=list)
    filters: list[AIFilter] = Field(default_factory=list)
    time_range: AITimeRange | None = None
    sort: AISort | None = None
    requested_limit: int | None = Field(default=None, ge=1, le=10_000_000)
    safe_limit: int | None = Field(default=None, ge=1)
    exceeds_limit: bool = False
    requires_data_access: bool = False
    requires_metadata: bool = False
    requires_relationships: bool = False
    confidence: AIConfidence = AIConfidence.MEDIUM
    requires_clarification: bool = False
    clarification_question: str | None = None
    unsupported_reason: str | None = None


class AIRequestPlan(BaseModel):
    """Structured execution plan. Never executed in this phase."""

    model_config = ConfigDict(extra="forbid")

    intent: AIIntentType
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
