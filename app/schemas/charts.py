from __future__ import annotations

from typing import Any, Annotated, Literal, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ai.sql_execution.models import SQLExecutionResult, SQLExecutionStatus


ChartType = Literal["bar", "line", "pie", "scatter", "table"]


class _ChartSpecBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: ChartType
    title: str | None = None


class LineChartSpec(_ChartSpecBase):
    type: Literal["line"] = "line"
    x_axis: str | None = None
    y_axis: str | None = None
    x: list[Any] = Field(default_factory=list)
    y: list[float] = Field(default_factory=list)
    truncated: bool = False


class BarChartSpec(_ChartSpecBase):
    type: Literal["bar"] = "bar"
    x_axis: str | None = None
    y_axis: str | None = None
    x: list[Any] = Field(default_factory=list)
    y: list[float] = Field(default_factory=list)
    truncated: bool = False


class PieChartSpec(_ChartSpecBase):
    type: Literal["pie"] = "pie"
    x_axis: str | None = None
    y_axis: str | None = None
    labels: list[Any] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)
    truncated: bool = False


class ScatterChartSpec(_ChartSpecBase):
    type: Literal["scatter"] = "scatter"
    x_axis: str | None = None
    y_axis: str | None = None
    x: list[float] = Field(default_factory=list)
    y: list[float] = Field(default_factory=list)
    truncated: bool = False


class TableChartSpec(_ChartSpecBase):
    type: Literal["table"] = "table"
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    truncated: bool = False
    empty: bool = False


ChartSpec = Annotated[
    Union[
        LineChartSpec,
        BarChartSpec,
        PieChartSpec,
        ScatterChartSpec,
        TableChartSpec,
    ],
    Field(discriminator="type"),
]


class ChartSpecificationRequest(BaseModel):
    """Request chart-spec generation from already-executed SQL results."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=32_000)
    query_result: SQLExecutionResult

    @field_validator("query_result")
    @classmethod
    def validate_query_succeeded(
        cls, value: SQLExecutionResult
    ) -> SQLExecutionResult:
        if value.status is not SQLExecutionStatus.SUCCEEDED:
            raise ValueError("query_result.status must be SUCCEEDED")
        return value


class ChartSpecificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    workspace_id: UUID
    data_source_id: UUID
    chart_spec: ChartSpec


class ChartReadyDataRequest(BaseModel):
    """Return chart-ready data derived from a validated chart specification."""

    model_config = ConfigDict(extra="forbid")

    chart_spec: ChartSpec


class ChartReadyDataResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    workspace_id: UUID
    data_source_id: UUID
    chart_ready_data: ChartSpec

