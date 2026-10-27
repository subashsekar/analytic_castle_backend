from __future__ import annotations

from app.ai.result_facts import ResultFacts
from app.ai.sql_execution.models import SQLExecutionResult, SQLExecutionStatus
from app.ai.visualization_agent import build_chart_spec
from app.schemas.charts import ChartSpec


def test_build_chart_spec_selects_scatter_when_two_numeric_columns() -> None:
    query_result = SQLExecutionResult(
        status=SQLExecutionStatus.SUCCEEDED,
        columns=["x", "y"],
        rows=[[1, 10.0], [2, 20.0], [3, 30.0]],
        row_count=3,
        truncated=False,
        duration_ms=0.0,
        applied_row_limit=None,
        sql_char_count=None,
        referenced_table_count=0,
        referenced_column_count=0,
    )

    spec = build_chart_spec(message="plot relationship", query_result=query_result, facts=None)
    validated = ChartSpec.model_validate(spec)
    assert validated.type == "scatter"


def test_build_chart_spec_selects_pie_from_categorical_facts_and_composition_message() -> None:
    query_result = SQLExecutionResult(
        status=SQLExecutionStatus.SUCCEEDED,
        columns=["category", "value"],
        rows=[["A", 10.0], ["B", 20.0], ["C", 30.0]],
        row_count=3,
        truncated=False,
        duration_ms=0.0,
        applied_row_limit=None,
        sql_char_count=None,
        referenced_table_count=0,
        referenced_column_count=0,
    )

    facts = ResultFacts(shape="categorical", headline=None)
    spec = build_chart_spec(
        message="distribution share of categories",
        query_result=query_result,
        facts=facts,
    )
    validated = ChartSpec.model_validate(spec)
    assert validated.type == "pie"

