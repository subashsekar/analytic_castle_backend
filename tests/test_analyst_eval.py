"""Analyst evaluation set and scorer. Chat and query functions are mocked: no LLM, no database."""

from __future__ import annotations

from typing import Any

from evals.analyst.models import EvalCase, QueryData, load_cases
from evals.analyst.runner import evaluate, summarize, validate_dataset
from evals.analyst.scoring import extract_numbers, result_matches, score_case, unsupported_claims

CASES = {case.id: case for case in load_cases()}
SQL = "SELECT COUNT(*) AS sales_count FROM public.sales WHERE sale_date >= DATE '2025-03-01'"


def _response(
    answer: str,
    *,
    sql: str | None = SQL,
    columns: list[str] | None = None,
    rows: list[list[Any]] | None = None,
    intent: str = "AGGREGATION",
    question_types: list[str] | None = None,
    clarify: bool = False,
    unsupported: bool = False,
) -> dict[str, Any]:
    analysis = None
    if sql is not None:
        analysis = {
            "sql": sql,
            "question_types": question_types or [],
            "query_preview": {"columns": columns or [], "sample_rows": rows or []},
        }
    return {
        "response": answer,
        "intent": {"type": intent},
        "plan": {"requires_clarification": clarify, "unsupported": unsupported},
        "analysis": analysis,
        "conversation_id": "c-1",
        "conversation_version": 2,
    }


def _check(result, name: str) -> bool | None:
    return next(c.passed for c in result.checks if c.name == name)


def test_dataset_has_50_valid_cases_covering_all_categories() -> None:
    cases = list(CASES.values())
    assert validate_dataset(cases) == []
    assert {case.category for case in cases} == {
        "counts_aggregations",
        "filters_dates",
        "rankings_breakdowns",
        "trends_comparisons",
        "anomalies_root_cause",
        "insights_recommendations",
        "ambiguous_followups",
    }
    assert {case.expected.behavior for case in cases} == {"answer", "clarify", "refuse"}
    assert any(case.setup for case in cases) and any(case.seed_sql for case in cases)


def test_correct_scalar_answer_passes_every_check() -> None:
    reference = QueryData(columns=["sales_count"], rows=[[42]])
    result = score_case(
        CASES["flt_01"],
        _response("42 sales were made in March 2025.", columns=["count"], rows=[[42]]),
        [reference],
    )
    assert result.passed, result.checks
    assert result.score == 1.0


def test_wrong_value_and_invented_number_fail() -> None:
    reference = QueryData(columns=["sales_count"], rows=[[42]])
    result = score_case(
        CASES["flt_01"],
        _response("There were 40 sales, 12,500 fewer than usual.", columns=["n"], rows=[[40]]),
        [reference],
    )
    assert _check(result, "result") is False
    assert _check(result, "answer_accuracy") is False
    assert _check(result, "unsupported_claims") is False
    assert not result.passed


def test_alternative_reference_is_accepted() -> None:
    references = [
        QueryData(columns=["sales_count"], rows=[[42]]),
        QueryData(columns=["total_orders"], rows=[[118]]),
    ]
    result = score_case(
        CASES["flt_01"],
        _response("118 orders were placed in March 2025.", columns=["orders"], rows=[[118]]),
        references,
    )
    assert _check(result, "result") is True


def test_schema_grounding_requires_needed_columns() -> None:
    reference = QueryData(columns=["total_revenue"], rows=[[1000]])
    result = score_case(
        CASES["agg_02"],
        _response("Total is 1,000.", sql="SELECT SUM(units) FROM public.sales", columns=["s"], rows=[[1000]]),
        [reference],
    )
    assert _check(result, "schema_grounding") is False


def test_series_matches_across_period_formats_and_derived_numbers_are_supported() -> None:
    reference = QueryData(
        columns=["period", "total_revenue"],
        rows=[["2025-02-01T00:00:00", 15000], ["2025-03-01T00:00:00", 7000]],
    )
    actual = QueryData(columns=["month", "revenue"], rows=[["2025-02", 15000], ["2025-03", 7000]])
    assert result_matches("series", actual, reference)
    answer = "Revenue fell from 15,000 to 7,000 in March 2025, down 8,000 (-53.3%)."
    assert unsupported_claims(answer, [reference]) == []
    result = score_case(
        CASES["trd_04"],
        _response(answer, sql="SELECT sale_date, revenue FROM public.sales", columns=actual.columns,
                  rows=actual.rows, intent="COMPARISON"),
        [reference],
    )
    assert _check(result, "result") and _check(result, "answer_accuracy")
    assert _check(result, "unsupported_claims")


def test_top_k_names_labels_and_tolerates_ties() -> None:
    reference = QueryData(columns=["region", "total"], rows=[["West", 90], ["East", 90], ["North", 10]])
    tied = QueryData(columns=["region", "total"], rows=[["East", 90], ["West", 90]])
    assert result_matches("top_k", tied, reference, top_k=2)
    wrong = QueryData(columns=["region", "total"], rows=[["North", 10]])
    assert not result_matches("top_k", wrong, reference, top_k=1)
    result = score_case(
        CASES["rnk_01"],
        _response("West had the highest revenue (90).", sql="SELECT region, revenue FROM public.sales",
                  columns=reference.columns, rows=reference.rows, intent="RANKING"),
        [reference],
    )
    assert result.passed, result.checks


def test_clarify_and_refuse_behaviors() -> None:
    clarified = score_case(
        CASES["amb_02"],
        _response("There is no profit field. Which field should I use?", sql=None, clarify=True),
        [],
    )
    assert clarified.passed, clarified.checks
    substituted = score_case(
        CASES["amb_02"],
        _response("Profit margin was 12,000 in West.", columns=["region", "revenue"], rows=[["West", 12000]]),
        [],
    )
    assert _check(substituted, "behavior") is False
    refused = score_case(
        CASES["ref_01"],
        _response("I can only read data.", sql=None, intent="UNSUPPORTED", unsupported=True),
        [],
    )
    assert refused.passed, refused.checks


def test_number_extraction_ignores_dates_years_and_small_counts() -> None:
    values = [v for v, _pct, _abbr in extract_numbers("On 2025-03-01, top 5 regions had 1.2k sales (12%).")]
    assert values == [5.0, 1200.0, 12.0]
    assert unsupported_claims("In 2025 the top 3 regions ...", []) == []


def test_evaluate_runs_follow_ups_seeds_and_isolates_errors() -> None:
    cases = [CASES["fu_03"], CASES["flt_04"], CASES["agg_02"]]
    chat_calls: list[tuple[str, dict[str, Any] | None]] = []
    queries: list[str] = []

    def query(sql: str) -> QueryData:
        queries.append(sql)
        if "GROUP BY region ORDER BY" in sql and "LIMIT 1" in sql:
            return QueryData(columns=["region"], rows=[["O'Hara"]])
        if "SUM(revenue)" in sql and "region =" in sql:
            return QueryData(columns=["total_revenue"], rows=[[500]])
        if sql.startswith("SELECT SUM(revenue) AS total_revenue FROM public.sales"):
            raise RuntimeError("reference unavailable")
        return QueryData(columns=["n"], rows=[[7]])

    def chat(message: str, conversation: dict[str, Any] | None) -> dict[str, Any]:
        chat_calls.append((message, conversation))
        if "region" in message:
            return _response("Revenue was 500.", sql="SELECT SUM(revenue) FROM public.sales WHERE region = 'x'",
                             columns=["total"], rows=[[500]])
        return _response("7 sales in February.", sql=SQL.replace("03", "02"), columns=["n"], rows=[[7]])

    results = evaluate(cases, chat=chat, query=query, log=lambda _line: None)
    assert chat_calls[0] == ("How many sales were made in March 2025?", None)
    assert chat_calls[1] == ("And in February?", {"conversation_id": "c-1", "conversation_version": 2})
    assert chat_calls[2][0] == "What was the total revenue in the O'Hara region?"
    assert any("region = 'O''Hara'" in sql for sql in queries)
    assert results[0].passed and results[1].passed, (results[0].checks, results[1].checks)
    assert results[2].error and "reference unavailable" in results[2].error

    summary = summarize(results)
    assert summary["cases"] == 3 and summary["errors"] == 1 and summary["passed"] == 2
    assert 0 < summary["mean_score"] < 1


def test_one_case_render_keeps_sql_literal_safe() -> None:
    case = EvalCase.model_validate(
        {
            "id": "x",
            "category": "filters_dates",
            "seed_sql": "SELECT region FROM public.sales LIMIT 1",
            "question": "Revenue in {seed}?",
            "expected": {"behavior": "answer", "check": "scalar",
                         "reference_sql": "SELECT SUM(revenue) FROM public.sales WHERE region = {seed_sql}"},
            "criteria": "c",
        }
    )
    question, sqls = case.render("a'; DROP TABLE x; --")
    assert question == "Revenue in a'; DROP TABLE x; --?"
    assert sqls == ["SELECT SUM(revenue) FROM public.sales WHERE region = 'a''; DROP TABLE x; --'"]
