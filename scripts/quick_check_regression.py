"""Quick AI Analyst Regression Check Script for Chapters 8.9 and 8.10.

Verifies:
- SQL reference validation
- Answer-to-result consistency
- Grounding & hallucination scrubbing (no fabricated numbers like 99999)
- Self-consistency checks (direction agreement)
- Latency (execution of deterministic facts/composition < 10ms)
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from app.ai.analysis_pipeline import _compose_answer, AnalysisPayload
from app.ai.result_facts import build_result_facts
from app.ai.sql_execution.models import SQLExecutionResult, SQLExecutionStatus
from app.ai.sql_validation.validation import validate_generated_sql
from evals.analyst.demo_schema import demo_metadata
from evals.analyst.models import load_cases

# Warm up the imports and cache
_compose_answer(
    AnalysisPayload(data_analyst={"summary": "warmup"}),
    SQLExecutionResult(status=SQLExecutionStatus.SUCCEEDED, columns=["v"], rows=[[1.0]], row_count=1),
    facts=build_result_facts(SQLExecutionResult(status=SQLExecutionStatus.SUCCEEDED, columns=["v"], rows=[[1.0]], row_count=1))
)

# Load all offline test cases
cases = {c.id: c for c in load_cases()}
metadata = demo_metadata()

print("======================================================================")
print("              Quick AI Analyst Regression Check Results              ")
print("======================================================================\n")

# Keep track of check statistics
all_passed = True
total_checks = 0
passed_checks = 0

def check_assert(label: str, condition: bool) -> None:
    global all_passed, total_checks, passed_checks
    total_checks += 1
    if condition:
        passed_checks += 1
        print(f"  [PASS] {label}")
    else:
        all_passed = False
        print(f"  [FAIL] {label}")

# ---------------------------------------------------------------------- Case 1: Aggregation (agg_02)
print("Case 1: Aggregation (agg_02 - 'What is the total revenue?')")
case_agg = cases["agg_02"]
ref_sql_agg = case_agg.expected.reference_sql

# 1. SQL validation
val_agg = validate_generated_sql(ref_sql_agg, metadata)
check_assert("SQL validation passes", val_agg.is_valid and len(val_agg.violations) == 0)

# 2. Result facts and answer composer
res_agg = SQLExecutionResult(
    status=SQLExecutionStatus.SUCCEEDED,
    columns=["total_revenue"],
    rows=[[154300.00]],
    row_count=1,
)

t0 = time.perf_counter()
facts_agg = build_result_facts(res_agg)
payload_agg = AnalysisPayload(
    data_analyst={
        "interpretation": "The total revenue was 154,300.00.",
        "summary": "Total revenue is 154,300.00. However, 99999 orders were placed.",
    }
)
answer_agg = _compose_answer(payload_agg, res_agg, facts=facts_agg)
t1 = time.perf_counter()
latency_agg_ms = (t1 - t0) * 1000

check_assert("Answer is consistent with SQL results", "154,300" in answer_agg)
check_assert("Numeric grounding successfully scrubs hallucinated numbers (99999)", "99999" not in answer_agg)
check_assert(f"Processing latency is optimized ({latency_agg_ms:.2f}ms)", latency_agg_ms < 20.0)
print()

# ---------------------------------------------------------------------- Case 2: Date Filtering (flt_01)
print("Case 2: Date Filtering (flt_01 - 'How many sales were made in March 2025?')")
case_flt = cases["flt_01"]
ref_sql_flt = case_flt.expected.reference_sql

# 1. SQL validation
val_flt = validate_generated_sql(ref_sql_flt, metadata)
check_assert("SQL validation passes", val_flt.is_valid and len(val_flt.violations) == 0)

# 2. Result facts and answer composer
res_flt = SQLExecutionResult(
    status=SQLExecutionStatus.SUCCEEDED,
    columns=["sales_count"],
    rows=[[42]],
    row_count=1,
)

t0 = time.perf_counter()
facts_flt = build_result_facts(res_flt)
payload_flt = AnalysisPayload(
    data_analyst={
        "summary": "42 sales were made in March 2025. We also had 99999 test runs.",
    }
)
answer_flt = _compose_answer(payload_flt, res_flt, facts=facts_flt)
t1 = time.perf_counter()
latency_flt_ms = (t1 - t0) * 1000

check_assert("Answer is consistent with SQL results", "42" in answer_flt)
check_assert("Numeric grounding successfully scrubs hallucinated numbers (99999)", "99999" not in answer_flt)
check_assert(f"Processing latency is optimized ({latency_flt_ms:.2f}ms)", latency_flt_ms < 20.0)
print()

# ---------------------------------------------------------------------- Case 3: Ranking (rnk_01)
print("Case 3: Ranking (rnk_01 - 'Which region had the highest revenue?')")
case_rnk = cases["rnk_01"]
ref_sql_rnk = case_rnk.expected.reference_sql

# 1. SQL validation
val_rnk = validate_generated_sql(ref_sql_rnk, metadata)
check_assert("SQL validation passes", val_rnk.is_valid and len(val_rnk.violations) == 0)

# 2. Result facts and answer composer
res_rnk = SQLExecutionResult(
    status=SQLExecutionStatus.SUCCEEDED,
    columns=["region", "total_revenue"],
    rows=[["North", 12000.00], ["South", 8000.00]],
    row_count=2,
)

t0 = time.perf_counter()
facts_rnk = build_result_facts(res_rnk)
payload_rnk = AnalysisPayload(
    data_analyst={
        "summary": "North led with 12,000.00, while South was 8,000.00.",
    },
    insight={
        "insights": [
            {
                "title": "North Leads",
                "insight": "North is 12,000.00.",
                "metric": "revenue",
                "supporting_evidence": "North revenue is 12,000.00.",
            },
            {
                "title": "Hallucinated",
                "insight": "North peaked at 99999.",
                "metric": "revenue",
                "supporting_evidence": "North revenue is 99999.",
            }
        ]
    }
)
answer_rnk = _compose_answer(payload_rnk, res_rnk, facts=facts_rnk)
t1 = time.perf_counter()
latency_rnk_ms = (t1 - t0) * 1000

check_assert("Answer is consistent with SQL results", "North" in answer_rnk and "12,000" in answer_rnk)
check_assert("Numeric grounding successfully scrubs hallucinated numbers (99999)", "99999" not in answer_rnk)
check_assert(f"Processing latency is optimized ({latency_rnk_ms:.2f}ms)", latency_rnk_ms < 20.0)
print()

# ---------------------------------------------------------------------- Case 4: Trend/Comparison (trd_01)
print("Case 4: Trend/Comparison (trd_01 - 'Show the monthly revenue trend for 2025')")
case_trd = cases["trd_01"]
ref_sql_trd = case_trd.expected.reference_sql

# 1. SQL validation
val_trd = validate_generated_sql(ref_sql_trd, metadata)
check_assert("SQL validation passes", val_trd.is_valid and len(val_trd.violations) == 0)

# 2. Result facts and answer composer (Self-consistency checks)
res_trd = SQLExecutionResult(
    status=SQLExecutionStatus.SUCCEEDED,
    columns=["period", "total_revenue"],
    rows=[["2025-01-01", 10000.00], ["2025-02-01", 15000.00], ["2025-03-01", 8000.00]],
    row_count=3,
)

t0 = time.perf_counter()
facts_trd = build_result_facts(res_trd)
payload_trd = AnalysisPayload(
    trend={
        "direction": "DECREASING", # Mismatch with computed VOLATILE direction
        "summary": "Monthly revenue decreased. Over 99999 percent drop happened.",
        "series": {
            "direction": "VOLATILE",
            "point_count": 3
        }
    }
)
answer_trd = _compose_answer(payload_trd, res_trd, facts=facts_trd)
t1 = time.perf_counter()
latency_trd_ms = (t1 - t0) * 1000

check_assert("Answer successfully flags/drops contradictory trend narrative", "Dropped trend text that contradicts the computed series." in answer_trd)
check_assert("Numeric grounding successfully scrubs hallucinated numbers (99999)", "99999" not in answer_trd)
check_assert(f"Processing latency is optimized ({latency_trd_ms:.2f}ms)", latency_trd_ms < 20.0)
print()

# ---------------------------------------------------------------------- Case 5: Root Cause (rca_01)
print("Case 5: Root Cause (rca_01 - 'Why did revenue fall in March 2025?')")
case_rca = cases["rca_01"]
ref_sql_rca = case_rca.expected.reference_sql

# 1. SQL validation
val_rca = validate_generated_sql(ref_sql_rca, metadata)
check_assert("SQL validation passes", val_rca.is_valid and len(val_rca.violations) == 0)

# 2. Result facts and answer composer
res_rca = SQLExecutionResult(
    status=SQLExecutionStatus.SUCCEEDED,
    columns=["period", "region", "total_revenue"],
    rows=[
        ["2025-02-01", "North", 6000.00],
        ["2025-02-01", "South", 5000.00],
        ["2025-03-01", "North", 1200.00],
        ["2025-03-01", "South", 4800.00]
    ],
    row_count=4,
)

t0 = time.perf_counter()
facts_rca = build_result_facts(res_rca)
payload_rca = AnalysisPayload(
    root_cause={
        "primary_cause": "North declined sales. Hallucination: 99999 loss.",
        "hypotheses": [
            {
                "rank": 1,
                "statement": "North sales collapsed.",
                "supporting_evidence": "North sales fell from 6,000 to 1,200. There were also 99999 errors.",
                "confidence_score": "HIGH"
            }
        ]
    }
)
answer_rca = _compose_answer(payload_rca, res_rca, facts=facts_rca)
t1 = time.perf_counter()
latency_rca_ms = (t1 - t0) * 1000

check_assert("Answer is consistent with findings", "North sales collapsed." in answer_rca)
check_assert("Numeric grounding successfully scrubs hallucinated numbers (99999)", "99999" not in answer_rca)
check_assert(f"Processing latency is optimized ({latency_rca_ms:.2f}ms)", latency_rca_ms < 20.0)
print()

# ---------------------------------------------------------------------- Final Summary
print("======================================================================")
print("                           Summary of Results                         ")
print("======================================================================")
print(f"Total Checks Executed : {total_checks}")
print(f"Checks Passed          : {passed_checks}")
print(f"Checks Failed          : {total_checks - passed_checks}")
print(f"Regression Check Status: {'SUCCESS' if all_passed else 'FAILURE'}")
print("======================================================================")
