"""Deterministic scoring of one analyst response against reference results.

Checks (each pass / fail / not applicable):
- intent: response intent type or profiled question type is an expected one.
- behavior: answered with executed SQL, asked for clarification, or refused, as expected.
- sql_valid: executed SQL is read-only and valid against the demo schema.
- schema_grounding: the SQL references the columns the question needs.
- result: the analyst's result rows match a reference query result.
- answer_accuracy: the answer text states the key figure or label from the data.
- required_content: answer contains one of the expected phrases (e.g. hypotheses).
- unsupported_claims: every number in the answer is traceable to result/reference
  data or simple derivations of it, and no forbidden claim appears.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from itertools import combinations
from typing import Any

from app.ai.sql_validation.validation import validate_generated_sql
from app.ai.trend_analysis.series import parse_number, parse_period
from evals.analyst.demo_schema import demo_metadata
from evals.analyst.models import CaseResult, CheckResult, EvalCase, QueryData

REL_TOL = 0.005
ABBREVIATED_REL_TOL = 0.05
PCT_ABS_TOL = 0.6
_MAX_DERIVED_ROWS = 60

_DATE_TOKEN_RE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b")
_NUMBER_RE = re.compile(
    r"(?<![\w.])(?P<sign>[-+−]?)\$?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<suffix>\s?(?:%|[kKmM]\b|million\b|thousand\b))?"
)
_READ_ONLY_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


# ------------------------------------------------------------------ table shape


def _is_id(name: str) -> bool:
    lowered = name.lower()
    return lowered == "id" or lowered.endswith("_id")


def _numeric_columns(data: QueryData) -> list[int]:
    out = []
    for index, name in enumerate(data.columns):
        values = [row[index] for row in data.rows if index < len(row) and row[index] is not None]
        if values and not _is_id(name) and all(parse_number(v) is not None for v in values):
            if not all(isinstance(v, (date, datetime)) for v in values):
                out.append(index)
    return out


def _period_column(data: QueryData) -> int | None:
    for index in range(len(data.columns)):
        values = [row[index] for row in data.rows if index < len(row) and row[index] is not None]
        if values and all(
            isinstance(v, (date, datetime)) or (isinstance(v, str) and parse_period(v) is not None)
            for v in values
        ):
            return index
    return None


def _label_column(data: QueryData) -> int | None:
    numeric = set(_numeric_columns(data))
    for index, name in enumerate(data.columns):
        if index not in numeric or _is_id(name):
            return index
    return None


def _value_column(data: QueryData) -> int | None:
    numeric = _numeric_columns(data)
    return numeric[-1] if numeric else None


def _label(value: Any) -> str:
    number = parse_number(value)
    if number is not None and float(number).is_integer():
        return str(int(number))
    period = parse_period(value) if isinstance(value, (date, datetime, str)) else None
    if period is not None and not isinstance(value, str):
        return period.date().isoformat()
    return str(value).strip().lower()


def _period_key(value: Any) -> str | None:
    period = parse_period(value)
    return period.date().isoformat() if period else None


def _close(a: float, b: float, rel: float = REL_TOL) -> bool:
    return abs(a - b) <= max(rel * max(abs(a), abs(b)), 0.51)


# ------------------------------------------------------------------ result match


def _pairs(data: QueryData, key_col: int | None, keyer) -> dict[str, float] | None:
    value_col = _value_column(data)
    if key_col is None or value_col is None:
        return None
    out: dict[str, float] = {}
    for row in data.rows:
        key = keyer(row[key_col])
        number = parse_number(row[value_col])
        if key is None or number is None:
            return None
        out[key] = out.get(key, 0.0) + number
    return out


def _mapping_matches(actual: dict[str, float] | None, expected: dict[str, float] | None) -> bool:
    if actual is None or expected is None:
        return False
    if not expected:
        return not actual
    overlap = set(actual) & set(expected)
    needed = min(len(actual), len(expected))
    return len(overlap) >= max(needed, 1) and all(
        _close(actual[key], expected[key]) for key in overlap
    )


def result_matches(kind: str, actual: QueryData, expected: QueryData, top_k: int = 1) -> bool:
    if kind == "nonempty":
        return bool(actual.rows) or not expected.rows
    if kind == "scalar":
        value_col = _value_column(actual)
        ref_col = _value_column(expected)
        if len(actual.rows) != 1 or value_col is None or ref_col is None or not expected.rows:
            return False
        a, e = parse_number(actual.rows[0][value_col]), parse_number(expected.rows[0][ref_col])
        return a is not None and e is not None and _close(a, e)
    if kind == "series":
        return _mapping_matches(
            _pairs(actual, _period_column(actual), _period_key),
            _pairs(expected, _period_column(expected), _period_key),
        )
    if kind == "breakdown":
        return _mapping_matches(
            _pairs(actual, _label_column(actual), _label),
            _pairs(expected, _label_column(expected), _label),
        )
    if kind == "top_k":
        a_label, e_label = _label_column(actual), _label_column(expected)
        a_value, e_value = _value_column(actual), _value_column(expected)
        if None in (a_label, e_label) or len(actual.rows) < min(top_k, len(expected.rows)):
            return False
        k = min(top_k, len(expected.rows))
        labels_ok = [_label(r[a_label]) for r in actual.rows[:k]] == [
            _label(r[e_label]) for r in expected.rows[:k]
        ]
        if labels_ok or a_value is None or e_value is None:
            return labels_ok
        # Ties: same values in the same order are equally correct.
        return all(
            _close(parse_number(a[a_value]) or 0.0, parse_number(e[e_value]) or 0.0)
            for a, e in zip(actual.rows[:k], expected.rows[:k], strict=False)
        )
    return True


# ------------------------------------------------------------------ answer text


def extract_numbers(text: str) -> list[tuple[float, bool, bool]]:
    """(value, is_percent, is_abbreviated) for each number stated in the text."""
    cleaned = _DATE_TOKEN_RE.sub(" ", text)
    out = []
    for match in _NUMBER_RE.finditer(cleaned):
        value = float(match.group("num").replace(",", ""))
        if match.group("sign") in {"-", "−"}:
            value = -value
        suffix = (match.group("suffix") or "").strip().lower()
        scale = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6}.get(suffix, 1.0)
        out.append((value * scale, suffix == "%", scale != 1.0))
    return out


def _is_incidental(value: float, is_percent: bool) -> bool:
    if is_percent:
        return False
    if float(value).is_integer() and 1900 <= value <= 2100:
        return True  # years
    return float(value).is_integer() and abs(value) < 32  # counts like "top 5", days, quarters


def supported_numbers(tables: list[QueryData]) -> tuple[list[float], list[float]]:
    """Values and percentages traceable to the data via simple derivations."""
    values: list[float] = []
    percents: list[float] = []
    for data in tables:
        rows = data.rows[:_MAX_DERIVED_ROWS]
        numeric = _numeric_columns(QueryData(columns=data.columns, rows=rows))
        for col in numeric:
            column = [n for n in (parse_number(r[col]) for r in rows) if n is not None]
            if not column:
                continue
            total = sum(column)
            values.extend(column)
            values.extend([total, total / len(column), max(column), min(column)])
            if total:
                percents.extend(v / total * 100 for v in column)
            for a, b in combinations(column, 2):
                values.extend([b - a, a - b])
                if a:
                    percents.append((b - a) / a * 100)
                if b:
                    percents.append((a - b) / b * 100)
        for left, right in combinations(numeric, 2):
            for row in rows:
                a, b = parse_number(row[left]), parse_number(row[right])
                if a is not None and b:
                    values.append(a / b)
                    percents.append(a / b * 100)
                if b is not None and a:
                    values.append(b / a)
                    percents.append(b / a * 100)
    return values, percents


def unsupported_claims(
    text: str,
    tables: list[QueryData],
    *,
    allow_text: str = "",
) -> list[float]:
    values, percents = supported_numbers(tables)
    allowed = [v for v, _pct, _abbr in extract_numbers(allow_text)]
    bad = []
    for value, is_percent, abbreviated in extract_numbers(text):
        if _is_incidental(value, is_percent):
            continue
        if any(_close(abs(value), abs(allowed_value), REL_TOL) for allowed_value in allowed):
            continue
        if is_percent:
            ok = any(abs(abs(value) - abs(p)) <= PCT_ABS_TOL for p in percents)
        else:
            rel = ABBREVIATED_REL_TOL if abbreviated else REL_TOL
            ok = any(_close(abs(value), abs(v), rel) for v in values)
        if not ok:
            bad.append(value)
    return bad


def mentions_number(text: str, target: float) -> bool:
    return any(
        _close(abs(value), abs(target), ABBREVIATED_REL_TOL if abbreviated else REL_TOL)
        for value, is_percent, abbreviated in extract_numbers(text)
        if not is_percent
    )


def mentions_label(text: str, label: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(label)}(?![\w])", text, re.IGNORECASE) is not None


def _key_facts(kind: str, data: QueryData, top_k: int) -> tuple[list[float], list[str]]:
    """Figures and labels a correct answer must state."""
    value_col, label_col = _value_column(data), _label_column(data)
    if not data.rows or value_col is None:
        return [], []
    first = parse_number(data.rows[0][value_col])
    if kind == "scalar":
        return ([first] if first is not None else []), []
    if kind in {"top_k", "breakdown"} and label_col is not None:
        k = top_k if kind == "top_k" else 1
        return [], [str(_label(row[label_col])) for row in data.rows[:k]]
    if kind == "series":
        column = [n for n in (parse_number(r[value_col]) for r in data.rows) if n is not None]
        return column[-1:] + column[:1] + [sum(column), max(column), min(column)], []
    return [], []


# ------------------------------------------------------------------ case scoring


def _preview(analysis: dict[str, Any] | None) -> QueryData | None:
    preview = (analysis or {}).get("query_preview")
    if not preview:
        return None
    return QueryData(columns=preview.get("columns") or [], rows=preview.get("sample_rows") or [])


def score_case(
    case: EvalCase,
    response: dict[str, Any],
    references: list[QueryData],
    *,
    question: str | None = None,
) -> CaseResult:
    expected = case.expected
    answer = str(response.get("response") or "")
    intent = response.get("intent") or {}
    plan = response.get("plan") or {}
    analysis = response.get("analysis") or None
    sql = (analysis or {}).get("sql")
    preview = _preview(analysis)
    executed = bool(sql) and preview is not None
    checks: list[CheckResult] = []

    kinds = set((analysis or {}).get("question_types") or [])
    if expected.intents or expected.question_types:
        ok = intent.get("type") in expected.intents or bool(kinds & set(expected.question_types))
        checks.append(CheckResult(name="intent", passed=ok, detail=f"{intent.get('type')} {sorted(kinds)}"))

    if expected.behavior == "answer":
        behavior_ok = executed
    elif expected.behavior == "clarify":
        behavior_ok = not executed and not plan.get("unsupported")
    else:
        behavior_ok = not executed and bool(plan.get("unsupported"))
    checks.append(
        CheckResult(
            name="behavior",
            passed=behavior_ok,
            detail=f"expected {expected.behavior}; executed={executed} "
            f"clarify={plan.get('requires_clarification')} unsupported={plan.get('unsupported')}",
        )
    )

    answering = expected.behavior == "answer"
    if answering:
        valid = bool(sql) and bool(_READ_ONLY_RE.match(sql or "")) and executed
        if valid:
            validation = validate_generated_sql(sql, demo_metadata())
            valid = validation.is_valid
            detail = "; ".join(v.code.value for v in validation.violations[:3])
        else:
            detail = "no executed read-only SQL"
        checks.append(CheckResult(name="sql_valid", passed=valid, detail=detail))

    if answering and expected.columns:
        missing = [c for c in expected.columns if not re.search(rf"\b{c}\b", sql or "", re.I)]
        checks.append(
            CheckResult(name="schema_grounding", passed=not missing, detail=f"missing={missing}")
        )

    matched: QueryData | None = None
    if answering and expected.check not in {"none"} and references:
        if preview is not None:
            matched = next(
                (ref for ref in references if result_matches(expected.check, preview, ref, expected.top_k)),
                None,
            )
        checks.append(
            CheckResult(
                name="result",
                passed=matched is not None,
                detail=f"{expected.check} vs {len(references)} reference(s)",
            )
        )

    if answering and expected.check in {"scalar", "top_k", "breakdown", "series"} and references:
        figures, labels = _key_facts(expected.check, matched or references[0], expected.top_k)
        if figures or labels:
            figure_ok = not figures or (
                any(mentions_number(answer, f) for f in figures)
                if expected.check == "series"
                else all(mentions_number(answer, f) for f in figures)
            )
            label_ok = all(mentions_label(answer, label) for label in labels)
            checks.append(
                CheckResult(
                    name="answer_accuracy",
                    passed=figure_ok and label_ok,
                    detail=f"figures={[round(f, 2) for f in figures[:3]]} labels={labels[:5]}",
                )
            )

    if expected.must_include_any:
        lowered = answer.lower()
        checks.append(
            CheckResult(
                name="required_content",
                passed=any(term.lower() in lowered for term in expected.must_include_any),
                detail=f"any of {expected.must_include_any}",
            )
        )

    tables = [t for t in (preview, *references) if t is not None]
    bad = unsupported_claims(
        answer,
        tables,
        allow_text=f"{question or case.question}\n{sql or ''}",
    )
    forbidden = [t for t in expected.forbidden_terms if t.lower() in answer.lower()]
    checks.append(
        CheckResult(
            name="unsupported_claims",
            passed=not bad and not forbidden,
            detail=f"numbers={bad[:5]} forbidden={forbidden}",
        )
    )

    return CaseResult(
        id=case.id,
        category=case.category,
        question=question or case.question,
        checks=checks,
        answer=answer,
        sql=sql,
    )
