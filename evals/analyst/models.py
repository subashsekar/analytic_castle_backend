"""Evaluation case and result models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CASES_PATH = Path(__file__).with_name("cases.json")

Behavior = Literal["answer", "clarify", "refuse"]
CheckKind = Literal["scalar", "top_k", "breakdown", "series", "nonempty", "none"]


class Expected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    behavior: Behavior
    intents: list[str] = Field(default_factory=list)
    question_types: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    check: CheckKind = "none"
    top_k: int = Field(default=1, ge=1, le=20)
    reference_sql: str | None = None
    alternative_sql: list[str] = Field(default_factory=list)
    must_include_any: list[str] = Field(default_factory=list)
    forbidden_terms: list[str] = Field(default_factory=list)


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    question: str
    setup: list[str] = Field(default_factory=list)
    seed_sql: str | None = None
    expected: Expected
    criteria: str

    def render(self, seed: str | None) -> tuple[str, list[str]]:
        """Question and reference SQLs with the seed value filled in."""
        literal = "'" + (seed or "").replace("'", "''") + "'"
        question = self.question.replace("{seed}", seed or "")
        sqls = [
            sql.replace("{seed_sql}", literal)
            for sql in (self.expected.reference_sql, *self.expected.alternative_sql)
            if sql
        ]
        return question, sqls


class QueryData(BaseModel):
    """Tabular result (reference query or the analyst's query preview)."""

    model_config = ConfigDict(extra="forbid")

    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool | None  # None = not applicable
    detail: str = ""


class CaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    question: str
    checks: list[CheckResult]
    answer: str = ""
    sql: str | None = None
    error: str | None = None
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    @property
    def applicable(self) -> list[CheckResult]:
        return [check for check in self.checks if check.passed is not None]

    @property
    def score(self) -> float:
        applicable = self.applicable
        if not applicable:
            return 0.0
        return sum(1 for check in applicable if check.passed) / len(applicable)

    @property
    def passed(self) -> bool:
        return self.error is None and all(check.passed for check in self.applicable)


def load_cases(path: Path = CASES_PATH) -> list[EvalCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [EvalCase.model_validate(item) for item in data["cases"]]
