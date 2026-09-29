"""AI Analyst evaluation runner.

Offline (default, no network or LLM calls): validates the dataset and checks every
reference SQL against the demo schema with the production SQL validator.

Live (explicit ``--live``): asks each question through the running chat API
(real LLM calls), executes the reference SQL through the read-only MCP
``postgres.query`` tool as the given user, and scores the answers.

    python -m evals.analyst.runner
    python -m evals.analyst.runner --live --data-source-id <uuid> --user-id <uuid>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

from app.ai.sql_validation.validation import validate_generated_sql
from evals.analyst.demo_schema import demo_metadata
from evals.analyst.models import CaseResult, EvalCase, QueryData, load_cases
from evals.analyst.scoring import score_case

EXPECTED_CASE_COUNT = 50
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("results")

ChatFn = Callable[[str, dict[str, Any] | None], dict[str, Any]]
QueryFn = Callable[[str], QueryData]


# ------------------------------------------------------------------ offline


def validate_dataset(cases: list[EvalCase]) -> list[str]:
    problems: list[str] = []
    ids = Counter(case.id for case in cases)
    problems += [f"duplicate id {case_id}" for case_id, n in ids.items() if n > 1]
    if len(cases) != EXPECTED_CASE_COUNT:
        problems.append(f"expected {EXPECTED_CASE_COUNT} cases, found {len(cases)}")
    metadata = demo_metadata()
    for case in cases:
        expected = case.expected
        if expected.behavior == "answer" and not expected.reference_sql:
            problems.append(f"{case.id}: answer case without reference_sql")
        if expected.behavior != "answer" and expected.check != "none":
            problems.append(f"{case.id}: non-answer case must use check 'none'")
        if ("{seed}" in case.question) != bool(case.seed_sql):
            problems.append(f"{case.id}: {{seed}} placeholder and seed_sql must go together")
        _question, sqls = case.render("Sample")
        for sql in (*sqls, *([case.seed_sql] if case.seed_sql else [])):
            result = validate_generated_sql(sql, metadata)
            if not result.is_valid:
                codes = ", ".join(v.code.value for v in result.violations)
                problems.append(f"{case.id}: invalid reference SQL ({codes})")
    return problems


# ------------------------------------------------------------------ evaluation


def _usage_tokens(response: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    usage = response.get("usage") or {}
    return usage.get("input_tokens"), usage.get("output_tokens"), usage.get("total_tokens")


def evaluate(
    cases: list[EvalCase],
    *,
    chat: ChatFn,
    query: QueryFn,
    delay_seconds: float = 0.0,
    log: Callable[[str], None] = print,
) -> list[CaseResult]:
    """Run cases with injected chat/query functions (mocked in unit tests)."""
    results: list[CaseResult] = []
    for index, case in enumerate(cases, start=1):
        try:
            seed = None
            if case.seed_sql:
                seeded = query(case.seed_sql)
                seed = str(seeded.rows[0][0]) if seeded.rows else None
            question, sqls = case.render(seed)
            references = [query(sql) for sql in sqls]
            conversation: dict[str, Any] | None = None
            latency_ms = 0.0
            input_tokens = output_tokens = total_tokens = 0
            saw_usage = False
            for message in case.setup:
                started = time.perf_counter()
                response = chat(message, conversation)
                latency_ms += (time.perf_counter() - started) * 1000
                conversation = _conversation(response)
                inp, out, tot = _usage_tokens(response)
                if any(v is not None for v in (inp, out, tot)):
                    saw_usage = True
                    input_tokens += inp or 0
                    output_tokens += out or 0
                    total_tokens += tot or (inp or 0) + (out or 0)
                _sleep(delay_seconds)
            started = time.perf_counter()
            response = chat(question, conversation)
            latency_ms += (time.perf_counter() - started) * 1000
            inp, out, tot = _usage_tokens(response)
            if any(v is not None for v in (inp, out, tot)):
                saw_usage = True
                input_tokens += inp or 0
                output_tokens += out or 0
                total_tokens += tot or (inp or 0) + (out or 0)
            _sleep(delay_seconds)
            result = score_case(case, response, references, question=question)
            result.latency_ms = round(latency_ms, 1)
            if saw_usage:
                result.input_tokens = input_tokens
                result.output_tokens = output_tokens
                result.total_tokens = total_tokens
        except Exception as exc:  # noqa: BLE001 — one failing case must not stop the run
            result = CaseResult(
                id=case.id,
                category=case.category,
                question=case.question,
                checks=[],
                error=f"{type(exc).__name__}: {str(exc)[:300]}",
            )
        results.append(result)
        status = "PASS" if result.passed else ("ERROR" if result.error else "FAIL")
        extra = ""
        if result.latency_ms is not None:
            extra = f" latency_ms={result.latency_ms:.0f}"
        if result.total_tokens is not None:
            extra += f" tokens={result.total_tokens}"
        log(f"[{index:02d}/{len(cases)}] {status} {case.id} score={result.score:.2f}{extra}")
    return results


def _conversation(response: dict[str, Any]) -> dict[str, Any] | None:
    if not response.get("conversation_id"):
        return None
    return {
        "conversation_id": response["conversation_id"],
        "conversation_version": response.get("conversation_version"),
    }


def _sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    by_check: dict[str, list[bool]] = defaultdict(list)
    by_category: dict[str, list[float]] = defaultdict(list)
    latencies = [r.latency_ms for r in results if r.latency_ms is not None]
    token_totals = [r.total_tokens for r in results if r.total_tokens is not None]
    for result in results:
        by_category[result.category].append(result.score)
        for check in result.applicable:
            by_check[check.name].append(bool(check.passed))
    total = len(results) or 1
    summary: dict[str, Any] = {
        "cases": len(results),
        "passed": sum(1 for r in results if r.passed),
        "errors": sum(1 for r in results if r.error),
        "pass_rate": round(sum(1 for r in results if r.passed) / total, 3),
        "mean_score": round(sum(r.score for r in results) / total, 3),
        "checks": {
            name: round(sum(values) / len(values), 3) for name, values in sorted(by_check.items())
        },
        "categories": {
            name: round(sum(values) / len(values), 3) for name, values in sorted(by_category.items())
        },
    }
    if latencies:
        ordered = sorted(latencies)
        summary["latency_ms"] = {
            "mean": round(sum(ordered) / len(ordered), 1),
            "p50": round(ordered[len(ordered) // 2], 1),
            "p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 1),
            "max": round(ordered[-1], 1),
        }
    if token_totals:
        summary["tokens"] = {
            "cases_with_usage": len(token_totals),
            "total": sum(token_totals),
            "mean_per_case": round(sum(token_totals) / len(token_totals), 1),
            "input_total": sum(r.input_tokens or 0 for r in results),
            "output_total": sum(r.output_tokens or 0 for r in results),
        }
    return summary


# ------------------------------------------------------------------ live adapters


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _live_adapters(args: argparse.Namespace) -> tuple[ChatFn, QueryFn, Callable[[], None]]:
    import httpx

    from app.core.security import create_access_token
    from app.db.models import DataSource
    from app.db.session import SessionLocal
    from app.mcp import MCPToolContext, build_postgres_mcp
    from app.mcp.servers.postgres.tools.query import POSTGRES_QUERY_TOOL_NAME

    data_source_id, user_id = UUID(args.data_source_id), UUID(args.user_id)
    session = SessionLocal()
    source = session.get(DataSource, data_source_id)
    if source is None:
        raise SystemExit("Data source not found")
    context = MCPToolContext(workspace_id=source.workspace_id, user_id=user_id)
    _registry, mcp = build_postgres_mcp(session)
    loop = asyncio.new_event_loop() if sys.platform != "win32" else asyncio.SelectorEventLoop()

    def query(sql: str) -> QueryData:
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                payload = loop.run_until_complete(
                    mcp.call_tool(
                        POSTGRES_QUERY_TOOL_NAME,
                        {"data_source_id": str(data_source_id), "sql": sql, "limit": 500},
                        context,
                    )
                )
                return QueryData(
                    columns=list(payload.columns),
                    rows=[[_jsonable(v) for v in row] for row in payload.rows],
                )
            except Exception as exc:  # noqa: BLE001 — retry transient MCP rate limits
                last_exc = exc
                name = type(exc).__name__.upper()
                if "RATE" not in name and "rate" not in str(exc).lower():
                    raise
                time.sleep(15 * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    client = httpx.Client(
        base_url=args.base_url,
        timeout=args.timeout,
    )

    def chat(message: str, conversation: dict[str, Any] | None) -> dict[str, Any]:
        body: dict[str, Any] = {"message": message, "data_source_id": str(data_source_id)}
        body.update({k: v for k, v in (conversation or {}).items() if v is not None})
        # Fresh JWT per call: access tokens expire in 15m; a full eval run is longer.
        headers = {"Authorization": f"Bearer {create_access_token(user_id)}"}
        response = client.post("/api/v1/ai/chat", json=body, headers=headers)
        if response.status_code in {429, 502, 503}:
            time.sleep(60 if response.status_code == 429 else 5)
            headers = {"Authorization": f"Bearer {create_access_token(user_id)}"}
            response = client.post("/api/v1/ai/chat", json=body, headers=headers)
        response.raise_for_status()
        return response.json()

    def close() -> None:
        client.close()
        loop.close()
        session.close()

    return chat, query, close


# ------------------------------------------------------------------ CLI


def _select(cases: list[EvalCase], args: argparse.Namespace) -> list[EvalCase]:
    if args.case:
        wanted = set(args.case)
        cases = [c for c in cases if c.id in wanted]
    if args.category:
        cases = [c for c in cases if c.category in set(args.category)]
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="call the chat API (real LLM calls)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--data-source-id", help="analyticcastle_demo data source id")
    parser.add_argument("--user-id", help="workspace member the questions run as")
    parser.add_argument("--case", action="append", help="run only this case id (repeatable)")
    parser.add_argument("--category", action="append", help="run only this category (repeatable)")
    parser.add_argument("--delay", type=float, default=7.0, help="seconds between chat calls (rate limit)")
    parser.add_argument("--timeout", type=float, default=240.0, help="chat request timeout seconds")
    parser.add_argument("--output", type=Path, help="JSON report path")
    args = parser.parse_args(argv)

    cases = load_cases()
    problems = validate_dataset(cases)
    for problem in problems:
        print(f"DATASET: {problem}")
    if not args.live:
        print(f"Offline check: {len(cases)} cases, {len(problems)} problem(s). Use --live to score answers.")
        return 1 if problems else 0
    if problems:
        return 1
    if not args.data_source_id or not args.user_id:
        parser.error("--live requires --data-source-id and --user-id")

    chat, query, close = _live_adapters(args)
    try:
        results = evaluate(_select(cases, args), chat=chat, query=query, delay_seconds=args.delay)
    finally:
        close()

    summary = summarize(results)
    output = args.output or DEFAULT_OUTPUT_DIR / f"run-{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "summary": summary,
                "results": [
                    {**r.model_dump(), "score": round(r.score, 3), "passed": r.passed} for r in results
                ],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
