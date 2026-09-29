"""Deterministic evaluation checks over the Phase 8 agent outputs.

Every check is decided against evidence the pipeline already computed itself:
result rows, the key metrics from `insight.evidence`, the trend series, and the
anomaly scan. A check that can only be judged by a heuristic (a cited number
that is not in the evidence, a sample too small for HIGH confidence, guarantee
wording) is reported as ESTIMATED so it is never read as proven incorrectness.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from app.ai.anomaly_detection.models import AnomalyAnalysisResult
from app.ai.data_analyst.models import AnalysisConfidence, DataAnalysisResult
from app.ai.evaluation.models import (
    CheckStatus,
    EvaluatedAgent,
    EvaluationCheck,
    EvaluationDimension,
    EvaluationMethod,
)
from app.ai.insight.evidence import groundable_names, identify_key_metrics
from app.ai.insight.models import InsightAnalysisResult, KeyMetric
from app.ai.recommendation.evidence import groundable_names as recommendation_names
from app.ai.recommendation.models import RecommendationResult, derive_priority
from app.ai.root_cause_analysis.models import RootCauseAnalysisResult
from app.ai.sql_execution.models import SQLExecutionResult
from app.ai.trend_analysis.models import TrendAnalysisResult, TrendDirection
from app.ai.trend_analysis.series import MIN_TREND_POINTS, parse_number

# Calibration knobs: narratives round their numbers and small samples are noisy,
# so these bands decide "matches the evidence" and "enough data for HIGH".
NUMBER_ABS_TOLERANCE = 0.5
NUMBER_REL_TOLERANCE = 0.02
FLOAT_EQUAL_TOLERANCE = 1e-6
MIN_ROWS_FOR_HIGH_CONFIDENCE = 5
MIN_POINTS_FOR_HIGH_CONFIDENCE = 4
MAX_REPORTED_ITEMS = 5

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_GUARANTEE_TERMS = (
    "guarantee",
    "guaranteed",
    "will increase",
    "will decrease",
    "will reduce",
    "will grow",
    "will double",
    "ensures",
    "definitely",
    "certainly",
    "always results",
    "risk-free",
)

_VERIFIED = EvaluationMethod.VERIFIED
_ESTIMATED = EvaluationMethod.ESTIMATED


def _make(
    agent: EvaluatedAgent,
    check_id: str,
    dimension: EvaluationDimension,
    *,
    passed: bool,
    method: EvaluationMethod,
    detail: str,
    evidence_reference: str | None = None,
) -> EvaluationCheck:
    """Build a check. A failed VERIFIED check is a FAIL; a failed estimate is a WARN."""
    if passed:
        status = CheckStatus.PASS
    else:
        status = CheckStatus.FAIL if method is _VERIFIED else CheckStatus.WARN
    return EvaluationCheck(
        agent=agent,
        check_id=check_id,
        dimension=dimension,
        status=status,
        method=method,
        detail=detail,
        evidence_reference=evidence_reference,
    )


def _skipped(
    agent: EvaluatedAgent,
    check_id: str,
    dimension: EvaluationDimension,
    *,
    method: EvaluationMethod,
    detail: str,
    evidence_reference: str | None = None,
) -> EvaluationCheck:
    return EvaluationCheck(
        agent=agent,
        check_id=check_id,
        dimension=dimension,
        status=CheckStatus.SKIPPED,
        method=method,
        detail=detail,
        evidence_reference=evidence_reference,
    )


def text_numbers(texts: Iterable[str | None]) -> list[float]:
    """Numbers a narrative cites, deduplicated, in the order they appear."""
    found: list[float] = []
    seen: set[float] = set()
    for text in texts:
        if not text:
            continue
        for token in _NUMBER_RE.findall(text):
            try:
                value = float(token.replace(",", ""))
            except ValueError:
                continue
            if value not in seen:
                seen.add(value)
                found.append(value)
    return found


def evidence_numbers(
    *,
    query_result: SQLExecutionResult | None = None,
    key_metrics: Sequence[KeyMetric] = (),
    trend: TrendAnalysisResult | None = None,
    anomalies: AnomalyAnalysisResult | None = None,
    extra_rows: Iterable[Sequence[object]] = (),
) -> list[float]:
    """Every number the pipeline itself produced, against which citations are checked."""
    values: set[float] = set()

    def add(*items: float | None) -> None:
        for item in items:
            if item is not None:
                values.add(float(item))

    rows: list[Sequence[object]] = list(query_result.rows) if query_result else []
    rows.extend(extra_rows)
    for row in rows:
        for cell in row:
            number = parse_number(cell)
            if number is not None:
                add(number)
            elif cell is not None:
                values.update(text_numbers([str(cell)]))

    if query_result is not None:
        add(query_result.row_count, len(query_result.columns))

    for metric in key_metrics:
        add(
            metric.value_count,
            metric.total,
            metric.average,
            metric.minimum,
            metric.maximum,
        )

    if trend is not None:
        series = trend.series
        add(
            series.point_count,
            series.skipped_row_count,
            series.direction_changes,
            series.first_value,
            series.last_value,
            series.minimum_value,
            series.maximum_value,
            series.total_change,
            series.growth_rate_percent,
            series.average_period_change_percent,
        )
        values.update(text_numbers([series.first_period, series.last_period]))
        for comparison in series.comparisons:
            add(
                comparison.previous_value,
                comparison.value,
                comparison.absolute_change,
                comparison.percent_change,
            )
            values.update(text_numbers([comparison.previous_period, comparison.period]))

    if anomalies is not None:
        add(anomalies.anomaly_count, anomalies.scan.scanned_row_count)
        for statistics in anomalies.scan.column_statistics.values():
            add(*statistics.values())
        for anomaly in anomalies.scan.anomalies:
            add(
                anomaly.value,
                anomaly.previous_value,
                anomaly.percent_change,
                anomaly.expected_low,
                anomaly.expected_high,
                anomaly.score,
            )
            values.update(text_numbers([anomaly.period]))

    return sorted(values)


def _grounded(value: float, evidence: Sequence[float]) -> bool:
    tolerance = max(NUMBER_ABS_TOLERANCE, NUMBER_REL_TOLERANCE * abs(value))
    return any(abs(value - candidate) <= tolerance for candidate in evidence)


def ungrounded_numbers(
    texts: Iterable[str | None],
    evidence: Sequence[float],
) -> list[float]:
    """Numbers cited by the agent that no computed evidence value matches."""
    return [value for value in text_numbers(texts) if not _grounded(value, evidence)]


def _citation_check(
    agent: EvaluatedAgent,
    texts: Sequence[str | None],
    evidence: Sequence[float],
    *,
    check_id: str,
    evidence_reference: str,
) -> EvaluationCheck:
    if not evidence:
        return _skipped(
            agent,
            check_id,
            EvaluationDimension.HALLUCINATION,
            method=_ESTIMATED,
            detail="No computed evidence values were available to check cited numbers against.",
            evidence_reference=evidence_reference,
        )
    unmatched = ungrounded_numbers(texts, evidence)
    return _make(
        agent,
        check_id,
        EvaluationDimension.HALLUCINATION,
        passed=not unmatched,
        method=_ESTIMATED,
        detail=(
            "Every number cited by the agent matches a computed evidence value."
            if not unmatched
            else "Cited numbers with no matching computed evidence value: "
            + ", ".join(_format_number(value) for value in unmatched[:MAX_REPORTED_ITEMS])
        ),
        evidence_reference=evidence_reference,
    )


def _cited_names(texts: Iterable[str | None], names: dict[str, str]) -> list[str]:
    blob = " ".join(text.lower() for text in texts if text)
    return [canonical for lowered, canonical in names.items() if lowered and lowered in blob]


def _confidence_vs_rows(
    agent: EvaluatedAgent,
    confidence: AnalysisConfidence,
    *,
    row_count: int,
    prefix: str,
) -> list[EvaluationCheck]:
    """Split calibration in two: a contradiction is verified, a thin sample is an estimate."""
    high = confidence is AnalysisConfidence.HIGH
    return [
        _make(
            agent,
            f"{prefix}.confidence_requires_evidence",
            EvaluationDimension.CALIBRATION,
            passed=not (high and row_count == 0),
            method=_VERIFIED,
            detail=(
                f"Confidence {confidence.value} reported with {row_count} result row(s)."
                if not (high and row_count == 0)
                else "HIGH confidence reported although no result rows were available."
            ),
            evidence_reference="SQLExecutionResult.rows",
        ),
        _make(
            agent,
            f"{prefix}.confidence_sample_size",
            EvaluationDimension.CALIBRATION,
            passed=not (high and 0 < row_count < MIN_ROWS_FOR_HIGH_CONFIDENCE),
            method=_ESTIMATED,
            detail=(
                f"Confidence {confidence.value} is plausible for {row_count} result row(s)."
                if not (high and 0 < row_count < MIN_ROWS_FOR_HIGH_CONFIDENCE)
                else f"HIGH confidence rests on only {row_count} result row(s), below the "
                f"{MIN_ROWS_FOR_HIGH_CONFIDENCE}-row band used here."
            ),
            evidence_reference="SQLExecutionResult.rows",
        ),
    ]


def _rank_check(
    agent: EvaluatedAgent,
    ranks: Sequence[int],
    *,
    check_id: str,
    evidence_reference: str,
) -> EvaluationCheck:
    expected = list(range(1, len(ranks) + 1))
    return _make(
        agent,
        check_id,
        EvaluationDimension.STRUCTURE,
        passed=list(ranks) == expected,
        method=_VERIFIED,
        detail=(
            f"Ranks are contiguous from 1 to {len(ranks)}."
            if list(ranks) == expected
            else f"Ranks {list(ranks)} are not contiguous from 1 to {len(ranks)}."
        ),
        evidence_reference=evidence_reference,
    )


def _format_number(value: float) -> str:
    return f"{value:g}"


def _same_number(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(left - right) <= FLOAT_EQUAL_TOLERANCE


def evaluate_data_analyst(
    analysis: DataAnalysisResult,
    *,
    query_result: SQLExecutionResult | None = None,
    key_metrics: Sequence[KeyMetric] = (),
) -> list[EvaluationCheck]:
    agent = EvaluatedAgent.DATA_ANALYST
    row_count = len(query_result.rows) if query_result else 0
    texts: list[str | None] = [
        analysis.interpretation,
        analysis.summary,
        analysis.comparisons,
        *analysis.conclusions,
    ]
    names = groundable_names(query_result, list(key_metrics))

    checks = [
        _make(
            agent,
            "data_analyst.conclusions_present",
            EvaluationDimension.COMPLETENESS,
            passed=bool(analysis.conclusions) or row_count == 0,
            method=_VERIFIED,
            detail=(
                f"{len(analysis.conclusions)} conclusion(s) returned for "
                f"{row_count} result row(s)."
            ),
            evidence_reference="DataAnalysisResult.conclusions",
        )
    ]

    if names:
        cited = _cited_names(texts, names)
        checks.append(
            _make(
                agent,
                "data_analyst.cites_result_columns",
                EvaluationDimension.GROUNDING,
                passed=bool(cited),
                method=_VERIFIED,
                detail=(
                    "Analysis references result column(s): "
                    + ", ".join(sorted(cited)[:MAX_REPORTED_ITEMS])
                    if cited
                    else "Analysis references none of the result columns or key metrics."
                ),
                evidence_reference="SQLExecutionResult.columns",
            )
        )

    checks.append(
        _citation_check(
            agent,
            texts,
            evidence_numbers(query_result=query_result, key_metrics=key_metrics),
            check_id="data_analyst.numbers_grounded",
            evidence_reference="SQLExecutionResult.rows, KeyMetric",
        )
    )
    checks.extend(
        _confidence_vs_rows(
            agent,
            analysis.confidence_score,
            row_count=row_count,
            prefix="data_analyst",
        )
    )
    return checks


def evaluate_trend(
    trend: TrendAnalysisResult,
    *,
    query_result: SQLExecutionResult | None = None,
) -> list[EvaluationCheck]:
    agent = EvaluatedAgent.TREND_ANALYSIS
    series = trend.series
    texts: list[str | None] = [
        trend.summary,
        trend.direction_explanation,
        trend.period_comparisons,
        trend.significant_changes,
        *trend.conclusions,
    ]
    high = trend.confidence_score is AnalysisConfidence.HIGH
    thin_series = series.point_count < MIN_TREND_POINTS
    insufficient = trend.direction is TrendDirection.INSUFFICIENT_DATA

    checks = [
        _make(
            agent,
            "trend.direction_matches_series",
            EvaluationDimension.ACCURACY,
            passed=trend.direction is series.direction,
            method=_VERIFIED,
            detail=(
                f"Reported direction {trend.direction.value} matches the computed series."
                if trend.direction is series.direction
                else f"Reported direction {trend.direction.value} differs from the computed "
                f"series direction {series.direction.value}."
            ),
            evidence_reference="TrendSeries.direction",
        ),
        _make(
            agent,
            "trend.growth_rate_matches_series",
            EvaluationDimension.ACCURACY,
            passed=_same_number(trend.growth_rate_percent, series.growth_rate_percent),
            method=_VERIFIED,
            detail=(
                "Reported growth rate matches the computed series."
                if _same_number(trend.growth_rate_percent, series.growth_rate_percent)
                else f"Reported growth rate {trend.growth_rate_percent} differs from the "
                f"computed {series.growth_rate_percent}."
            ),
            evidence_reference="TrendSeries.growth_rate_percent",
        ),
        _make(
            agent,
            "trend.conclusions_present",
            EvaluationDimension.COMPLETENESS,
            passed=bool(trend.conclusions) or thin_series,
            method=_VERIFIED,
            detail=(
                f"{len(trend.conclusions)} conclusion(s) returned for "
                f"{series.point_count} series point(s)."
            ),
            evidence_reference="TrendAnalysisResult.conclusions",
        ),
    ]

    if series.significant_changes:
        periods = [item.period for item in series.significant_changes if item.period]
        mentioned = [
            period for period in periods if period.lower() in (trend.significant_changes or "").lower()
        ]
        checks.append(
            _make(
                agent,
                "trend.significant_changes_named",
                EvaluationDimension.GROUNDING,
                passed=bool(mentioned),
                method=_VERIFIED,
                detail=(
                    "Narrative names detected significant period(s): "
                    + ", ".join(mentioned[:MAX_REPORTED_ITEMS])
                    if mentioned
                    else f"{len(periods)} significant change(s) were detected but the narrative "
                    "names none of their periods."
                ),
                evidence_reference="TrendSeries.significant_changes",
            )
        )

    checks.append(
        _make(
            agent,
            "trend.confidence_requires_series",
            EvaluationDimension.CALIBRATION,
            passed=not (high and (thin_series or insufficient)),
            method=_VERIFIED,
            detail=(
                f"Confidence {trend.confidence_score.value} is consistent with a "
                f"{series.point_count}-point series and direction {trend.direction.value}."
                if not (high and (thin_series or insufficient))
                else f"HIGH confidence reported although the series has "
                f"{series.point_count} point(s) and direction {trend.direction.value}."
            ),
            evidence_reference="TrendSeries.point_count, TrendSeries.direction",
        )
    )
    checks.append(
        _make(
            agent,
            "trend.confidence_series_length",
            EvaluationDimension.CALIBRATION,
            passed=not (high and MIN_TREND_POINTS <= series.point_count < MIN_POINTS_FOR_HIGH_CONFIDENCE),
            method=_ESTIMATED,
            detail=(
                f"Confidence {trend.confidence_score.value} is plausible for "
                f"{series.point_count} series point(s)."
                if not (high and MIN_TREND_POINTS <= series.point_count < MIN_POINTS_FOR_HIGH_CONFIDENCE)
                else f"HIGH confidence rests on {series.point_count} series point(s), below the "
                f"{MIN_POINTS_FOR_HIGH_CONFIDENCE}-point band used here."
            ),
            evidence_reference="TrendSeries.point_count",
        )
    )
    checks.append(
        _citation_check(
            agent,
            texts,
            evidence_numbers(query_result=query_result, trend=trend),
            check_id="trend.numbers_grounded",
            evidence_reference="TrendSeries, SQLExecutionResult.rows",
        )
    )
    return checks


def evaluate_anomalies(
    anomalies: AnomalyAnalysisResult,
    *,
    query_result: SQLExecutionResult | None = None,
) -> list[EvaluationCheck]:
    agent = EvaluatedAgent.ANOMALY_DETECTION
    scan = anomalies.scan
    texts: list[str | None] = [
        anomalies.summary,
        anomalies.outliers,
        anomalies.unexpected_changes,
        anomalies.threshold_breaches,
        *anomalies.conclusions,
    ]
    high = anomalies.confidence_score is AnalysisConfidence.HIGH

    checks = [
        _make(
            agent,
            "anomaly.count_matches_scan",
            EvaluationDimension.ACCURACY,
            passed=anomalies.anomaly_count == len(scan.anomalies),
            method=_VERIFIED,
            detail=(
                f"Reported anomaly count {anomalies.anomaly_count} matches the scan."
                if anomalies.anomaly_count == len(scan.anomalies)
                else f"Reported anomaly count {anomalies.anomaly_count} differs from the "
                f"{len(scan.anomalies)} anomalies in the scan."
            ),
            evidence_reference="AnomalyScan.anomalies",
        ),
        _make(
            agent,
            "anomaly.severity_matches_scan",
            EvaluationDimension.ACCURACY,
            passed=anomalies.highest_severity is scan.highest_severity,
            method=_VERIFIED,
            detail=(
                "Reported highest severity matches the scan."
                if anomalies.highest_severity is scan.highest_severity
                else f"Reported highest severity {anomalies.highest_severity} differs from the "
                f"scan's {scan.highest_severity}."
            ),
            evidence_reference="AnomalyScan.highest_severity",
        ),
        _make(
            agent,
            "anomaly.conclusions_present",
            EvaluationDimension.COMPLETENESS,
            passed=bool(anomalies.conclusions) or not scan.anomalies,
            method=_VERIFIED,
            detail=(
                f"{len(anomalies.conclusions)} conclusion(s) returned for "
                f"{len(scan.anomalies)} detected anomaly/anomalies."
            ),
            evidence_reference="AnomalyAnalysisResult.conclusions",
        ),
    ]

    if scan.anomalies:
        columns = {anomaly.column for anomaly in scan.anomalies}
        cited = _cited_names(texts, {column.lower(): column for column in columns})
        checks.append(
            _make(
                agent,
                "anomaly.columns_named",
                EvaluationDimension.GROUNDING,
                passed=bool(cited),
                method=_VERIFIED,
                detail=(
                    "Narrative names anomalous column(s): "
                    + ", ".join(sorted(cited)[:MAX_REPORTED_ITEMS])
                    if cited
                    else "Narrative names none of the columns the scan flagged: "
                    + ", ".join(sorted(columns)[:MAX_REPORTED_ITEMS])
                ),
                evidence_reference="AnomalyScan.anomalies[].column",
            )
        )

    checks.append(
        _make(
            agent,
            "anomaly.confidence_requires_scan",
            EvaluationDimension.CALIBRATION,
            passed=not (high and not scan.analyzed),
            method=_VERIFIED,
            detail=(
                f"Confidence {anomalies.confidence_score.value} is consistent with a scan that "
                f"{'ran' if scan.analyzed else 'did not run'}."
                if not (high and not scan.analyzed)
                else "HIGH confidence reported although no anomaly scan could be run."
            ),
            evidence_reference="AnomalyScan.analyzed",
        )
    )
    checks.append(
        _make(
            agent,
            "anomaly.confidence_sample_size",
            EvaluationDimension.CALIBRATION,
            passed=not (
                high and 0 < scan.scanned_row_count < MIN_ROWS_FOR_HIGH_CONFIDENCE
            ),
            method=_ESTIMATED,
            detail=(
                f"Confidence {anomalies.confidence_score.value} is plausible for "
                f"{scan.scanned_row_count} scanned row(s)."
                if not (high and 0 < scan.scanned_row_count < MIN_ROWS_FOR_HIGH_CONFIDENCE)
                else f"HIGH confidence rests on {scan.scanned_row_count} scanned row(s), below "
                f"the {MIN_ROWS_FOR_HIGH_CONFIDENCE}-row band used here."
            ),
            evidence_reference="AnomalyScan.scanned_row_count",
        )
    )
    checks.append(
        _citation_check(
            agent,
            texts,
            evidence_numbers(query_result=query_result, anomalies=anomalies),
            check_id="anomaly.numbers_grounded",
            evidence_reference="AnomalyScan, SQLExecutionResult.rows",
        )
    )
    return checks


def evaluate_root_cause(
    root_cause: RootCauseAnalysisResult,
    *,
    query_result: SQLExecutionResult | None = None,
    key_metrics: Sequence[KeyMetric] = (),
) -> list[EvaluationCheck]:
    agent = EvaluatedAgent.ROOT_CAUSE_ANALYSIS
    hypotheses = root_cause.hypotheses
    texts: list[str | None] = [root_cause.summary, root_cause.primary_cause]
    for item in hypotheses:
        texts.extend(
            [
                item.statement,
                item.supporting_evidence,
                item.contradicting_evidence,
                *item.contributing_factors,
            ]
        )
    row_count = len(query_result.rows) if query_result else 0
    gathered_rows = [row for item in root_cause.evidence for row in item.rows]

    checks = [
        _make(
            agent,
            "root_cause.hypotheses_present",
            EvaluationDimension.COMPLETENESS,
            passed=bool(hypotheses),
            method=_VERIFIED,
            detail=f"{len(hypotheses)} hypothesis/hypotheses returned.",
            evidence_reference="RootCauseAnalysisResult.hypotheses",
        ),
        _rank_check(
            agent,
            [item.rank for item in hypotheses],
            check_id="root_cause.hypothesis_ranks",
            evidence_reference="RootCauseHypothesis.rank",
        ),
        _make(
            agent,
            "root_cause.primary_matches_top_hypothesis",
            EvaluationDimension.STRUCTURE,
            passed=(
                root_cause.primary_cause == hypotheses[0].statement
                if hypotheses
                else root_cause.primary_cause is None
            ),
            method=_VERIFIED,
            detail=(
                "Primary cause is the highest-ranked hypothesis."
                if (
                    root_cause.primary_cause == hypotheses[0].statement
                    if hypotheses
                    else root_cause.primary_cause is None
                )
                else "Primary cause does not match the highest-ranked hypothesis."
            ),
            evidence_reference="RootCauseAnalysisResult.primary_cause",
        ),
    ]

    unresolved_high = [
        item
        for item in hypotheses
        if item.confidence_score is AnalysisConfidence.HIGH and item.investigation_question
    ]
    checks.append(
        _make(
            agent,
            "root_cause.high_confidence_not_open_question",
            EvaluationDimension.CALIBRATION,
            passed=not unresolved_high,
            method=_VERIFIED,
            detail=(
                "No hypothesis claims HIGH confidence while still requesting investigation."
                if not unresolved_high
                else f"{len(unresolved_high)} hypothesis/hypotheses claim HIGH confidence while "
                "still requesting further investigation."
            ),
            evidence_reference="RootCauseHypothesis.investigation_question",
        )
    )
    contested_high = [
        item
        for item in hypotheses
        if item.confidence_score is AnalysisConfidence.HIGH and item.contradicting_evidence
    ]
    checks.append(
        _make(
            agent,
            "root_cause.high_confidence_uncontested",
            EvaluationDimension.CALIBRATION,
            passed=not contested_high,
            method=_ESTIMATED,
            detail=(
                "No HIGH-confidence hypothesis carries contradicting evidence."
                if not contested_high
                else f"{len(contested_high)} HIGH-confidence hypothesis/hypotheses also record "
                "contradicting evidence."
            ),
            evidence_reference="RootCauseHypothesis.contradicting_evidence",
        )
    )
    high_without_evidence = (
        root_cause.confidence_score is AnalysisConfidence.HIGH
        and row_count == 0
        and not gathered_rows
    )
    checks.append(
        _make(
            agent,
            "root_cause.confidence_requires_evidence",
            EvaluationDimension.CALIBRATION,
            passed=not high_without_evidence,
            method=_VERIFIED,
            detail=(
                f"Confidence {root_cause.confidence_score.value} reported with {row_count} "
                f"result row(s) and {root_cause.additional_queries_run} investigation query/queries."
                if not high_without_evidence
                else "HIGH confidence reported although neither result rows nor gathered "
                "evidence rows were available."
            ),
            evidence_reference="RootCauseAnalysisResult.evidence, SQLExecutionResult.rows",
        )
    )

    names = groundable_names(query_result, list(key_metrics))
    for item in root_cause.evidence:
        names.update({column.strip().lower(): column for column in item.columns})
    if names:
        cited = _cited_names(texts, names)
        checks.append(
            _make(
                agent,
                "root_cause.cites_data_names",
                EvaluationDimension.GROUNDING,
                passed=bool(cited),
                method=_ESTIMATED,
                detail=(
                    "Hypotheses reference data name(s): "
                    + ", ".join(sorted(cited)[:MAX_REPORTED_ITEMS])
                    if cited
                    else "No hypothesis references a result column, key metric, or gathered "
                    "evidence column by name."
                ),
                evidence_reference="SQLExecutionResult.columns, RootCauseEvidence.columns",
            )
        )

    checks.append(
        _citation_check(
            agent,
            texts,
            evidence_numbers(
                query_result=query_result,
                key_metrics=key_metrics,
                extra_rows=gathered_rows,
            ),
            check_id="root_cause.numbers_grounded",
            evidence_reference="SQLExecutionResult.rows, RootCauseEvidence.rows",
        )
    )
    return checks


def evaluate_insight(
    insights: InsightAnalysisResult,
    *,
    query_result: SQLExecutionResult | None = None,
) -> list[EvaluationCheck]:
    agent = EvaluatedAgent.INSIGHT
    items = insights.insights
    row_count = len(query_result.rows) if query_result else 0
    texts: list[str | None] = [insights.summary, insights.top_insight]
    for item in items:
        texts.extend([item.title, item.insight, item.business_impact, item.supporting_evidence])

    checks = [
        _make(
            agent,
            "insight.insights_present",
            EvaluationDimension.COMPLETENESS,
            passed=bool(items) or row_count == 0,
            method=_VERIFIED,
            detail=f"{len(items)} insight(s) returned for {row_count} result row(s).",
            evidence_reference="InsightAnalysisResult.insights",
        ),
        _rank_check(
            agent,
            [item.rank for item in items],
            check_id="insight.ranks_contiguous",
            evidence_reference="BusinessInsight.rank",
        ),
        _make(
            agent,
            "insight.top_matches_first_rank",
            EvaluationDimension.STRUCTURE,
            passed=(
                insights.top_insight == items[0].insight if items else insights.top_insight is None
            ),
            method=_VERIFIED,
            detail=(
                "Top insight is the highest-ranked insight."
                if (insights.top_insight == items[0].insight if items else insights.top_insight is None)
                else "Top insight does not match the highest-ranked insight."
            ),
            evidence_reference="InsightAnalysisResult.top_insight",
        ),
    ]

    if query_result is not None:
        recomputed = identify_key_metrics(query_result)
        matches = recomputed == list(insights.key_metrics)
        checks.append(
            _make(
                agent,
                "insight.key_metrics_reproducible",
                EvaluationDimension.ACCURACY,
                passed=matches,
                method=_VERIFIED,
                detail=(
                    f"{len(recomputed)} key metric(s) recomputed from the result rows match the "
                    "reported metrics."
                    if matches
                    else f"Reported key metrics differ from the {len(recomputed)} metric(s) "
                    "recomputed from the result rows."
                ),
                evidence_reference="identify_key_metrics(SQLExecutionResult)",
            )
        )

    names = groundable_names(query_result, list(insights.key_metrics))
    ungrounded = [
        item.metric
        for item in items
        if item.metric and item.metric.strip().lower() not in names
    ]
    checks.append(
        _make(
            agent,
            "insight.metrics_grounded",
            EvaluationDimension.GROUNDING,
            passed=not ungrounded,
            method=_VERIFIED,
            detail=(
                "Every cited metric exists in the result columns or key metrics."
                if not ungrounded
                else "Insights cite metric(s) absent from the evidence: "
                + ", ".join(str(name) for name in ungrounded[:MAX_REPORTED_ITEMS])
            ),
            evidence_reference="SQLExecutionResult.columns, KeyMetric.column",
        )
    )
    checks.append(
        _make(
            agent,
            "insight.confidence_matches_top_insight",
            EvaluationDimension.CALIBRATION,
            passed=(
                insights.confidence_score is items[0].confidence_score if items else True
            ),
            method=_VERIFIED,
            detail=(
                "Overall confidence matches the highest-ranked insight."
                if (insights.confidence_score is items[0].confidence_score if items else True)
                else f"Overall confidence {insights.confidence_score.value} differs from the "
                f"top insight's {items[0].confidence_score.value}."
            ),
            evidence_reference="BusinessInsight.confidence_score",
        )
    )
    high_with_gaps = insights.confidence_score is AnalysisConfidence.HIGH and insights.data_gaps
    checks.append(
        _make(
            agent,
            "insight.confidence_vs_data_gaps",
            EvaluationDimension.CALIBRATION,
            passed=not high_with_gaps,
            method=_ESTIMATED,
            detail=(
                "Confidence is consistent with the reported data gaps."
                if not high_with_gaps
                else f"HIGH confidence reported alongside {len(insights.data_gaps)} data gap(s)."
            ),
            evidence_reference="InsightAnalysisResult.data_gaps",
        )
    )
    checks.extend(
        _confidence_vs_rows(
            agent,
            insights.confidence_score,
            row_count=row_count,
            prefix="insight",
        )
    )
    checks.append(
        _citation_check(
            agent,
            texts,
            evidence_numbers(query_result=query_result, key_metrics=insights.key_metrics),
            check_id="insight.numbers_grounded",
            evidence_reference="SQLExecutionResult.rows, KeyMetric",
        )
    )
    return checks


def evaluate_recommendation(
    recommendations: RecommendationResult,
    *,
    query_result: SQLExecutionResult | None = None,
    insights: InsightAnalysisResult | None = None,
) -> list[EvaluationCheck]:
    agent = EvaluatedAgent.RECOMMENDATION
    items = recommendations.recommendations
    row_count = len(query_result.rows) if query_result else 0
    key_metrics = (
        list(insights.key_metrics) if insights is not None else identify_key_metrics(query_result)
    )
    outcome_texts: list[str | None] = [item.expected_outcome for item in items]
    texts: list[str | None] = [recommendations.summary, *outcome_texts]
    for item in items:
        texts.extend([item.recommendation, item.supporting_evidence])

    checks = [
        _make(
            agent,
            "recommendation.recommendations_present",
            EvaluationDimension.COMPLETENESS,
            passed=bool(items) or row_count == 0,
            method=_VERIFIED,
            detail=f"{len(items)} recommendation(s) returned for {row_count} result row(s).",
            evidence_reference="RecommendationResult.recommendations",
        ),
        _rank_check(
            agent,
            [item.rank for item in items],
            check_id="recommendation.ranks_contiguous",
            evidence_reference="Recommendation.rank",
        ),
        _make(
            agent,
            "recommendation.top_matches_first_rank",
            EvaluationDimension.STRUCTURE,
            passed=(
                recommendations.top_recommendation == items[0].recommendation
                if items
                else recommendations.top_recommendation is None
            ),
            method=_VERIFIED,
            detail=(
                "Top recommendation is the highest-ranked recommendation."
                if (
                    recommendations.top_recommendation == items[0].recommendation
                    if items
                    else recommendations.top_recommendation is None
                )
                else "Top recommendation does not match the highest-ranked recommendation."
            ),
            evidence_reference="RecommendationResult.top_recommendation",
        ),
    ]

    mispriced = [
        item.title
        for item in items
        if item.priority is not derive_priority(item.impact, item.feasibility)
    ]
    checks.append(
        _make(
            agent,
            "recommendation.priority_derived",
            EvaluationDimension.ACCURACY,
            passed=not mispriced,
            method=_VERIFIED,
            detail=(
                "Every priority equals impact and feasibility combined."
                if not mispriced
                else "Priority does not follow from impact and feasibility for: "
                + ", ".join(mispriced[:MAX_REPORTED_ITEMS])
            ),
            evidence_reference="derive_priority(impact, feasibility)",
        )
    )

    names = recommendation_names(query_result, key_metrics, insights)
    ungrounded = [
        item.evidence_reference
        for item in items
        if item.evidence_reference.strip().lower() not in names
    ]
    checks.append(
        _make(
            agent,
            "recommendation.evidence_references_grounded",
            EvaluationDimension.GROUNDING,
            passed=not ungrounded,
            method=_VERIFIED,
            detail=(
                "Every evidence reference exists in the result columns, key metrics, or insights."
                if not ungrounded
                else "Recommendations cite reference(s) absent from the evidence: "
                + ", ".join(ungrounded[:MAX_REPORTED_ITEMS])
            ),
            evidence_reference="SQLExecutionResult.columns, KeyMetric.column, BusinessInsight.title",
        )
    )

    guaranteed = [
        item.title
        for item in items
        if any(term in (item.expected_outcome or "").lower() for term in _GUARANTEE_TERMS)
    ]
    checks.append(
        _make(
            agent,
            "recommendation.outcomes_not_guaranteed",
            EvaluationDimension.HALLUCINATION,
            passed=not guaranteed,
            method=_ESTIMATED,
            detail=(
                "No expected outcome is stated as a guaranteed result."
                if not guaranteed
                else "Expected outcomes state a guaranteed result for: "
                + ", ".join(guaranteed[:MAX_REPORTED_ITEMS])
            ),
            evidence_reference="Recommendation.expected_outcome",
        )
    )

    high_with_gaps = (
        recommendations.confidence_score is AnalysisConfidence.HIGH and recommendations.data_gaps
    )
    checks.append(
        _make(
            agent,
            "recommendation.confidence_vs_data_gaps",
            EvaluationDimension.CALIBRATION,
            passed=not high_with_gaps,
            method=_ESTIMATED,
            detail=(
                "Confidence is consistent with the reported data gaps."
                if not high_with_gaps
                else f"HIGH confidence reported alongside "
                f"{len(recommendations.data_gaps)} data gap(s)."
            ),
            evidence_reference="RecommendationResult.data_gaps",
        )
    )
    checks.extend(
        _confidence_vs_rows(
            agent,
            recommendations.confidence_score,
            row_count=row_count,
            prefix="recommendation",
        )
    )
    checks.append(
        _citation_check(
            agent,
            texts,
            evidence_numbers(query_result=query_result, key_metrics=key_metrics),
            check_id="recommendation.numbers_grounded",
            evidence_reference="SQLExecutionResult.rows, KeyMetric",
        )
    )
    return checks
