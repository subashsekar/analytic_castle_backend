"""Schema-aware metric mapping: counts, numeric measures, glossary (no database)."""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from app.ai.glossary import GlossaryEntry, match_glossary, parse_glossary
from app.ai.intent_types import AggregationType, AIIntent, AIIntentType, AIRequestPlan
from app.ai.metadata_resolver import _apply_glossary, _concept, _refine_metric, _search_terms
from app.ai.metadata_types import (
    ConceptColumnResolution,
    MetadataColumnCandidate,
    MetadataMatchReason,
    MetadataPrimaryKey,
    MetadataTableCandidate,
    ResolvedMetadataContext,
)
from app.ai.orchestrator import _apply_clarification_policy
from app.ai.sql_generation.schema_context import build_schema_prompt_context

TABLE_ID = uuid.uuid4()


def _col(name: str, data_type: str, *, pk: bool = False, description: str | None = None):
    return MetadataColumnCandidate(
        column_id=uuid.uuid4(),
        table_id=TABLE_ID,
        schema_name="public",
        table_name="sales",
        column_name=name,
        data_type=data_type,
        is_primary_key=pk,
        description=description,
        match_reason=MetadataMatchReason.EXACT,
        relevance_score=100,
    )


SALE_ID = _col("sale_id", "integer", pk=True)
REGION = _col("region", "text")
REVENUE = _col("revenue", "numeric", description="Booked revenue in USD")
ORDERS = _col("orders", "integer")
CUSTOMER_ID = _col("customer_id", "integer")
PRODUCT_ID = _col("product_id", "integer")
COLUMNS = [SALE_ID, REGION, REVENUE, ORDERS, CUSTOMER_ID, PRODUCT_ID]
TABLE = MetadataTableCandidate(
    table_id=TABLE_ID,
    schema_name="public",
    table_name="sales",
    match_reason=MetadataMatchReason.EXACT,
    relevance_score=100,
    primary_key_columns=[MetadataPrimaryKey(column_id=SALE_ID.column_id, column_name="sale_id")],
)


def _refine(item, aggregation=AggregationType.NONE, entry=None):
    return _refine_metric(
        item, aggregation=aggregation, glossary_entry=entry, tables=[TABLE], columns=COLUMNS
    )


def _unresolved(name: str) -> ConceptColumnResolution:
    return ConceptColumnResolution(requested=name, resolved=False)


def test_count_of_entity_resolves_to_row_count() -> None:
    for item, aggregation in (
        (_unresolved("sales count"), AggregationType.NONE),
        (_unresolved("sales_count"), AggregationType.NONE),
        (_unresolved("number of sales"), AggregationType.NONE),
        (_unresolved("sales"), AggregationType.COUNT),
    ):
        result = _refine(item, aggregation)
        assert result.resolved is True, item.requested
        assert result.resolution_note == "COUNT(*) of public.sales rows"
        assert [c.column_name for c in result.candidates] == ["sale_id"]


def test_count_of_related_entity_uses_distinct_id() -> None:
    result = _refine(_unresolved("customers"), AggregationType.COUNT_DISTINCT)
    assert result.resolved is True
    assert result.resolution_note == "COUNT(DISTINCT public.sales.customer_id)"
    sold = _refine(_unresolved("distinct products sold"), AggregationType.NONE)
    assert sold.resolved is True
    assert "product_id" in (sold.resolution_note or "")


def test_count_over_numeric_count_column_is_explained() -> None:
    item = ConceptColumnResolution(requested="orders", resolved=True, candidates=[ORDERS])
    note = _refine(item, AggregationType.COUNT).resolution_note or ""
    assert "SUM it" in note and "public.sales.orders" in note


def test_measures_require_numeric_evidence() -> None:
    # "sales" is a table here, not a revenue column: no silent substitution.
    assert _refine(_unresolved("sales"), AggregationType.SUM).resolved is False
    text_hit = ConceptColumnResolution(requested="region", resolved=True, candidates=[REGION])
    assert _refine(text_hit, AggregationType.SUM).resolved is False
    numeric_hit = ConceptColumnResolution(
        requested="revenue", resolved=True, candidates=[REGION, REVENUE]
    )
    assert [c.column_name for c in _refine(numeric_hit, AggregationType.SUM).candidates] == [
        "revenue"
    ]


def test_count_cue_does_not_fire_on_names_containing_number() -> None:
    assert "phone number" in _search_terms("phone number")
    assert "sales" in _search_terms("sales count")
    assert "sales" in _search_terms("sales_count")
    assert "units" in _search_terms("units sold")
    assert "aov" in _search_terms("average order value")
    # "phone number" must not be treated as a COUNT cue entity search.
    assert "phone" in _search_terms("phone number")
    from app.ai.metadata_resolver import _COUNT_CUE_RE

    assert not _COUNT_CUE_RE.search("phone number")


def test_glossary_parse_match_and_sql_rejection() -> None:
    entries = parse_glossary(
        [
            {
                "term": "sales",
                "synonyms": ["turnover"],
                "column": "sales.revenue",
                "aggregation": "SUM",
                "definition": "Booked revenue in USD",
            },
            {"term": "bad", "definition": "SELECT 1; DROP TABLE users"},
            "not-an-entry",
        ]
    )
    assert [entry.term for entry in entries] == ["sales"]
    assert match_glossary(entries, "total sales") is entries[0]
    assert match_glossary(entries, "Turnover") is entries[0]
    assert match_glossary(entries, "margin") is None
    with pytest.raises(ValidationError):
        GlossaryEntry(term="x", column="revenue; drop table")


def test_glossary_points_concept_at_declared_column() -> None:
    entry = GlossaryEntry(term="sales", column="sales.revenue", aggregation=AggregationType.SUM)
    work, hits = _apply_glossary([_concept("metric", "sales")], [entry])
    assert work[0].terms == ("revenue",)
    assert (work[0].table_name, work[0].column_name) == ("sales", "revenue")
    assert hits == {"metric:sales": entry}
    resolved = ConceptColumnResolution(requested="sales", resolved=True, candidates=[REVENUE])
    note = _refine(resolved, entry=entry).resolution_note or ""
    assert note.startswith("glossary: sales = SUM of sales.revenue")


def test_schema_context_includes_descriptions_glossary_and_notes() -> None:
    metadata = ResolvedMetadataContext(
        data_source_id=uuid.uuid4(),
        tables=[TABLE],
        columns=COLUMNS,
        resolved_metrics=[
            ConceptColumnResolution(
                requested="sales count",
                resolved=True,
                candidates=[SALE_ID],
                resolution_note="COUNT(*) of public.sales rows",
            )
        ],
        glossary=["aov = revenue / orders"],
    )
    text = build_schema_prompt_context(metadata).text
    assert "public.sales.revenue (numeric) - Booked revenue in USD" in text
    assert "Business glossary" in text and "aov = revenue / orders" in text
    assert "- sales count: COUNT(*) of public.sales rows" in text


def test_unmatched_metric_clarification_offers_real_numeric_fields() -> None:
    metadata = ResolvedMetadataContext(
        data_source_id=uuid.uuid4(),
        tables=[TABLE],
        columns=COLUMNS,
        resolved_metrics=[_unresolved("profit margin")],
    )
    intent = AIIntent(intent=AIIntentType.AGGREGATION, requires_data_access=True)
    plan = AIRequestPlan(intent=AIIntentType.AGGREGATION, requires_database=True)
    _intent, clarified, _meta = _apply_clarification_policy(intent, plan, metadata)
    question = clarified.clarification_question or ""
    assert clarified.requires_clarification is True
    assert "profit margin" in question
    assert "sales.revenue" in question and "sales.orders" in question
    assert "sale_id" not in question and "customer_id" not in question
