"""Static catalog of analyticcastle_demo, used to validate SQL offline."""

from __future__ import annotations

import uuid

from app.ai.metadata_types import (
    MetadataColumnCandidate,
    MetadataMatchReason,
    MetadataPrimaryKey,
    MetadataTableCandidate,
    ResolvedMetadataContext,
)

DEMO_SALES_COLUMNS: tuple[tuple[str, str], ...] = (
    ("sale_id", "integer"),
    ("sale_date", "date"),
    ("product_id", "integer"),
    ("region", "text"),
    ("units", "integer"),
    ("revenue", "numeric"),
    ("orders", "integer"),
    ("aov", "numeric"),
    ("conversions", "integer"),
)

DEMO_PRODUCT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("product_id", "integer"),
    ("product_name", "text"),
    ("category", "text"),
)

# Back-compat for imports that still expect DEMO_COLUMNS.
DEMO_COLUMNS = DEMO_SALES_COLUMNS


def demo_metadata() -> ResolvedMetadataContext:
    sales_id = uuid.UUID(int=1)
    products_id = uuid.UUID(int=2)
    sales_columns = [
        MetadataColumnCandidate(
            column_id=uuid.UUID(int=100 + index),
            table_id=sales_id,
            schema_name="public",
            table_name="sales",
            column_name=name,
            data_type=data_type,
            is_primary_key=name == "sale_id",
            match_reason=MetadataMatchReason.EXACT,
            relevance_score=100,
        )
        for index, (name, data_type) in enumerate(DEMO_SALES_COLUMNS)
    ]
    product_columns = [
        MetadataColumnCandidate(
            column_id=uuid.UUID(int=200 + index),
            table_id=products_id,
            schema_name="public",
            table_name="products",
            column_name=name,
            data_type=data_type,
            is_primary_key=name == "product_id",
            match_reason=MetadataMatchReason.EXACT,
            relevance_score=100,
        )
        for index, (name, data_type) in enumerate(DEMO_PRODUCT_COLUMNS)
    ]
    sales = MetadataTableCandidate(
        table_id=sales_id,
        schema_name="public",
        table_name="sales",
        match_reason=MetadataMatchReason.EXACT,
        relevance_score=100,
        primary_key_columns=[
            MetadataPrimaryKey(column_id=sales_columns[0].column_id, column_name="sale_id")
        ],
    )
    products = MetadataTableCandidate(
        table_id=products_id,
        schema_name="public",
        table_name="products",
        match_reason=MetadataMatchReason.EXACT,
        relevance_score=100,
        primary_key_columns=[
            MetadataPrimaryKey(
                column_id=product_columns[0].column_id, column_name="product_id"
            )
        ],
    )
    return ResolvedMetadataContext(
        data_source_id=uuid.UUID(int=0),
        tables=[sales, products],
        columns=[*sales_columns, *product_columns],
        resolved_time_columns=[sales_columns[1]],
    )
