"""add metadata search and relationship indexes

Revision ID: b91c4e7f6a12
Revises: e7b2c91d4a08
Create Date: 2026-08-17 16:40:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b91c4e7f6a12"
down_revision: Union[str, Sequence[str], None] = "e7b2c91d4a08"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_data_source_schemas_data_source_id_lower_name",
        "data_source_schemas",
        ["data_source_id", sa.text("lower(name)")],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        "ix_data_source_tables_schema_id_lower_name",
        "data_source_tables",
        ["schema_id", sa.text("lower(name)")],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        "ix_data_source_columns_table_id_lower_name",
        "data_source_columns",
        ["table_id", sa.text("lower(name)")],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        op.f("ix_data_source_relationships_source_column_id"),
        "data_source_relationships",
        ["source_column_id"],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        op.f("ix_data_source_relationships_target_column_id"),
        "data_source_relationships",
        ["target_column_id"],
        unique=False,
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_data_source_relationships_target_column_id"),
        table_name="data_source_relationships",
        if_exists=True,
    )
    op.drop_index(
        op.f("ix_data_source_relationships_source_column_id"),
        table_name="data_source_relationships",
        if_exists=True,
    )
    op.drop_index(
        "ix_data_source_columns_table_id_lower_name",
        table_name="data_source_columns",
        if_exists=True,
    )
    op.drop_index(
        "ix_data_source_tables_schema_id_lower_name",
        table_name="data_source_tables",
        if_exists=True,
    )
    op.drop_index(
        "ix_data_source_schemas_data_source_id_lower_name",
        table_name="data_source_schemas",
        if_exists=True,
    )
