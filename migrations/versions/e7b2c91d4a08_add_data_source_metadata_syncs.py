"""add data source metadata syncs

Revision ID: e7b2c91d4a08
Revises: c8f3a91b2d47
Create Date: 2026-08-17 15:20:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e7b2c91d4a08"
down_revision: Union[str, Sequence[str], None] = "c8f3a91b2d47"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "data_source_metadata_syncs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("data_source_id", sa.Uuid(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING",
                "RUNNING",
                "SUCCESS",
                "FAILED",
                name="metadata_sync_status",
            ),
            server_default="PENDING",
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("schema_count", sa.Integer(), nullable=True),
        sa.Column("table_count", sa.Integer(), nullable=True),
        sa.Column("column_count", sa.Integer(), nullable=True),
        sa.Column("relationship_count", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "schema_count IS NULL OR schema_count >= 0",
            name="ck_data_source_metadata_syncs_schema_count_non_negative",
        ),
        sa.CheckConstraint(
            "table_count IS NULL OR table_count >= 0",
            name="ck_data_source_metadata_syncs_table_count_non_negative",
        ),
        sa.CheckConstraint(
            "column_count IS NULL OR column_count >= 0",
            name="ck_data_source_metadata_syncs_column_count_non_negative",
        ),
        sa.CheckConstraint(
            "relationship_count IS NULL OR relationship_count >= 0",
            name="ck_data_source_metadata_syncs_relationship_count_non_negative",
        ),
        sa.ForeignKeyConstraint(
            ["data_source_id"],
            ["data_sources.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "data_source_id",
            name="uq_data_source_metadata_syncs_data_source_id",
        ),
    )


def downgrade() -> None:
    op.drop_table("data_source_metadata_syncs")
    sa.Enum(name="metadata_sync_status").drop(op.get_bind(), checkfirst=True)
