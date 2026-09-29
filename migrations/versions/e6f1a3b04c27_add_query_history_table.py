"""add query history table

Revision ID: e6f1a3b04c27
Revises: d5e0f2a93b16
Create Date: 2026-09-15 10:50:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "e6f1a3b04c27"
down_revision: str | Sequence[str] | None = "d5e0f2a93b16"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

query_history_status = postgresql.ENUM(
    "SUCCEEDED",
    "FAILED",
    "REJECTED",
    "CORRECTED",
    name="query_history_status",
    create_type=False,
)


def upgrade() -> None:
    bind = op.get_bind()
    query_history_status.create(bind, checkfirst=True)

    op.create_table(
        "query_history",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("data_source_id", sa.Uuid(), nullable=True),
        sa.Column("generated_sql", sa.String(length=4000), nullable=False),
        sa.Column("validated_sql", sa.String(length=4000), nullable=True),
        sa.Column("corrected_sql", sa.String(length=4000), nullable=True),
        sa.Column("status", query_history_status, nullable=False),
        sa.Column("duration_ms", sa.Float(), nullable=False),
        sa.Column("result_metadata", sa.JSON(), nullable=True),
        sa.Column("error_metadata", sa.JSON(), nullable=True),
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
            "length(btrim(generated_sql)) > 0",
            name="ck_query_history_generated_sql_not_empty",
        ),
        sa.CheckConstraint(
            "duration_ms >= 0",
            name="ck_query_history_duration_ms_non_negative",
        ),
        sa.ForeignKeyConstraint(
            ["data_source_id"], ["data_sources.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"], ["organizations.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["workspace_id"], ["workspaces.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )
    op.create_index(
        op.f("ix_query_history_created_at"),
        "query_history",
        ["created_at"],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        op.f("ix_query_history_data_source_id"),
        "query_history",
        ["data_source_id"],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        op.f("ix_query_history_organization_id"),
        "query_history",
        ["organization_id"],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        op.f("ix_query_history_status"),
        "query_history",
        ["status"],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        op.f("ix_query_history_user_id"),
        "query_history",
        ["user_id"],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        op.f("ix_query_history_workspace_id"),
        "query_history",
        ["workspace_id"],
        unique=False,
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_query_history_workspace_id"), table_name="query_history")
    op.drop_index(op.f("ix_query_history_user_id"), table_name="query_history")
    op.drop_index(op.f("ix_query_history_status"), table_name="query_history")
    op.drop_index(op.f("ix_query_history_organization_id"), table_name="query_history")
    op.drop_index(op.f("ix_query_history_data_source_id"), table_name="query_history")
    op.drop_index(op.f("ix_query_history_created_at"), table_name="query_history")
    op.drop_table("query_history")
    query_history_status.drop(op.get_bind(), checkfirst=True)
