"""add reports table

Revision ID: c2f1a9b4d6e0
Revises: f3a8b2c91d04
Create Date: 2026-10-27 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c2f1a9b4d6e0"
down_revision: str | Sequence[str] | None = "f3a8b2c91d04"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "reports",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("data_source_id", sa.Uuid(), nullable=False),
        sa.Column("question", sa.String(length=32000), nullable=False),
        sa.Column("summary", sa.String(length=32000), nullable=False),
        sa.Column(
            "key_findings",
            sa.JSON(),
            server_default="[]",
            nullable=False,
        ),
        sa.Column(
            "chart_specifications",
            sa.JSON(),
            server_default="[]",
            nullable=False,
        ),
        sa.Column("supporting_data", sa.JSON(), nullable=True),
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
            "length(btrim(question)) > 0",
            name="ck_reports_question_not_empty",
        ),
        sa.CheckConstraint(
            "length(btrim(summary)) > 0",
            name="ck_reports_summary_not_empty",
        ),
        sa.ForeignKeyConstraint(
            ["data_source_id"], ["data_sources.id"], ondelete="CASCADE"
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
        op.f("ix_reports_created_at"),
        "reports",
        ["created_at"],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        op.f("ix_reports_data_source_id"),
        "reports",
        ["data_source_id"],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        op.f("ix_reports_organization_id"),
        "reports",
        ["organization_id"],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        op.f("ix_reports_user_id"),
        "reports",
        ["user_id"],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        op.f("ix_reports_workspace_id"),
        "reports",
        ["workspace_id"],
        unique=False,
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_reports_workspace_id"), table_name="reports")
    op.drop_index(op.f("ix_reports_user_id"), table_name="reports")
    op.drop_index(op.f("ix_reports_organization_id"), table_name="reports")
    op.drop_index(op.f("ix_reports_data_source_id"), table_name="reports")
    op.drop_index(op.f("ix_reports_created_at"), table_name="reports")
    op.drop_table("reports")

