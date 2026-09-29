"""add agent state tables

Revision ID: f3a8b2c91d04
Revises: b91c4e7f6a12
Create Date: 2026-08-31 12:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "f3a8b2c91d04"
down_revision: str | Sequence[str] | None = "b91c4e7f6a12"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

analysis_session_status = postgresql.ENUM(
    "ACTIVE",
    "COMPLETED",
    "FAILED",
    name="analysis_session_status",
    create_type=False,
)
agent_phase = postgresql.ENUM(
    "INITIAL",
    "INTENT",
    "PLANNING",
    "EXECUTING",
    "RESPONDING",
    "COMPLETED",
    "FAILED",
    name="agent_phase",
    create_type=False,
)


def upgrade() -> None:
    bind = op.get_bind()
    analysis_session_status.create(bind, checkfirst=True)
    agent_phase.create(bind, checkfirst=True)

    op.create_table(
        "analysis_sessions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("organization_id", sa.Uuid(), nullable=False),
        sa.Column("data_source_id", sa.Uuid(), nullable=True),
        sa.Column(
            "status",
            analysis_session_status,
            server_default="ACTIVE",
            nullable=False,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
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
        op.f("ix_analysis_sessions_data_source_id"),
        "analysis_sessions",
        ["data_source_id"],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        op.f("ix_analysis_sessions_organization_id"),
        "analysis_sessions",
        ["organization_id"],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        op.f("ix_analysis_sessions_user_id"),
        "analysis_sessions",
        ["user_id"],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        op.f("ix_analysis_sessions_workspace_id"),
        "analysis_sessions",
        ["workspace_id"],
        unique=False,
        if_not_exists=True,
    )

    op.create_table(
        "agent_states",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column(
            "phase",
            agent_phase,
            server_default="INITIAL",
            nullable=False,
        ),
        sa.Column("payload", sa.JSON(), server_default="{}", nullable=False),
        sa.Column("version", sa.Integer(), server_default="1", nullable=False),
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
        sa.CheckConstraint("version >= 1", name="ck_agent_states_version_positive"),
        sa.ForeignKeyConstraint(
            ["session_id"], ["analysis_sessions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id"),
        if_not_exists=True,
    )

    op.create_table(
        "conversation_contexts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("messages", sa.JSON(), server_default="[]", nullable=False),
        sa.Column("char_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("message_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("version", sa.Integer(), server_default="1", nullable=False),
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
            "char_count >= 0",
            name="ck_conversation_contexts_char_count_non_negative",
        ),
        sa.CheckConstraint(
            "message_count >= 0",
            name="ck_conversation_contexts_message_count_non_negative",
        ),
        sa.CheckConstraint(
            "version >= 1",
            name="ck_conversation_contexts_version_positive",
        ),
        sa.ForeignKeyConstraint(
            ["session_id"], ["analysis_sessions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id"),
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_table("conversation_contexts")
    op.drop_table("agent_states")
    op.drop_index(
        op.f("ix_analysis_sessions_workspace_id"), table_name="analysis_sessions"
    )
    op.drop_index(op.f("ix_analysis_sessions_user_id"), table_name="analysis_sessions")
    op.drop_index(
        op.f("ix_analysis_sessions_organization_id"), table_name="analysis_sessions"
    )
    op.drop_index(
        op.f("ix_analysis_sessions_data_source_id"), table_name="analysis_sessions"
    )
    op.drop_table("analysis_sessions")
    agent_phase.drop(op.get_bind(), checkfirst=True)
    analysis_session_status.drop(op.get_bind(), checkfirst=True)
