"""add awaiting clarification agent phase

Revision ID: d5e0f2a93b16
Revises: c4d9e1f82a05
Create Date: 2026-09-03 12:50:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "d5e0f2a93b16"
down_revision: str | Sequence[str] | None = "c4d9e1f82a05"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TYPE agent_phase ADD VALUE IF NOT EXISTS 'AWAITING_CLARIFICATION'"
    )


def downgrade() -> None:
    # PostgreSQL cannot remove enum values safely without recreating the type.
    pass
