"""add failed agent phase

Revision ID: c4d9e1f82a05
Revises: f3a8b2c91d04
Create Date: 2026-08-31 16:46:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "c4d9e1f82a05"
down_revision: str | Sequence[str] | None = "f3a8b2c91d04"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TYPE agent_phase ADD VALUE IF NOT EXISTS 'FAILED'")


def downgrade() -> None:
    # PostgreSQL cannot remove enum values safely without recreating the type.
    pass
