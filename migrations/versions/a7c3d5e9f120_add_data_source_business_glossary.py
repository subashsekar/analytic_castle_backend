"""add data source business glossary

Revision ID: a7c3d5e9f120
Revises: e6f1a3b04c27
Create Date: 2026-09-28 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a7c3d5e9f120"
down_revision: str | Sequence[str] | None = "e6f1a3b04c27"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "data_sources",
        sa.Column("business_glossary", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("data_sources", "business_glossary")
