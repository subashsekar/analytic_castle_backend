"""User report persistence (Phase 9.4).

Reports are structured narratives derived from Phase 9.1–9.3 outputs.
They store the client-provided content (question/summary/findings/charts)
and a JSON supporting payload for UI rendering.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class Report(Base):
    """Persisted user report.

    Tenancy:
      - scoped by `workspace_id`
      - tenant isolation by `organization_id`
      - ownership by `user_id`
    """

    __tablename__ = "reports"

    __table_args__ = (
        CheckConstraint(
            "length(btrim(question)) > 0",
            name="ck_reports_question_not_empty",
        ),
        CheckConstraint(
            "length(btrim(summary)) > 0",
            name="ck_reports_summary_not_empty",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    organization_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    data_source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("data_sources.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    question: Mapped[str] = mapped_column(String(32000), nullable=False)
    summary: Mapped[str] = mapped_column(String(32000), nullable=False)

    # Stored content. Keep as JSON to avoid rigid coupling to evolving UI.
    key_findings: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        server_default="[]",
    )
    chart_specifications: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        server_default="[]",
    )
    supporting_data: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        default=None,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"Report(id={self.id!r}, user_id={self.user_id!r}, "
            f"workspace_id={self.workspace_id!r})"
        )

