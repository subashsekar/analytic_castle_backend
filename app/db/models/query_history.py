"""Query history model for generated and executed SQL lifecycle records."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, CheckConstraint, DateTime, Enum, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base
from app.enums import QueryHistoryStatus

if TYPE_CHECKING:
    from app.db.models.data_source import DataSource
    from app.db.models.organization import Organization
    from app.db.models.user import User
    from app.db.models.workspace import Workspace


class QueryHistory(Base):
    """Persisted query lifecycle record.

    Stores generated SQL, execution status, duration, and result/error metadata
    without retaining sensitive credentials or raw customer result rows.
    """

    __tablename__ = "query_history"
    __table_args__ = (
        CheckConstraint(
            "length(btrim(generated_sql)) > 0",
            name="ck_query_history_generated_sql_not_empty",
        ),
        CheckConstraint(
            "duration_ms >= 0",
            name="ck_query_history_duration_ms_non_negative",
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
    data_source_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("data_sources.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    generated_sql: Mapped[str] = mapped_column(String(4000), nullable=False)
    validated_sql: Mapped[str | None] = mapped_column(String(4000), nullable=True)
    corrected_sql: Mapped[str | None] = mapped_column(String(4000), nullable=True)
    status: Mapped[QueryHistoryStatus] = mapped_column(
        Enum(
            QueryHistoryStatus,
            name="query_history_status",
            native_enum=True,
        ),
        nullable=False,
        index=True,
    )
    duration_ms: Mapped[float] = mapped_column(nullable=False, default=0.0)
    result_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        default=None,
    )
    error_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        default=None,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user: Mapped[User] = relationship()
    workspace: Mapped[Workspace] = relationship()
    organization: Mapped[Organization] = relationship()
    data_source: Mapped[DataSource | None] = relationship()

    def __repr__(self) -> str:
        return (
            f"QueryHistory(id={self.id!r}, user_id={self.user_id!r}, "
            f"workspace_id={self.workspace_id!r}, status={self.status!r})"
        )
