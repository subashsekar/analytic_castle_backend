import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base
from app.enums import AnalysisSessionStatus

if TYPE_CHECKING:
    from app.db.models.agent_state import AgentState
    from app.db.models.conversation_context import ConversationContext
    from app.db.models.data_source import DataSource
    from app.db.models.organization import Organization
    from app.db.models.user import User
    from app.db.models.workspace import Workspace


class AnalysisSession(Base):
    """Lifecycle container for one agent analysis run."""

    __tablename__ = "analysis_sessions"

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
    status: Mapped[AnalysisSessionStatus] = mapped_column(
        Enum(
            AnalysisSessionStatus,
            name="analysis_session_status",
            native_enum=True,
        ),
        nullable=False,
        default=AnalysisSessionStatus.ACTIVE,
        server_default=AnalysisSessionStatus.ACTIVE.value,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user: Mapped["User"] = relationship()
    workspace: Mapped["Workspace"] = relationship()
    organization: Mapped["Organization"] = relationship()
    data_source: Mapped["DataSource | None"] = relationship()
    agent_state: Mapped["AgentState | None"] = relationship(
        back_populates="session",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    conversation_context: Mapped["ConversationContext | None"] = relationship(
        back_populates="session",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return (
            f"AnalysisSession(id={self.id!r}, "
            f"workspace_id={self.workspace_id!r}, status={self.status!r})"
        )
