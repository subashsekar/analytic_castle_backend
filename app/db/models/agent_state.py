import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, CheckConstraint, DateTime, Enum, ForeignKey, Integer, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base
from app.enums import AgentPhase

if TYPE_CHECKING:
    from app.db.models.analysis_session import AnalysisSession


class AgentState(Base):
    """Current typed agent state for one analysis session."""

    __tablename__ = "agent_states"
    __table_args__ = (
        CheckConstraint("version >= 1", name="ck_agent_states_version_positive"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("analysis_sessions.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    phase: Mapped[AgentPhase] = mapped_column(
        Enum(AgentPhase, name="agent_phase", native_enum=True),
        nullable=False,
        default=AgentPhase.INITIAL,
        server_default=AgentPhase.INITIAL.value,
    )
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        server_default="{}",
    )
    version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )
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

    session: Mapped["AnalysisSession"] = relationship(back_populates="agent_state")

    def __repr__(self) -> str:
        return (
            f"AgentState(id={self.id!r}, session_id={self.session_id!r}, "
            f"phase={self.phase!r}, version={self.version!r})"
        )
