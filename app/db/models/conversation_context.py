import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, CheckConstraint, DateTime, ForeignKey, Integer, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base

if TYPE_CHECKING:
    from app.db.models.analysis_session import AnalysisSession


class ConversationContext(Base):
    """Serialized conversation messages for one analysis session."""

    __tablename__ = "conversation_contexts"
    __table_args__ = (
        CheckConstraint(
            "char_count >= 0",
            name="ck_conversation_contexts_char_count_non_negative",
        ),
        CheckConstraint(
            "message_count >= 0",
            name="ck_conversation_contexts_message_count_non_negative",
        ),
        CheckConstraint(
            "version >= 1",
            name="ck_conversation_contexts_version_positive",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("analysis_sessions.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    messages: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        server_default="[]",
    )
    char_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    message_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
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

    session: Mapped["AnalysisSession"] = relationship(
        back_populates="conversation_context"
    )

    def __repr__(self) -> str:
        return (
            f"ConversationContext(id={self.id!r}, session_id={self.session_id!r}, "
            f"message_count={self.message_count!r}, char_count={self.char_count!r})"
        )
