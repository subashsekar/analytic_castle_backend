import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Enum, Index, String, false, func, true
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.db.session import Base
from app.enums import UserRole

if TYPE_CHECKING:
    from app.db.models.data_source import DataSource
    from app.db.models.email_verification_token import EmailVerificationToken
    from app.db.models.password_reset_token import PasswordResetToken
    from app.db.models.refresh_token import RefreshToken
    from app.db.models.workspace_member import WorkspaceMember


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    first_name: Mapped[str] = mapped_column(String(255), nullable=False)
    last_name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, name="user_role", native_enum=True),
        nullable=False,
    )
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=false(),
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=true(),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )  # ORM updates only; raw SQL will not bump this.

    __table_args__ = (Index("ix_users_email_lower", func.lower(email), unique=True),)

    workspace_memberships: Mapped[list["WorkspaceMember"]] = relationship(
        back_populates="user",
    )
    refresh_tokens: Mapped[list["RefreshToken"]] = relationship(
        back_populates="user",
    )
    email_verification_tokens: Mapped[list["EmailVerificationToken"]] = relationship(
        back_populates="user",
    )
    password_reset_tokens: Mapped[list["PasswordResetToken"]] = relationship(
        back_populates="user",
    )
    created_data_sources: Mapped[list["DataSource"]] = relationship(
        back_populates="created_by_user",
        foreign_keys="DataSource.created_by",
    )

    @validates("email")
    def _normalize_email(self, _key: str, value: str) -> str:
        return value.strip().lower()

    def __repr__(self) -> str:
        return f"User(id={self.id!r}, email={self.email!r})"
