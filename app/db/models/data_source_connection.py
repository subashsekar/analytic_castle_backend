import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base

if TYPE_CHECKING:
    from app.db.models.data_source import DataSource

_DEFAULT_POSTGRES_PORT = 5432
_DEFAULT_SSL_MODE = "prefer"


class DataSourceConnection(Base):
    __tablename__ = "data_source_connections"
    __table_args__ = (
        UniqueConstraint(
            "data_source_id",
            name="uq_data_source_connections_data_source_id",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    data_source_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("data_sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    host: Mapped[str] = mapped_column(String(255), nullable=False)
    port: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=_DEFAULT_POSTGRES_PORT,
        server_default=str(_DEFAULT_POSTGRES_PORT),
    )
    database_name: Mapped[str] = mapped_column(String(255), nullable=False)
    username: Mapped[str] = mapped_column(String(255), nullable=False)
    # Ciphertext from the credential service. Never persist a plaintext password.
    encrypted_password: Mapped[str] = mapped_column(String(1024), nullable=False)
    ssl_mode: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=_DEFAULT_SSL_MODE,
        server_default=_DEFAULT_SSL_MODE,
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
    )  # ORM updates only; raw SQL will not bump this.

    data_source: Mapped["DataSource"] = relationship(back_populates="connection")

    def __repr__(self) -> str:
        return (
            f"DataSourceConnection(id={self.id!r}, "
            f"data_source_id={self.data_source_id!r})"
        )
