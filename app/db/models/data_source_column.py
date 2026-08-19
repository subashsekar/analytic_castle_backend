import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    false,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.db.session import Base

if TYPE_CHECKING:
    from app.db.models.data_source_table import DataSourceTable


class DataSourceColumn(Base):
    __tablename__ = "data_source_columns"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    table_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("data_source_tables.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    ordinal_position: Mapped[int] = mapped_column(Integer, nullable=False)
    data_type: Mapped[str] = mapped_column(String(255), nullable=False)
    database_type: Mapped[str] = mapped_column(String(255), nullable=False)
    is_nullable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    # Database default expression stored as metadata only; never execute.
    default_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_primary_key: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=false(),
    )
    is_unique: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=false(),
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
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

    __table_args__ = (
        UniqueConstraint(
            "table_id",
            "name",
            name="uq_data_source_columns_table_id_name",
        ),
        UniqueConstraint(
            "id",
            "table_id",
            name="uq_data_source_columns_id_table_id",
        ),
        CheckConstraint(
            "length(btrim(name)) > 0",
            name="ck_data_source_columns_name_not_empty",
        ),
        CheckConstraint(
            "ordinal_position > 0",
            name="ck_data_source_columns_ordinal_position_positive",
        ),
        Index(
            "ix_data_source_columns_table_id_ordinal_position",
            "table_id",
            "ordinal_position",
        ),
        Index(
            "ix_data_source_columns_table_id_lower_name",
            "table_id",
            func.lower(name),
        ),
    )

    table: Mapped["DataSourceTable"] = relationship(back_populates="columns")

    @validates("name")
    def _normalize_name(self, _key: str, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("name cannot be empty")
        return stripped

    def __repr__(self) -> str:
        return f"DataSourceColumn(id={self.id!r}, name={self.name!r})"
