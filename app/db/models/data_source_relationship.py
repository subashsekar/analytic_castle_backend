import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, ForeignKeyConstraint, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base
from app.enums import DataSourceRelationshipType

if TYPE_CHECKING:
    from app.db.models.data_source_column import DataSourceColumn
    from app.db.models.data_source_table import DataSourceTable


class DataSourceRelationship(Base):
    """A discovered association between two metadata tables/columns.

    Foreign keys guarantee that source and target tables and columns exist, and
    composite foreign keys guarantee that each column belongs to its declared
    table. Same-data-source membership (both tables belonging to one DataSource)
    is enforced by application/service validation during metadata sync, not by a
    database trigger. Self-referential relationships are allowed: source_table_id
    and target_table_id may be the same table.
    """

    __tablename__ = "data_source_relationships"
    __table_args__ = (
        ForeignKeyConstraint(
            ["source_column_id", "source_table_id"],
            ["data_source_columns.id", "data_source_columns.table_id"],
            ondelete="CASCADE",
            name="fk_data_source_relationships_source_column_table",
        ),
        ForeignKeyConstraint(
            ["target_column_id", "target_table_id"],
            ["data_source_columns.id", "data_source_columns.table_id"],
            ondelete="CASCADE",
            name="fk_data_source_relationships_target_column_table",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    source_table_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("data_source_tables.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    source_column_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("data_source_columns.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    target_table_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("data_source_tables.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    target_column_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("data_source_columns.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    relationship_type: Mapped[DataSourceRelationshipType] = mapped_column(
        Enum(
            DataSourceRelationshipType,
            name="data_source_relationship_type",
            native_enum=True,
        ),
        nullable=False,
    )
    constraint_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
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

    source_table: Mapped["DataSourceTable"] = relationship(
        back_populates="source_relationships",
        foreign_keys=[source_table_id],
    )
    target_table: Mapped["DataSourceTable"] = relationship(
        back_populates="target_relationships",
        foreign_keys=[target_table_id],
    )
    source_column: Mapped["DataSourceColumn"] = relationship(
        foreign_keys=[source_column_id],
    )
    target_column: Mapped["DataSourceColumn"] = relationship(
        foreign_keys=[target_column_id],
    )

    def __repr__(self) -> str:
        return (
            f"DataSourceRelationship(id={self.id!r}, "
            f"relationship_type={self.relationship_type!r})"
        )
