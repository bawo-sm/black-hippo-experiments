from datetime import datetime
from sqlalchemy import (
    DateTime,
    func,
    BIGINT,
    Text
)
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import Enum as SQLAlchemyEnum
from src.common.enums import TaskEnum, TaskStatusEnum


class Base(DeclarativeBase):
    pass


class SQLTaskStatus(Base):
    __tablename__ = "task_status"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(
        BIGINT, 
        primary_key=True, 
        unique=True, 
        nullable=False, 
        autoincrement=True
    )
    task_uuid: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    task: Mapped[TaskEnum] = mapped_column(
        SQLAlchemyEnum(
            TaskEnum,
            values_callable=lambda x: [e.value for e in x],
            name="task",
            native_enum=True,
        ),
        nullable=False,
    )
    status: Mapped[TaskStatusEnum] = mapped_column(
        SQLAlchemyEnum(
            TaskStatusEnum,
            values_callable=lambda x: [e.value for e in x],
            name="status",
            native_enum=True,
        ),
        nullable=False,
    )
    info: Mapped[str] = mapped_column(
        Text,
        default=None,
        nullable=True,
    )
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now()
    )

    def __repr__(self) -> str:
        return (
            f"<SQLTaskStatus(id={self.id}, task_uuid={self.task_uuid}, task={self.task}, "
            f"status={self.status}, info={self.info}, "
            f"created_at={self.created_at}, updated_at={self.updated_at})>"
        )


class SQLItem(Base):
    __tablename__ = "items"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(
        BIGINT, 
        primary_key=True, 
        unique=True, 
        nullable=False, 
        autoincrement=True
    )
    origin_id: Mapped[int] = mapped_column(
        BIGINT, 
        unique=True, 
        nullable=False
    )
    season: Mapped[str] = mapped_column(Text, nullable=False)
    supplier_name: Mapped[str] = mapped_column(Text, nullable=False)
    supplier_reference_description: Mapped[str] = mapped_column(Text, nullable=False)
    materials: Mapped[str] = mapped_column(Text, default=None, nullable=True)
    main: Mapped[str] = mapped_column(Text, default=None, nullable=True)
    sub: Mapped[str] = mapped_column(Text, default=None, nullable=True)
    detail: Mapped[str] = mapped_column(Text, default=None, nullable=True)
    level4: Mapped[str] = mapped_column(Text, default=None, nullable=True)
    colors: Mapped[str] = mapped_column(Text, default=None, nullable=True)
    hs_code: Mapped[str] = mapped_column(Text, default=None, nullable=True)

    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now()
    )

    def __repr__(self) -> str:
        return (
            f"<SQLItems(id={self.id}, origin_id={self.origin_id}, "
            f"supplier_name={self.supplier_name}, supplier_reference_description={self.supplier_reference_description}, "
            f"materials={self.materials}, main={self.main}, sub={self.sub}, detail={self.detail}, "
            f"level4={self.level4}, colors={self.colors}, hs_code={self.hs_code}"
            f"created_at={self.created_at}, updated_at={self.updated_at})>"
        )