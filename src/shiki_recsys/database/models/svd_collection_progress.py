from datetime import datetime

from sqlalchemy import (
    DateTime,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base


class SVDCollectionProgress(Base):
    """
    Состояние сбора полной истории пользователя
    для обучающего датасета.
    """

    __tablename__ = "svd_collection_progress"

    __table_args__ = (
        Index(
            "idx_svd_collection_progress_status",
            "status",
        ),
    )

    user_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=False,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    rates_saved: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
