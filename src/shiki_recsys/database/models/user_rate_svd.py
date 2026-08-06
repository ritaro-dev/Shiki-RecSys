from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base


class UserRateSVD(Base):
    """
    Полная история взаимодействий пользователя,
    собранная для обучения рекомендательных моделей.
    """

    __tablename__ = "user_rates_svd"

    __table_args__ = (
        CheckConstraint(
            "rating >= 0 AND rating <= 10",
        ),
        UniqueConstraint(
            "user_id",
            "anime_id",
            name="unique_svd_user_anime",
        ),
        Index(
            "idx_user_rates_svd_user_id",
            "user_id",
        ),
        Index(
            "idx_user_rates_svd_anime_id",
            "anime_id",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    user_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    anime_id: Mapped[int] = mapped_column(
        ForeignKey(
            "animes.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    rating: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    status: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
