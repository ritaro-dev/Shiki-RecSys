from decimal import Decimal

from sqlalchemy import (
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base


class Anime(Base):
    """
    Аниме из каталога Shikimori.
    """

    __tablename__ = "animes"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=False,
    )

    name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    russian_name: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    kind: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    status: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    score: Mapped[Decimal | None] = mapped_column(
        Numeric(4, 2),
        nullable=True,
    )

    score_std: Mapped[Decimal | None] = mapped_column(
        Numeric(6, 4),
        nullable=True,
    )

    episodes: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    duration: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    rating: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    genres: Mapped[list[str] | None] = mapped_column(
        ARRAY(Text),
        nullable=True,
    )

    studios: Mapped[list[str] | None] = mapped_column(
        ARRAY(Text),
        nullable=True,
    )

    stat_completed: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    stat_dropped: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    stat_watching: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    stat_planned: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
