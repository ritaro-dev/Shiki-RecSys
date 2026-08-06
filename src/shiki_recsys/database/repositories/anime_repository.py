from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..models.anime import Anime


class AnimeRepository:
    """
    Выполняет операции с таблицей animes.
    """

    def get_all_ids(
        self,
        session: Session,
    ) -> set[int]:
        """
        Возвращает идентификаторы всех аниме,
        сохранённых в таблице animes.
        """

        statement = select(Anime.id)

        return set(session.scalars(statement).all())

    def get_all(
        self,
        session: Session,
    ) -> list[dict[str, Any]]:
        """
        Возвращает все аниме из таблицы animes.

        Args:
            session: Сессия базы данных.

        Returns:
            Строки каталога, упорядоченные по идентификатору аниме.
        """

        statement = select(
            Anime.id,
            Anime.name,
            Anime.russian_name,
            Anime.kind,
            Anime.status,
            Anime.score,
            Anime.score_std,
            Anime.episodes,
            Anime.duration,
            Anime.rating,
            Anime.genres,
            Anime.studios,
            Anime.stat_completed,
            Anime.stat_dropped,
            Anime.stat_watching,
            Anime.stat_planned,
        ).order_by(Anime.id)

        rows = session.execute(statement).mappings().all()

        return [dict(row) for row in rows]

    def upsert_many(
        self,
        session: Session,
        anime_rows: Sequence[dict[str, Any]],
    ) -> int:
        """
        Добавляет или обновляет пачку аниме.

        Не выполняет commit: транзакцией управляет
        вызывающий код.
        """

        if not anime_rows:
            return 0

        statement = insert(Anime)

        statement = statement.on_conflict_do_update(
            index_elements=[Anime.id],
            set_={
                "name": statement.excluded.name,
                "russian_name": (statement.excluded.russian_name),
                "kind": statement.excluded.kind,
                "status": statement.excluded.status,
                "score": statement.excluded.score,
                "episodes": statement.excluded.episodes,
                "duration": statement.excluded.duration,
                "rating": statement.excluded.rating,
                "genres": statement.excluded.genres,
                "studios": statement.excluded.studios,
                "stat_completed": (statement.excluded.stat_completed),
                "stat_dropped": (statement.excluded.stat_dropped),
                "stat_watching": (statement.excluded.stat_watching),
                "stat_planned": (statement.excluded.stat_planned),
            },
        )

        session.execute(
            statement,
            list(anime_rows),
        )

        return len(anime_rows)
