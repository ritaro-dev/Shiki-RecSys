from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..models.anime import Anime


class AnimeRepository:
    """Provide persistence operations for the anime catalog."""

    def get_all_ids(
        self,
        session: Session,
    ) -> set[int]:
        """
        Return IDs of all stored anime.

        Args:
            session: Database session.

        Returns:
            Stored anime IDs.
        """

        statement = select(Anime.id)

        return set(session.scalars(statement).all())

    def get_titles_by_ids(
        self,
        session: Session,
        *,
        anime_ids: Sequence[int],
    ) -> list[dict[str, object]]:
        """
        Return title fields for the requested anime.

        Args:
            session: Database session.
            anime_ids: Anime IDs to retrieve.

        Returns:
            Anime IDs with default and Russian titles.
        """
        if not anime_ids:
            return []

        statement = (
            select(
                Anime.id,
                Anime.name,
                Anime.russian_name,
            )
            .where(Anime.id.in_(anime_ids))
            .order_by(Anime.id)
        )

        rows = session.execute(statement).mappings().all()

        return [dict(row) for row in rows]

    def get_all(
        self,
        session: Session,
    ) -> list[dict[str, Any]]:
        """
        Return the stored anime catalog.

        Args:
            session: Database session.

        Returns:
            Catalog rows ordered by anime ID.
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
        Insert or update anime catalog rows.

        The caller is responsible for transaction management.

        Args:
            session: Database session.
            anime_rows: Anime rows to persist.

        Returns:
            Number of processed rows.
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
