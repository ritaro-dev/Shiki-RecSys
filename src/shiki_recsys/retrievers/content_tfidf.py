import numpy as np
import pandas as pd

from shiki_recsys.features.content_items import ContentItemFeatures
from shiki_recsys.features.content_users import ContentUserProfiles


class ContentTFIDFRetriever:
    """Формирует персональные кандидаты по TF-IDF content-профилю."""

    def __init__(self) -> None:
        """Инициализирует TF-IDF content retriever."""

        self._item_features: ContentItemFeatures | None = None
        self._user_profiles: ContentUserProfiles | None = None
        self._supported_anime_ids: frozenset[int] = frozenset()
        self._is_fitted = False

    @property
    def supported_anime_ids(self) -> frozenset[int]:
        """
        Возвращает идентификаторы поддерживаемых аниме.

        Returns:
            Идентификаторы аниме из content-представления каталога.

        Raises:
            RuntimeError: Если retriever ещё не обучен.
        """
        self._require_fitted()
        return self._supported_anime_ids

    def fit(
        self,
        item_features: ContentItemFeatures,
        user_profiles: ContentUserProfiles,
    ) -> "ContentTFIDFRetriever":
        """
        Сохраняет item-признаки и пользовательские content-профили.

        Args:
            item_features: TF-IDF content-представление каталога.
            user_profiles: L2-нормализованные content-профили пользователей.

        Returns:
            Обученный retriever.

        Raises:
            ValueError: Если размерности item-признаков
                и пользовательских профилей несовместимы.
        """
        if (
            item_features.item_feature_matrix.shape[1]
            != user_profiles.user_profile_matrix.shape[1]
        ):
            raise ValueError(
                "Размерности item-признаков и пользовательских "
                "content-профилей не совпадают."
            )

        self._item_features = item_features
        self._user_profiles = user_profiles
        self._supported_anime_ids = frozenset(
            int(anime_id) for anime_id in item_features.raw_anime_ids
        )
        self._is_fitted = True

        return self

    def retrieve(
        self,
        *,
        user_id: int,
        candidate_count: int | None = None,
    ) -> pd.DataFrame:
        """
        Возвращает ранжированных content-кандидатов пользователя.

        Args:
            user_id: Идентификатор пользователя.
            candidate_count: Максимальное количество кандидатов.
                Значение None означает возврат полного рейтинга.

        Returns:
            Таблицу идентификаторов, scores, источников
            и позиций кандидатов.

        Raises:
            RuntimeError: Если retriever ещё не обучен.
            ValueError: Если количество кандидатов некорректно.
        """
        self._require_fitted()

        if candidate_count is not None and candidate_count <= 0:
            raise ValueError("candidate_count должен быть больше 0 или равен None.")

        assert self._item_features is not None
        assert self._user_profiles is not None

        inner_user_id = self._user_profiles.user_to_inner.get(user_id)

        if inner_user_id is None:
            return self._empty_candidates()

        user_profile = self._user_profiles.user_profile_matrix.getrow(inner_user_id)

        scores = (
            (self._item_features.item_feature_matrix @ user_profile.T).toarray().ravel()
        )

        candidates = (
            pd.DataFrame(
                {
                    "anime_id": self._item_features.raw_anime_ids,
                    "score": scores.astype("float64"),
                }
            )
            .sort_values(
                [
                    "score",
                    "anime_id",
                ],
                ascending=[
                    False,
                    True,
                ],
                kind="stable",
            )
            .reset_index(drop=True)
        )

        candidates["source"] = pd.Series(
            "content_tfidf",
            index=candidates.index,
            dtype="string",
        )

        candidates["source_rank"] = np.arange(
            1,
            len(candidates) + 1,
            dtype=np.int32,
        )

        candidates = candidates[
            [
                "anime_id",
                "score",
                "source",
                "source_rank",
            ]
        ]

        if candidate_count is not None:
            candidates = candidates.head(candidate_count)

        return candidates.copy().reset_index(drop=True)

    def _require_fitted(self) -> None:
        """
        Проверяет состояние retriever.

        Raises:
            RuntimeError: Если retriever ещё не обучен.
        """
        if not self._is_fitted:
            raise RuntimeError("ContentTFIDFRetriever ещё не обучен.")

    @staticmethod
    def _empty_candidates() -> pd.DataFrame:
        """
        Создаёт пустую таблицу кандидатов.

        Returns:
            Пустую таблицу стандартного формата.
        """
        return pd.DataFrame(
            {
                "anime_id": pd.Series(dtype="int64"),
                "score": pd.Series(dtype="float64"),
                "source": pd.Series(dtype="string"),
                "source_rank": pd.Series(dtype="int32"),
            }
        )
