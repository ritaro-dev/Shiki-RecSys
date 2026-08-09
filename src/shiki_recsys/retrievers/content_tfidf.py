import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from shiki_recsys.features.content_items import ContentItemFeatures
from shiki_recsys.features.content_users import ContentUserProfiles
from shiki_recsys.retrievers.common import (
    RetrieverName,
    build_candidate_frame,
    empty_candidates,
    exclude_scored_items,
    validate_candidate_count,
)


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

    @property
    def item_features(self) -> ContentItemFeatures:
        """
        Возвращает content-представление каталога.

        Returns:
            Content-признаки текущего artifact.

        Raises:
            RuntimeError: Если retriever ещё не обучен.
        """
        self._require_fitted()

        assert self._item_features is not None
        return self._item_features

    def retrieve_from_profile(
        self,
        *,
        profile: csr_matrix,
        candidate_count: int | None = None,
        exclude_anime_ids: set[int] | None = None,
    ) -> pd.DataFrame:
        """
        Возвращает кандидатов по внешнему content-профилю.

        Args:
            profile: L2-нормализованный профиль размерности 1 x n_features.
            candidate_count: Максимальное количество кандидатов.
            exclude_anime_ids: Идентификаторы исключаемых аниме.

        Returns:
            Ранжированных content-кандидатов.

        Raises:
            RuntimeError: Если retriever ещё не обучен.
            ValueError: Если размерность профиля некорректна.
        """
        self._require_fitted()
        validate_candidate_count(candidate_count)

        assert self._item_features is not None

        expected_shape = (
            1,
            self._item_features.item_feature_matrix.shape[1],
        )
        if profile.shape != expected_shape:
            raise ValueError(
                f"profile должен иметь размерность {expected_shape}, "
                f"получено {profile.shape}."
            )

        scores = (self._item_features.item_feature_matrix @ profile.T).toarray().ravel()

        anime_ids, scores = exclude_scored_items(
            self._item_features.raw_anime_ids,
            scores,
            exclude_anime_ids,
        )

        ranked_items = (
            pd.DataFrame(
                {
                    "anime_id": anime_ids,
                    "score": scores,
                }
            )
            .sort_values(
                ["score", "anime_id"],
                ascending=[False, True],
                kind="stable",
            )
            .reset_index(drop=True)
        )

        return build_candidate_frame(
            anime_ids=ranked_items["anime_id"].to_numpy(),
            scores=ranked_items["score"].to_numpy(),
            source=RetrieverName.CONTENT_TFIDF,
            candidate_count=candidate_count,
        )

    def supports_user(self, user_id: int) -> bool:
        """
        Проверяет поддержку пользователя retriever-ом.

        Args:
            user_id: Идентификатор пользователя.

        Returns:
            True, если для пользователя существует content-профиль.

        Raises:
            RuntimeError: Если retriever ещё не обучен.
        """
        self._require_fitted()

        assert self._user_profiles is not None
        return user_id in self._user_profiles.user_to_inner

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
        exclude_anime_ids: set[int] | None = None,
    ) -> pd.DataFrame:
        """
        Возвращает ранжированных content-кандидатов пользователя.

        Args:
            user_id: Идентификатор пользователя.
            candidate_count: Максимальное количество кандидатов.
                Значение None означает возврат полного рейтинга.
            exclude_anime_ids: Идентификаторы аниме,
                исключаемые из выдачи.

        Returns:
            Таблицу идентификаторов, scores, источников
            и позиций кандидатов.

        Raises:
            RuntimeError: Если retriever ещё не обучен.
            ValueError: Если количество кандидатов некорректно.
        """
        self._require_fitted()

        validate_candidate_count(candidate_count)

        assert self._item_features is not None
        assert self._user_profiles is not None

        inner_user_id = self._user_profiles.user_to_inner.get(user_id)

        if inner_user_id is None:
            return empty_candidates()

        user_profile = self._user_profiles.user_profile_matrix.getrow(inner_user_id)

        scores = (
            (self._item_features.item_feature_matrix @ user_profile.T).toarray().ravel()
        )

        anime_ids = self._item_features.raw_anime_ids

        anime_ids, scores = exclude_scored_items(
            anime_ids,
            scores,
            exclude_anime_ids,
        )

        ranked_items = (
            pd.DataFrame(
                {
                    "anime_id": anime_ids,
                    "score": scores,
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

        return build_candidate_frame(
            anime_ids=ranked_items["anime_id"].to_numpy(),
            scores=ranked_items["score"].to_numpy(),
            source=RetrieverName.CONTENT_TFIDF,
            candidate_count=candidate_count,
        )

    def _require_fitted(self) -> None:
        """
        Проверяет состояние retriever.

        Raises:
            RuntimeError: Если retriever ещё не обучен.
        """
        if not self._is_fitted:
            raise RuntimeError("ContentTFIDFRetriever ещё не обучен.")

    def score_items(
        self,
        *,
        user_id: int,
        anime_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Рассчитывает TF-IDF scores для заданных аниме.

        Args:
            user_id: Идентификатор пользователя.
            anime_ids: Идентификаторы оцениваемых аниме.

        Returns:
            Scores в порядке переданных anime_id.
        """
        self._require_fitted()

        scores = np.full(
            len(anime_ids),
            np.nan,
            dtype=np.float64,
        )

        assert self._item_features is not None
        assert self._user_profiles is not None

        inner_user_id = self._user_profiles.user_to_inner.get(user_id)

        if inner_user_id is None:
            return scores

        supported_positions = [
            position
            for position, anime_id in enumerate(anime_ids)
            if int(anime_id) in self._item_features.anime_to_inner
        ]

        if not supported_positions:
            return scores

        inner_anime_ids = np.array(
            [
                self._item_features.anime_to_inner[int(anime_ids[position])]
                for position in supported_positions
            ],
            dtype=np.int64,
        )

        user_profile = self._user_profiles.user_profile_matrix.getrow(inner_user_id)

        scores[supported_positions] = (
            (self._item_features.item_feature_matrix[inner_anime_ids] @ user_profile.T)
            .toarray()
            .ravel()
        )

        return scores
