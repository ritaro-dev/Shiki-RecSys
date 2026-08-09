import numpy as np
import pandas as pd

from shiki_recsys.features.content_users import build_content_profile
from shiki_recsys.inference.user_state import UserState
from shiki_recsys.retrievers.content_tfidf import ContentTFIDFRetriever
from shiki_recsys.retrievers.popularity import PopularityRetriever


def build_new_user_ranking(
    *,
    interactions: pd.DataFrame,
    known_anime_ids: set[int],
    candidate_count: int,
    relevance_threshold: float,
    max_positive_items: int,
    popularity: PopularityRetriever,
    content_tfidf: ContentTFIDFRetriever,
) -> pd.DataFrame:
    """
    Формирует ranking для нового пользователя.

    Args:
        interactions: Актуальная история пользователя.
        known_anime_ids: Известные пользователю anime.
        candidate_count: Число кандидатов от каждого retriever-а.
        relevance_threshold: Минимальная положительная оценка.
        max_positive_items: Максимальное число positive items в профиле.
        popularity: Обученный popularity retriever.
        content_tfidf: Обученный TF-IDF retriever.

    Returns:
        Кандидатов, отсортированных по cold score.
    """
    profile = build_content_profile(
        interactions,
        content_tfidf.item_features,
        relevance_threshold=relevance_threshold,
        max_positive_items=max_positive_items,
    )

    content_candidates = content_tfidf.retrieve_from_profile(
        profile=profile,
        candidate_count=candidate_count,
        exclude_anime_ids=known_anime_ids,
    )

    popularity_candidates = popularity.retrieve(
        candidate_count=candidate_count,
        exclude_anime_ids=known_anime_ids,
    )

    content_ranks = content_candidates.loc[
        :,
        ["anime_id", "source_rank"],
    ].rename(columns={"source_rank": "content_rank"})

    popularity_ranks = popularity_candidates.loc[
        :,
        ["anime_id", "source_rank"],
    ].rename(columns={"source_rank": "popularity_rank"})

    ranking = content_ranks.merge(
        popularity_ranks,
        on="anime_id",
        how="outer",
    )

    ranking["cold_score"] = (1.0 / ranking["content_rank"]).fillna(0.0) + (
        1.0 / ranking["popularity_rank"]
    ).fillna(0.0)

    ranking = ranking.sort_values(
        ["cold_score", "anime_id"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    ranking["rank"] = np.arange(
        1,
        len(ranking) + 1,
        dtype=np.int32,
    )

    return ranking.loc[
        :,
        [
            "anime_id",
            "cold_score",
            "rank",
        ],
    ]


def build_cold_ranking(
    *,
    state: UserState,
    interactions: pd.DataFrame,
    known_anime_ids: set[int],
    candidate_count: int,
    relevance_threshold: float,
    max_positive_items: int,
    popularity: PopularityRetriever,
    content_tfidf: ContentTFIDFRetriever,
) -> pd.DataFrame:
    """
    Формирует ranking для cold-пользователя.

    Args:
        state: Состояние пользователя.
        interactions: Актуальная история пользователя.
        known_anime_ids: Известные пользователю anime.
        candidate_count: Максимальное число кандидатов от источника.
        relevance_threshold: Минимальная положительная оценка.
        max_positive_items: Максимальное число positive items в профиле.
        popularity: Обученный popularity retriever.
        content_tfidf: Обученный TF-IDF retriever.

    Returns:
        Кандидатов, отсортированных по cold score.

    Raises:
        ValueError: Если состояние не относится к cold inference.
    """
    if state in {
        UserState.EMPTY_HISTORY,
        UserState.NO_PREFERENCE_SIGNAL,
    }:
        candidates = popularity.retrieve(
            candidate_count=candidate_count,
            exclude_anime_ids=known_anime_ids,
        )

        return (
            candidates.loc[
                :,
                [
                    "anime_id",
                    "score",
                    "source_rank",
                ],
            ]
            .rename(
                columns={
                    "score": "cold_score",
                    "source_rank": "rank",
                }
            )
            .reset_index(drop=True)
        )

    if state in {
        UserState.SPARSE_COLD,
        UserState.PERSONALIZED_COLD,
    }:
        return build_new_user_ranking(
            interactions=interactions,
            known_anime_ids=known_anime_ids,
            candidate_count=candidate_count,
            relevance_threshold=relevance_threshold,
            max_positive_items=max_positive_items,
            popularity=popularity,
            content_tfidf=content_tfidf,
        )

    raise ValueError(f"Состояние {state} не относится к cold inference.")
