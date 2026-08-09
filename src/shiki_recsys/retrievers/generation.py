import numpy as np
import pandas as pd

from shiki_recsys.retrievers.common import RetrieverName
from shiki_recsys.retrievers.content_tfidf import ContentTFIDFRetriever
from shiki_recsys.retrievers.explicit_svd import ExplicitSVDRetriever
from shiki_recsys.retrievers.implicit_als import ImplicitALSRetriever
from shiki_recsys.retrievers.popularity import PopularityRetriever


def generate_retriever_candidates(
    *,
    user_id: int,
    known_anime_ids: set[int],
    candidate_count: int,
    popularity: PopularityRetriever,
    explicit_svd: ExplicitSVDRetriever,
    implicit_als: ImplicitALSRetriever,
    content_tfidf: ContentTFIDFRetriever,
) -> dict[RetrieverName, pd.DataFrame]:
    """
    Формирует candidate set отдельных retriever-ов.

    Args:
        user_id: Идентификатор пользователя.
        known_anime_ids: Известные пользователю аниме.
        candidate_count: Максимальное число кандидатов от retriever-а.
        popularity: Popularity retriever.
        explicit_svd: Explicit SVD retriever.
        implicit_als: Implicit ALS retriever.
        content_tfidf: TF-IDF content retriever.

    Returns:
        Кандидатов, сгруппированных по retriever-ам.
    """
    return {
        RetrieverName.POPULARITY: popularity.retrieve(
            candidate_count=candidate_count,
            exclude_anime_ids=known_anime_ids,
        ),
        RetrieverName.EXPLICIT_SVD: explicit_svd.retrieve(
            user_id=user_id,
            candidate_count=candidate_count,
            exclude_anime_ids=known_anime_ids,
        ),
        RetrieverName.IMPLICIT_ALS: implicit_als.retrieve(
            user_id=user_id,
            candidate_count=candidate_count,
            exclude_anime_ids=known_anime_ids,
        ),
        RetrieverName.CONTENT_TFIDF: content_tfidf.retrieve(
            user_id=user_id,
            candidate_count=candidate_count,
            exclude_anime_ids=known_anime_ids,
        ),
    }


def build_candidate_union(
    retriever_candidates: dict[RetrieverName, pd.DataFrame],
) -> np.ndarray:
    """
    Объединяет выдачи retriever-ов в общий candidate set.

    Args:
        retriever_candidates: Кандидаты отдельных retriever-ов.

    Returns:
        Уникальные anime_id общего candidate set.
    """
    anime_id_parts = [
        retriever_candidates[source]["anime_id"]
        for source in RetrieverName
        if source in retriever_candidates
    ]

    if not anime_id_parts:
        return np.array(
            [],
            dtype=np.int64,
        )

    return (
        pd.concat(
            anime_id_parts,
            ignore_index=True,
        )
        .drop_duplicates()
        .to_numpy(dtype=np.int64)
    )


def score_candidate_union(
    *,
    user_id: int,
    anime_ids: np.ndarray,
    retriever_candidates: dict[RetrieverName, pd.DataFrame],
    popularity: PopularityRetriever,
    explicit_svd: ExplicitSVDRetriever,
    implicit_als: ImplicitALSRetriever,
    content_tfidf: ContentTFIDFRetriever,
) -> dict[RetrieverName, np.ndarray]:
    """
    Рассчитывает scores retriever-ов для общего candidate set.

    Args:
        user_id: Идентификатор пользователя.
        anime_ids: Anime общего candidate set.
        retriever_candidates: Top-K выдачи retriever-ов.
        popularity: Popularity retriever.
        explicit_svd: Explicit SVD retriever.
        implicit_als: Implicit ALS retriever.
        content_tfidf: TF-IDF content retriever.

    Returns:
        Scores retriever-ов в порядке переданных anime_id.
    """
    retriever_scores = {}

    for source in RetrieverName:
        candidates = retriever_candidates[source]
        known_scores = candidates.set_index("anime_id")["score"]

        scores = (
            pd.Series(anime_ids)
            .map(known_scores)
            .to_numpy(
                dtype=np.float64,
                copy=True,
            )
        )

        missing_mask = np.isnan(scores)

        if missing_mask.any():
            missing_anime_ids = anime_ids[missing_mask]

            if source == RetrieverName.POPULARITY:
                missing_scores = popularity.score_items(
                    anime_ids=missing_anime_ids,
                )
            elif source == RetrieverName.EXPLICIT_SVD:
                missing_scores = explicit_svd.score_items(
                    user_id=user_id,
                    anime_ids=missing_anime_ids,
                )
            elif source == RetrieverName.IMPLICIT_ALS:
                missing_scores = implicit_als.score_items(
                    user_id=user_id,
                    anime_ids=missing_anime_ids,
                )
            else:
                missing_scores = content_tfidf.score_items(
                    user_id=user_id,
                    anime_ids=missing_anime_ids,
                )

            scores[missing_mask] = missing_scores

        retriever_scores[source] = scores

    return retriever_scores
