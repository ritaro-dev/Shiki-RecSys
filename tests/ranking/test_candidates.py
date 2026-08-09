import numpy as np
import pandas as pd

from shiki_recsys.ranking.candidates import build_ranker_candidate_features
from shiki_recsys.retrievers.common import RetrieverName, empty_candidates


def test_build_ranker_candidate_features_builds_retriever_features() -> None:
    """Проверяет построение признаков retriever-ов для candidate set."""

    anime_ids = np.array(
        [10, 20, 30],
        dtype=np.int64,
    )

    popularity = pd.DataFrame(
        {
            "anime_id": pd.Series([10, 20], dtype="int64"),
            "score": pd.Series([5.0, 4.0], dtype="float64"),
            "source": pd.Series(
                [RetrieverName.POPULARITY.value] * 2,
                dtype="string",
            ),
            "source_rank": pd.Series([1, 2], dtype="int32"),
        }
    )

    explicit_svd = pd.DataFrame(
        {
            "anime_id": pd.Series([20, 30], dtype="int64"),
            "score": pd.Series([9.0, 8.0], dtype="float64"),
            "source": pd.Series(
                [RetrieverName.EXPLICIT_SVD.value] * 2,
                dtype="string",
            ),
            "source_rank": pd.Series([1, 2], dtype="int32"),
        }
    )

    retriever_candidates = {
        RetrieverName.POPULARITY: popularity,
        RetrieverName.EXPLICIT_SVD: explicit_svd,
        RetrieverName.IMPLICIT_ALS: empty_candidates(),
        RetrieverName.CONTENT_TFIDF: empty_candidates(),
    }

    retriever_scores = {
        RetrieverName.POPULARITY: np.array(
            [5.0, 4.0, 3.0],
            dtype=np.float64,
        ),
        RetrieverName.EXPLICIT_SVD: np.array(
            [7.0, 9.0, 8.0],
            dtype=np.float64,
        ),
        RetrieverName.IMPLICIT_ALS: np.full(
            3,
            np.nan,
            dtype=np.float64,
        ),
        RetrieverName.CONTENT_TFIDF: np.full(
            3,
            np.nan,
            dtype=np.float64,
        ),
    }

    features = build_ranker_candidate_features(
        user_id=7,
        anime_ids=anime_ids,
        retriever_candidates=retriever_candidates,
        retriever_scores=retriever_scores,
    )

    assert features["anime_id"].tolist() == [20, 10, 30]
    assert features["user_id"].tolist() == [7, 7, 7]

    by_anime = features.set_index("anime_id")

    assert by_anime.loc[10, "from_popularity"] == 1
    assert by_anime.loc[10, "from_explicit_svd"] == 0

    assert by_anime.loc[10, "score_popularity"] == 5.0
    assert by_anime.loc[10, "score_explicit_svd"] == 7.0

    assert by_anime.loc[10, "rank_popularity"] == 1
    assert pd.isna(by_anime.loc[10, "rank_explicit_svd"])

    assert by_anime.loc[10, "rr_popularity"] == 1.0
    assert by_anime.loc[10, "rr_explicit_svd"] == 0.0

    assert by_anime.loc[20, "retriever_count"] == 2
    assert by_anime.loc[20, "best_rank"] == 1
    assert by_anime.loc[20, "mean_rank_present"] == 1.5
    assert by_anime.loc[20, "rr_sum"] == 1.5
    assert by_anime.loc[20, "rr_max"] == 1.0

    assert by_anime.loc[30, "retriever_count"] == 1
    assert by_anime.loc[30, "best_rank"] == 2

    assert by_anime.loc[30, "score_popularity"] == 3.0
    assert by_anime.loc[30, "from_popularity"] == 0
    assert pd.isna(by_anime.loc[30, "rank_popularity"])


def test_build_ranker_candidate_features_returns_stable_empty_frame() -> None:
    """Проверяет схему признаков при пустом candidate set."""

    retriever_candidates = {source: empty_candidates() for source in RetrieverName}
    retriever_scores = {
        source: np.array([], dtype=np.float64) for source in RetrieverName
    }

    features = build_ranker_candidate_features(
        user_id=7,
        anime_ids=np.array([], dtype=np.int64),
        retriever_candidates=retriever_candidates,
        retriever_scores=retriever_scores,
    )

    expected_columns = [
        "user_id",
        "anime_id",
        "score_popularity",
        "rank_popularity",
        "from_popularity",
        "rr_popularity",
        "score_explicit_svd",
        "rank_explicit_svd",
        "from_explicit_svd",
        "rr_explicit_svd",
        "score_implicit_als",
        "rank_implicit_als",
        "from_implicit_als",
        "rr_implicit_als",
        "score_content_tfidf",
        "rank_content_tfidf",
        "from_content_tfidf",
        "rr_content_tfidf",
        "retriever_count",
        "best_rank",
        "mean_rank_present",
        "rr_sum",
        "rr_max",
    ]

    assert features.empty
    assert features.columns.tolist() == expected_columns


def test_build_ranker_candidate_features_uses_rr_sum_as_tiebreak() -> None:
    """Проверяет rr_sum как tie-break при одинаковом best_rank."""

    anime_ids = np.array(
        [10, 20],
        dtype=np.int64,
    )

    popularity = pd.DataFrame(
        {
            "anime_id": pd.Series([10, 20], dtype="int64"),
            "score": pd.Series([5.0, 4.0], dtype="float64"),
            "source": pd.Series(
                [RetrieverName.POPULARITY.value] * 2,
                dtype="string",
            ),
            "source_rank": pd.Series([1, 2], dtype="int32"),
        }
    )

    explicit_svd = pd.DataFrame(
        {
            "anime_id": pd.Series([20, 10], dtype="int64"),
            "score": pd.Series([9.0, 8.0], dtype="float64"),
            "source": pd.Series(
                [RetrieverName.EXPLICIT_SVD.value] * 2,
                dtype="string",
            ),
            "source_rank": pd.Series([1, 3], dtype="int32"),
        }
    )

    retriever_candidates = {
        RetrieverName.POPULARITY: popularity,
        RetrieverName.EXPLICIT_SVD: explicit_svd,
        RetrieverName.IMPLICIT_ALS: empty_candidates(),
        RetrieverName.CONTENT_TFIDF: empty_candidates(),
    }

    retriever_scores = {
        source: np.zeros(
            len(anime_ids),
            dtype=np.float64,
        )
        for source in RetrieverName
    }

    features = build_ranker_candidate_features(
        user_id=7,
        anime_ids=anime_ids,
        retriever_candidates=retriever_candidates,
        retriever_scores=retriever_scores,
    )

    assert features["anime_id"].tolist() == [20, 10]
