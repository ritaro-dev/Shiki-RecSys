from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from shiki_recsys.evaluation.retriever_analysis import RetrieverContribution
from shiki_recsys.evaluation.retrievers import (
    RetrieverCandidateRun,
    RetrieverEvaluationResult,
    evaluate_retriever,
    evaluate_retriever_run,
    generate_retriever_evaluation_run,
)
from shiki_recsys.retrievers.common import RetrieverName


def test_evaluate_retriever_calculates_recall_and_coverage() -> None:
    history_interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "anime_id": [10, 20, 30],
        }
    )

    target_interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "anime_id": [40, 50, 60],
            "rating": [9, 10, 8],
        }
    )

    calls: list[tuple[int, int, set[int]]] = []

    def retrieve_candidates(
        user_id: int,
        candidate_count: int,
        exclude_anime_ids: set[int],
    ) -> pd.DataFrame:
        calls.append(
            (
                user_id,
                candidate_count,
                exclude_anime_ids,
            )
        )

        candidates_by_user = {
            1: [40, 70],
            2: [80, 60],
        }

        return pd.DataFrame(
            {
                "anime_id": candidates_by_user[user_id],
            }
        )

    result = evaluate_retriever(
        retrieve_candidates=retrieve_candidates,
        history_interactions=history_interactions,
        target_interactions=target_interactions,
        positive_rating_threshold=8,
        candidate_count=2,
        catalog_size=10,
    )

    assert result == RetrieverEvaluationResult(
        candidate_count=2,
        recall_at_k=pytest.approx(0.75),
        catalog_coverage_at_k=pytest.approx(0.4),
        evaluated_users=2,
    )

    assert calls == [
        (1, 2, {10, 20}),
        (2, 2, {30}),
    ]


def test_evaluate_retriever_uses_only_users_with_positive_targets() -> None:
    history_interactions = pd.DataFrame(
        {
            "user_id": [1, 2],
            "anime_id": [10, 20],
        }
    )

    target_interactions = pd.DataFrame(
        {
            "user_id": [1, 2],
            "anime_id": [30, 40],
            "rating": [9, 6],
        }
    )

    evaluated_user_ids: list[int] = []

    def retrieve_candidates(
        user_id: int,
        candidate_count: int,
        exclude_anime_ids: set[int],
    ) -> pd.DataFrame:
        evaluated_user_ids.append(user_id)

        return pd.DataFrame(
            {
                "anime_id": [30],
            }
        )

    result = evaluate_retriever(
        retrieve_candidates=retrieve_candidates,
        history_interactions=history_interactions,
        target_interactions=target_interactions,
        positive_rating_threshold=8,
        candidate_count=1,
        catalog_size=10,
    )

    assert evaluated_user_ids == [1]
    assert result.evaluated_users == 1
    assert result.recall_at_k == 1.0


def test_generate_retriever_evaluation_run_uses_production_generation() -> None:
    history_interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "anime_id": [10, 20, 30],
        }
    )

    target_interactions = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "anime_id": [40, 50, 60],
            "rating": [9, 6, 10],
        }
    )

    popularity = MagicMock()
    explicit_svd = MagicMock()
    implicit_als = MagicMock()
    content_tfidf = MagicMock()

    user_1_candidates = {
        RetrieverName.POPULARITY: pd.DataFrame({"anime_id": [40]}),
    }
    user_3_candidates = {
        RetrieverName.POPULARITY: pd.DataFrame({"anime_id": [60]}),
    }

    with patch(
        "shiki_recsys.evaluation.retrievers.generate_retriever_candidates",
        side_effect=[
            user_1_candidates,
            user_3_candidates,
        ],
    ) as generate_candidates:
        result = generate_retriever_evaluation_run(
            history_interactions=history_interactions,
            target_interactions=target_interactions,
            positive_rating_threshold=8,
            candidate_count=100,
            popularity=popularity,
            explicit_svd=explicit_svd,
            implicit_als=implicit_als,
            content_tfidf=content_tfidf,
        )

    assert result.positive_items_by_user == {
        1: {40},
        3: {60},
    }
    assert result.candidates_by_user == {
        1: user_1_candidates,
        3: user_3_candidates,
    }

    assert generate_candidates.call_args_list == [
        call(
            user_id=1,
            known_anime_ids={10, 20},
            candidate_count=100,
            popularity=popularity,
            explicit_svd=explicit_svd,
            implicit_als=implicit_als,
            content_tfidf=content_tfidf,
        ),
        call(
            user_id=3,
            known_anime_ids=set(),
            candidate_count=100,
            popularity=popularity,
            explicit_svd=explicit_svd,
            implicit_als=implicit_als,
            content_tfidf=content_tfidf,
        ),
    ]


def test_evaluate_retriever_run_summarizes_generated_candidates() -> None:
    run = RetrieverCandidateRun(
        positive_items_by_user={
            1: {20, 50},
            2: {30},
        },
        candidates_by_user={
            1: {
                RetrieverName.EXPLICIT_SVD: pd.DataFrame({"anime_id": [10, 20]}),
                RetrieverName.IMPLICIT_ALS: pd.DataFrame({"anime_id": [20, 50]}),
            },
            2: {
                RetrieverName.EXPLICIT_SVD: pd.DataFrame({"anime_id": [30, 40]}),
                RetrieverName.IMPLICIT_ALS: pd.DataFrame({"anime_id": [30, 60]}),
            },
        },
    )

    result = evaluate_retriever_run(
        run,
        candidate_count=2,
        catalog_size=10,
    )

    metrics = result.individual_metrics.set_index("source")

    assert metrics.loc[
        RetrieverName.EXPLICIT_SVD.value,
        "recall_at_k",
    ] == pytest.approx(0.75)

    assert (
        metrics.loc[
            RetrieverName.IMPLICIT_ALS.value,
            "recall_at_k",
        ]
        == 1.0
    )

    assert metrics.loc[
        RetrieverName.EXPLICIT_SVD.value,
        "catalog_coverage_at_k",
    ] == pytest.approx(0.4)

    assert metrics.loc[
        RetrieverName.IMPLICIT_ALS.value,
        "catalog_coverage_at_k",
    ] == pytest.approx(0.4)

    assert result.union_recall == 1.0

    assert result.contributions == [
        RetrieverContribution(
            source=RetrieverName.EXPLICIT_SVD,
            full_recall=1.0,
            recall_without_source=1.0,
            marginal_recall=0.0,
            unique_relevant_hits=0,
        ),
        RetrieverContribution(
            source=RetrieverName.IMPLICIT_ALS,
            full_recall=1.0,
            recall_without_source=0.75,
            marginal_recall=0.25,
            unique_relevant_hits=1,
        ),
    ]
