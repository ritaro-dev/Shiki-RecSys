import pandas as pd
import pytest

from shiki_recsys.evaluation.retriever_analysis import (
    RetrieverContribution,
    analyze_retriever_contributions,
    candidate_jaccard_matrix,
    candidate_union_recall,
)
from shiki_recsys.retrievers.common import RetrieverName


def _candidates(anime_ids: list[int]) -> pd.DataFrame:
    """Build candidate data for evaluation tests."""
    return pd.DataFrame(
        {
            "anime_id": anime_ids,
        }
    )


def test_candidate_union_recall_combines_retriever_hits() -> None:
    candidates_by_user = {
        1: {
            RetrieverName.EXPLICIT_SVD: _candidates([10, 20]),
            RetrieverName.IMPLICIT_ALS: _candidates([50, 60]),
        },
        2: {
            RetrieverName.EXPLICIT_SVD: _candidates([30, 40]),
            RetrieverName.IMPLICIT_ALS: _candidates([70, 80]),
        },
    }

    positive_items_by_user = {
        1: {20, 50},
        2: {30, 90},
    }

    result = candidate_union_recall(
        candidates_by_user,
        positive_items_by_user,
        candidate_count=2,
    )

    assert result == pytest.approx(0.75)


def test_analyze_retriever_contributions_finds_unique_hits() -> None:
    candidates_by_user = {
        1: {
            RetrieverName.EXPLICIT_SVD: _candidates([10, 20]),
            RetrieverName.IMPLICIT_ALS: _candidates([20, 50]),
        },
        2: {
            RetrieverName.EXPLICIT_SVD: _candidates([30, 40]),
            RetrieverName.IMPLICIT_ALS: _candidates([30, 60]),
        },
    }

    positive_items_by_user = {
        1: {20, 50},
        2: {30},
    }

    results = analyze_retriever_contributions(
        candidates_by_user,
        positive_items_by_user,
        candidate_count=2,
    )

    assert results == [
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


def test_candidate_jaccard_matrix_uses_user_item_pairs() -> None:
    candidates_by_user = {
        1: {
            RetrieverName.EXPLICIT_SVD: _candidates([10, 20]),
            RetrieverName.IMPLICIT_ALS: _candidates([10, 40]),
        },
        2: {
            RetrieverName.EXPLICIT_SVD: _candidates([30]),
            RetrieverName.IMPLICIT_ALS: _candidates([20]),
        },
    }

    matrix = candidate_jaccard_matrix(
        candidates_by_user,
        candidate_count=2,
    )

    svd = RetrieverName.EXPLICIT_SVD.value
    als = RetrieverName.IMPLICIT_ALS.value

    assert matrix.loc[svd, svd] == 1.0
    assert matrix.loc[als, als] == 1.0
    assert matrix.loc[svd, als] == pytest.approx(1 / 5)
    assert matrix.loc[als, svd] == pytest.approx(1 / 5)
