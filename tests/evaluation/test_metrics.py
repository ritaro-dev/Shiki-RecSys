import pytest

from shiki_recsys.evaluation.metrics import (
    RankingMetricResult,
    ndcg_at_k,
    recall_at_k,
)


def test_recall_at_k_calculates_mean_across_users() -> None:
    recommendations_by_user = {
        1: [10, 20, 30],
        2: [40, 50, 60],
    }

    positive_items_by_user = {
        1: {20, 30},
        2: {40, 70},
    }

    result = recall_at_k(
        recommendations_by_user,
        positive_items_by_user,
        k=3,
    )

    expected = RankingMetricResult(
        value=0.75,
        evaluated_users=2,
    )

    assert result == expected


def test_recall_at_k_uses_only_first_k_recommendations() -> None:
    recommendations_by_user = {
        1: [10, 20, 30, 40],
    }

    positive_items_by_user = {
        1: {30, 40},
    }

    result = recall_at_k(
        recommendations_by_user,
        positive_items_by_user,
        k=2,
    )

    assert result == RankingMetricResult(
        value=0.0,
        evaluated_users=1,
    )


def test_recall_at_k_assigns_zero_when_user_has_no_recommendations() -> None:
    recommendations_by_user = {}

    positive_items_by_user = {
        1: {10, 20},
    }

    result = recall_at_k(
        recommendations_by_user,
        positive_items_by_user,
        k=10,
    )

    assert result == RankingMetricResult(
        value=0.0,
        evaluated_users=1,
    )


def test_recall_at_k_skips_users_with_empty_positive_items() -> None:
    recommendations_by_user = {
        1: [10],
        2: [20],
    }

    positive_items_by_user = {
        1: {10},
        2: set(),
    }

    result = recall_at_k(
        recommendations_by_user,
        positive_items_by_user,
        k=1,
    )

    assert result == RankingMetricResult(
        value=1.0,
        evaluated_users=1,
    )


def test_recall_at_k_does_not_count_duplicate_recommendations_twice() -> None:
    recommendations_by_user = {
        1: [10, 10, 20],
    }

    positive_items_by_user = {
        1: {10, 20},
    }

    result = recall_at_k(
        recommendations_by_user,
        positive_items_by_user,
        k=2,
    )

    assert result == RankingMetricResult(
        value=0.5,
        evaluated_users=1,
    )


def test_recall_at_k_removes_duplicate_target_items() -> None:
    recommendations_by_user = {
        1: [10],
    }

    positive_items_by_user = {
        1: [10, 10],
    }

    result = recall_at_k(
        recommendations_by_user,
        positive_items_by_user,
        k=1,
    )

    assert result == RankingMetricResult(
        value=1.0,
        evaluated_users=1,
    )


@pytest.mark.parametrize(
    "k",
    [
        0,
        -1,
    ],
)
def test_recall_at_k_rejects_invalid_k(
    k: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="k должен быть больше 0",
    ):
        recall_at_k(
            recommendations_by_user={
                1: [10],
            },
            positive_items_by_user={
                1: {10},
            },
            k=k,
        )


def test_recall_at_k_rejects_missing_evaluation_users() -> None:
    with pytest.raises(
        ValueError,
        match="Нет пользователей с положительными",
    ):
        recall_at_k(
            recommendations_by_user={},
            positive_items_by_user={},
            k=10,
        )


def test_recall_at_k_rejects_only_empty_positive_sets() -> None:
    with pytest.raises(
        ValueError,
        match="Нет пользователей с положительными",
    ):
        recall_at_k(
            recommendations_by_user={
                1: [10],
            },
            positive_items_by_user={
                1: set(),
            },
            k=10,
        )


def test_ndcg_at_k_returns_one_for_ideal_ranking() -> None:
    recommendations_by_user = {
        1: [10, 20, 30],
    }

    positive_items_by_user = {
        1: {10, 20},
    }

    result = ndcg_at_k(
        recommendations_by_user,
        positive_items_by_user,
        k=3,
    )

    assert result == RankingMetricResult(
        value=1.0,
        evaluated_users=1,
    )


def test_ndcg_at_k_penalizes_relevant_item_at_lower_position() -> None:
    recommendations_by_user = {
        1: [10, 20, 30],
    }

    positive_items_by_user = {
        1: {30},
    }

    result = ndcg_at_k(
        recommendations_by_user,
        positive_items_by_user,
        k=3,
    )

    assert result.evaluated_users == 1
    assert result.value == pytest.approx(0.5)


def test_ndcg_at_k_calculates_mean_across_users() -> None:
    recommendations_by_user = {
        1: [10, 20, 30],
        2: [40, 50, 60],
    }

    positive_items_by_user = {
        1: {10},
        2: {60},
    }

    result = ndcg_at_k(
        recommendations_by_user,
        positive_items_by_user,
        k=3,
    )

    assert result.evaluated_users == 2
    assert result.value == pytest.approx(0.75)


def test_ndcg_at_k_uses_only_first_k_recommendations() -> None:
    recommendations_by_user = {
        1: [10, 20, 30],
    }

    positive_items_by_user = {
        1: {30},
    }

    result = ndcg_at_k(
        recommendations_by_user,
        positive_items_by_user,
        k=2,
    )

    assert result == RankingMetricResult(
        value=0.0,
        evaluated_users=1,
    )


def test_ndcg_at_k_assigns_zero_when_user_has_no_recommendations() -> None:
    result = ndcg_at_k(
        recommendations_by_user={},
        positive_items_by_user={
            1: {10},
        },
        k=5,
    )

    assert result == RankingMetricResult(
        value=0.0,
        evaluated_users=1,
    )


def test_ndcg_at_k_skips_users_with_empty_positive_items() -> None:
    recommendations_by_user = {
        1: [10],
        2: [20],
    }

    positive_items_by_user = {
        1: {10},
        2: set(),
    }

    result = ndcg_at_k(
        recommendations_by_user,
        positive_items_by_user,
        k=1,
    )

    assert result == RankingMetricResult(
        value=1.0,
        evaluated_users=1,
    )


def test_ndcg_at_k_does_not_count_duplicate_recommendations_twice() -> None:
    recommendations_by_user = {
        1: [10, 10, 20],
    }

    positive_items_by_user = {
        1: {10, 20},
    }

    result = ndcg_at_k(
        recommendations_by_user,
        positive_items_by_user,
        k=3,
    )

    expected_dcg = 1.0 + 1.0 / 2.0

    expected_idcg = 1.0 + 1.0 / 1.584962500721156

    assert result.evaluated_users == 1
    assert result.value == pytest.approx(expected_dcg / expected_idcg)


@pytest.mark.parametrize(
    "k",
    [
        0,
        -1,
    ],
)
def test_ndcg_at_k_rejects_invalid_k(
    k: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="k должен быть больше 0",
    ):
        ndcg_at_k(
            recommendations_by_user={
                1: [10],
            },
            positive_items_by_user={
                1: {10},
            },
            k=k,
        )


def test_ndcg_at_k_rejects_missing_evaluation_users() -> None:
    with pytest.raises(
        ValueError,
        match="Нет пользователей с положительными",
    ):
        ndcg_at_k(
            recommendations_by_user={},
            positive_items_by_user={},
            k=10,
        )


def test_ndcg_at_k_rejects_only_empty_positive_sets() -> None:
    with pytest.raises(
        ValueError,
        match="Нет пользователей с положительными",
    ):
        ndcg_at_k(
            recommendations_by_user={
                1: [10],
            },
            positive_items_by_user={
                1: set(),
            },
            k=10,
        )
