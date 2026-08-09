import pandas as pd

from shiki_recsys.ranking.targets import attach_ranker_target


def test_attach_ranker_target_marks_positive_candidates() -> None:
    """Проверяет добавление бинарного target к кандидатам."""

    candidate_features = pd.DataFrame(
        {
            "user_id": pd.Series([1, 1, 2, 3], dtype="int64"),
            "anime_id": pd.Series([10, 20, 30, 40], dtype="int64"),
            "score_popularity": pd.Series(
                [5.0, 4.0, 3.0, 2.0],
                dtype="float64",
            ),
        }
    )

    result = attach_ranker_target(
        candidate_features,
        positive_items_by_user={
            1: {20},
            2: {30, 50},
        },
    )

    assert result["target"].tolist() == [0, 1, 1, 0]
    assert result["target"].dtype == "int8"

    assert "target" not in candidate_features.columns
    assert result["score_popularity"].tolist() == [5.0, 4.0, 3.0, 2.0]


def test_attach_ranker_target_handles_empty_candidates() -> None:
    """Проверяет добавление target к пустой таблице кандидатов."""

    candidate_features = pd.DataFrame(
        {
            "user_id": pd.Series(dtype="int64"),
            "anime_id": pd.Series(dtype="int64"),
        }
    )

    result = attach_ranker_target(
        candidate_features,
        positive_items_by_user={1: {10}},
    )

    assert result.empty
    assert result.columns.tolist() == [
        "user_id",
        "anime_id",
        "target",
    ]
    assert result["target"].dtype == "int8"
