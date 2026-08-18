from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import shiki_recsys.evaluation.model as model_module
from shiki_recsys.evaluation.model import evaluate_model_bundle
from shiki_recsys.preprocessing.splitting import InteractionSplit


def test_evaluate_model_bundle_uses_warm_test_users(
    monkeypatch,
) -> None:
    """Evaluate warm users against held-out positive test items."""
    train = pd.DataFrame(
        {
            "user_id": [1, 2],
            "anime_id": [10, 20],
            "rating": [8, 8],
        }
    )
    validation = pd.DataFrame(
        {
            "user_id": [1, 2],
            "anime_id": [11, 21],
            "rating": [9, 9],
        }
    )
    test = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3],
            "anime_id": [30, 31, 40, 50],
            "rating": [10, 5, 10, 10],
        }
    )

    split = InteractionSplit(
        train=train,
        validation=validation,
        test=test,
        evaluation_user_ids=frozenset({1, 2, 3}),
        train_only_user_ids=frozenset(),
    )

    bundle = MagicMock()
    bundle.supports_personal_user.side_effect = lambda user_id: user_id in {1, 2}

    rankings = {
        1: pd.DataFrame(
            {
                "anime_id": [30, 99],
            }
        ),
        2: pd.DataFrame(
            {
                "anime_id": [99, 40],
            }
        ),
    }

    build_warm_ranking = MagicMock(
        side_effect=lambda **kwargs: rankings[kwargs["user_id"]]
    )
    monkeypatch.setattr(
        model_module,
        "build_warm_ranking",
        build_warm_ranking,
    )

    config = SimpleNamespace(
        target=SimpleNamespace(
            positive_rating_threshold=8,
        ),
        candidate_generation=SimpleNamespace(
            retrieval_k=100,
        ),
        evaluation=SimpleNamespace(
            ranking_k=2,
        ),
    )

    result = evaluate_model_bundle(
        bundle=bundle,
        split=split,
        config=config,
    )

    assert result.ranking_k == 2
    assert result.recall_at_k == pytest.approx(1.0)
    assert result.ndcg_at_k == pytest.approx((1.0 + 1.0 / 1.584962500721156) / 2)
    assert result.evaluated_users == 2

    assert build_warm_ranking.call_count == 2

    first_call = build_warm_ranking.call_args_list[0]
    assert first_call.kwargs["user_id"] == 1
    assert first_call.kwargs["known_anime_ids"] == {10, 11}
    assert first_call.kwargs["candidate_count"] == 100

    second_call = build_warm_ranking.call_args_list[1]
    assert second_call.kwargs["user_id"] == 2
    assert second_call.kwargs["known_anime_ids"] == {20, 21}
