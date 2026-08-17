from dataclasses import dataclass

import pandas as pd

from shiki_recsys.config.training import TrainingConfig
from shiki_recsys.evaluation.ground_truth import (
    build_positive_items_by_user,
)
from shiki_recsys.evaluation.metrics import (
    ndcg_at_k,
    recall_at_k,
)
from shiki_recsys.inference.model_bundle import ModelBundle
from shiki_recsys.inference.warm_pipeline import build_warm_ranking
from shiki_recsys.preprocessing.known_items import build_known_items
from shiki_recsys.preprocessing.splitting import InteractionSplit


@dataclass(frozen=True)
class ModelEvaluationResult:
    """Store offline ranking evaluation results."""

    ranking_k: int
    recall_at_k: float
    ndcg_at_k: float
    evaluated_users: int


def evaluate_model_bundle(
    *,
    bundle: ModelBundle,
    split: InteractionSplit,
    config: TrainingConfig,
) -> ModelEvaluationResult:
    """
    Evaluate a warm model bundle on the held-out test interactions.

    Args:
        bundle: Evaluation model bundle.
        split: Chronological interaction split.
        config: Training configuration.

    Returns:
        Ranking metrics calculated on supported warm users.
    """
    history_interactions = pd.concat(
        [
            split.train,
            split.validation,
        ],
        ignore_index=True,
    )

    known_items_by_user = build_known_items(
        history_interactions,
    )
    positive_items_by_user = build_positive_items_by_user(
        split.test,
        positive_rating_threshold=(config.target.positive_rating_threshold),
    )

    warm_positive_items_by_user = {
        user_id: positive_items
        for user_id, positive_items in positive_items_by_user.items()
        if bundle.supports_personal_user(user_id)
    }

    recommendations_by_user: dict[int, list[int]] = {}

    for user_id in warm_positive_items_by_user:
        ranking = build_warm_ranking(
            user_id=user_id,
            known_anime_ids=known_items_by_user.get(
                user_id,
                set(),
            ),
            candidate_count=(config.candidate_generation.retrieval_k),
            popularity=bundle.popularity,
            explicit_svd=bundle.explicit_svd,
            implicit_als=bundle.implicit_als,
            content_tfidf=bundle.content_tfidf,
            ranker=bundle.ranker,
        )

        recommendations_by_user[user_id] = ranking["anime_id"].tolist()

    ranking_k = config.evaluation.ranking_k

    recall = recall_at_k(
        recommendations_by_user,
        warm_positive_items_by_user,
        k=ranking_k,
    )
    ndcg = ndcg_at_k(
        recommendations_by_user,
        warm_positive_items_by_user,
        k=ranking_k,
    )

    return ModelEvaluationResult(
        ranking_k=ranking_k,
        recall_at_k=recall.value,
        ndcg_at_k=ndcg.value,
        evaluated_users=recall.evaluated_users,
    )
