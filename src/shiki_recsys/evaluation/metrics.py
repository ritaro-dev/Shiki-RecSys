import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RankingMetricResult:
    """Хранит результат ranking-метрики."""

    value: float
    evaluated_users: int


def recall_at_k(
    recommendations_by_user: Mapping[
        int,
        Sequence[int],
    ],
    positive_items_by_user: Mapping[
        int,
        Collection[int],
    ],
    *,
    k: int,
) -> RankingMetricResult:
    """
    Рассчитывает средний Recall@K по пользователям.

    Args:
        recommendations_by_user: Ранжированные рекомендации
            пользователей.
        positive_items_by_user: Положительные целевые объекты
            пользователей.
        k: Размер оцениваемой части списка рекомендаций.

    Returns:
        Значение Recall@K и количество оценённых пользователей.

    Raises:
        ValueError: Если k некорректен или отсутствуют пользователи
            с положительными целевыми объектами.
    """

    if k <= 0:
        raise ValueError("k должен быть больше 0.")

    user_recalls: list[float] = []

    for (
        user_id,
        positive_items,
    ) in positive_items_by_user.items():
        target_items = set(positive_items)

        if not target_items:
            continue

        recommendations = recommendations_by_user.get(
            user_id,
            (),
        )

        top_k_items = recommendations[:k]

        relevant_recommendations = set(top_k_items).intersection(target_items)

        user_recall = len(relevant_recommendations) / len(target_items)

        user_recalls.append(user_recall)

    if not user_recalls:
        raise ValueError("Нет пользователей с положительными целевыми объектами.")

    return RankingMetricResult(
        value=sum(user_recalls) / len(user_recalls),
        evaluated_users=len(user_recalls),
    )


def ndcg_at_k(
    recommendations_by_user: Mapping[
        int,
        Sequence[int],
    ],
    positive_items_by_user: Mapping[
        int,
        Collection[int],
    ],
    *,
    k: int,
) -> RankingMetricResult:
    """
    Рассчитывает средний NDCG@K по пользователям.

    Args:
        recommendations_by_user: Ранжированные рекомендации
            пользователей.
        positive_items_by_user: Положительные целевые объекты
            пользователей.
        k: Размер оцениваемой части списка рекомендаций.

    Returns:
        Значение NDCG@K и количество оценённых пользователей.

    Raises:
        ValueError: Если k некорректен или отсутствуют пользователи
            с положительными целевыми объектами.
    """

    if k <= 0:
        raise ValueError("k должен быть больше 0.")

    user_ndcg_values: list[float] = []

    for (
        user_id,
        positive_items,
    ) in positive_items_by_user.items():
        target_items = set(positive_items)

        if not target_items:
            continue

        recommendations = recommendations_by_user.get(
            user_id,
            (),
        )

        top_k_items = recommendations[:k]

        dcg = 0.0
        seen_items: set[int] = set()

        for rank, anime_id in enumerate(
            top_k_items,
            start=1,
        ):
            if anime_id in seen_items:
                continue

            seen_items.add(anime_id)

            if anime_id in target_items:
                dcg += 1.0 / math.log2(rank + 1)

        ideal_relevant_count = min(
            len(target_items),
            k,
        )

        idcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(
                1,
                ideal_relevant_count + 1,
            )
        )

        user_ndcg_values.append(dcg / idcg)

    if not user_ndcg_values:
        raise ValueError("Нет пользователей с положительными целевыми объектами.")

    return RankingMetricResult(
        value=(sum(user_ndcg_values) / len(user_ndcg_values)),
        evaluated_users=len(user_ndcg_values),
    )


def precision_at_k(
    recommendations_by_user: Mapping[
        int,
        Sequence[int],
    ],
    positive_items_by_user: Mapping[
        int,
        Collection[int],
    ],
    *,
    k: int,
) -> RankingMetricResult:
    """
    Calculate mean Precision@K across users.

    Args:
        recommendations_by_user: Ranked recommendations for each user.
        positive_items_by_user: Positive target items for each user.
        k: Number of top recommendations to evaluate.

    Returns:
        Precision@K and the number of evaluated users.

    Raises:
        ValueError: If k is invalid or no users have positive target items.
    """
    if k <= 0:
        raise ValueError("k должен быть больше 0.")

    user_precisions: list[float] = []

    for user_id, positive_items in positive_items_by_user.items():
        target_items = set(positive_items)

        if not target_items:
            continue

        recommendations = recommendations_by_user.get(
            user_id,
            (),
        )

        top_k_items = recommendations[:k]

        relevant_recommendations = set(top_k_items).intersection(target_items)

        user_precisions.append(len(relevant_recommendations) / k)

    if not user_precisions:
        raise ValueError("Нет пользователей с положительными целевыми объектами.")

    return RankingMetricResult(
        value=sum(user_precisions) / len(user_precisions),
        evaluated_users=len(user_precisions),
    )


def hit_rate_at_k(
    recommendations_by_user: Mapping[
        int,
        Sequence[int],
    ],
    positive_items_by_user: Mapping[
        int,
        Collection[int],
    ],
    *,
    k: int,
) -> RankingMetricResult:
    """
    Calculate HitRate@K across users.

    Args:
        recommendations_by_user: Ranked recommendations for each user.
        positive_items_by_user: Positive target items for each user.
        k: Number of top recommendations to evaluate.

    Returns:
        HitRate@K and the number of evaluated users.

    Raises:
        ValueError: If k is invalid or no users have positive target items.
    """
    if k <= 0:
        raise ValueError("k должен быть больше 0.")

    user_hits: list[float] = []

    for user_id, positive_items in positive_items_by_user.items():
        target_items = set(positive_items)

        if not target_items:
            continue

        recommendations = recommendations_by_user.get(
            user_id,
            (),
        )

        top_k_items = recommendations[:k]

        user_hits.append(float(bool(set(top_k_items).intersection(target_items))))

    if not user_hits:
        raise ValueError("Нет пользователей с положительными целевыми объектами.")

    return RankingMetricResult(
        value=sum(user_hits) / len(user_hits),
        evaluated_users=len(user_hits),
    )


def catalog_coverage_at_k(
    recommendations_by_user: Mapping[
        int,
        Sequence[int],
    ],
    *,
    catalog_size: int,
    k: int,
) -> RankingMetricResult:
    """
    Calculate catalog coverage of top-K recommendations.

    Args:
        recommendations_by_user: Ranked recommendations for each user.
        catalog_size: Number of recommendable items in the catalog.
        k: Number of top recommendations to evaluate.

    Returns:
        Catalog coverage and the number of evaluated users.

    Raises:
        ValueError: If k or catalog_size is invalid or no users are provided.
    """
    if k <= 0:
        raise ValueError("k должен быть больше 0.")

    if catalog_size <= 0:
        raise ValueError("catalog_size должен быть больше 0.")

    if not recommendations_by_user:
        raise ValueError("Нет пользователей для расчёта coverage.")

    recommended_items = {
        anime_id
        for recommendations in recommendations_by_user.values()
        for anime_id in recommendations[:k]
    }

    return RankingMetricResult(
        value=len(recommended_items) / catalog_size,
        evaluated_users=len(recommendations_by_user),
    )
