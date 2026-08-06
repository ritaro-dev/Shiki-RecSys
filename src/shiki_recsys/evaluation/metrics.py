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
