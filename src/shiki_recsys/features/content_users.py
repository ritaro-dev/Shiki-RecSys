from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from shiki_recsys.features.content_items import ContentItemFeatures


@dataclass(frozen=True)
class ContentUserProfiles:
    """Хранит content-профили пользователей."""

    user_profile_matrix: csr_matrix
    raw_user_ids: np.ndarray
    user_to_inner: dict[int, int]


def build_content_user_profiles(
    train_interactions: pd.DataFrame,
    item_features: ContentItemFeatures,
    *,
    relevance_threshold: float,
    max_positive_items: int,
) -> ContentUserProfiles:
    """
    Формирует content-профили пользователей по недавним положительным объектам.

    Args:
        train_interactions: Подготовленные взаимодействия из train-части.
        item_features: Content-представление каталога.
        relevance_threshold: Минимальная положительная оценка.
        max_positive_items: Максимальное число недавних положительных объектов.

    Returns:
        L2-нормализованные профили пользователей и соответствие
        user_id внутренним индексам.

    Raises:
        ValueError: Если входные данные или параметры некорректны
            либо положительный anime_id отсутствует в item mapping.
    """
    required_columns = {
        "user_id",
        "anime_id",
        "rating",
        "updated_at",
    }
    missing_columns = required_columns.difference(train_interactions.columns)

    if missing_columns:
        raise ValueError(
            f"В train_interactions отсутствуют столбцы: {sorted(missing_columns)}."
        )

    if train_interactions.empty:
        raise ValueError("train_interactions не должен быть пустым.")

    if not np.isfinite(relevance_threshold) or not 0 < relevance_threshold <= 10:
        raise ValueError("relevance_threshold должен быть конечным числом от 0 до 10.")

    if max_positive_items <= 0:
        raise ValueError("max_positive_items должен быть больше 0.")

    positive_interactions = (
        train_interactions.loc[
            train_interactions["rating"] >= relevance_threshold,
            [
                "user_id",
                "anime_id",
                "updated_at",
            ],
        ]
        .sort_values(
            [
                "user_id",
                "updated_at",
                "anime_id",
            ],
            kind="stable",
        )
        .groupby(
            "user_id",
            sort=False,
            group_keys=False,
        )
        .tail(max_positive_items)
        .reset_index(drop=True)
    )

    if positive_interactions.empty:
        return ContentUserProfiles(
            user_profile_matrix=csr_matrix(
                (0, item_features.item_feature_matrix.shape[1]),
                dtype=np.float32,
            ),
            raw_user_ids=np.array([], dtype=np.int64),
            user_to_inner={},
        )

    missing_anime_ids = sorted(
        set(positive_interactions["anime_id"]).difference(item_features.anime_to_inner)
    )

    if missing_anime_ids:
        raise ValueError(
            "Положительные взаимодействия содержат anime_id, "
            f"отсутствующие в item mapping: {missing_anime_ids}."
        )

    raw_user_ids = np.array(
        sorted(positive_interactions["user_id"].unique()),
        dtype=np.int64,
    )

    user_to_inner = {
        int(user_id): int(inner_user_id)
        for inner_user_id, user_id in enumerate(raw_user_ids)
    }

    row_indices = (
        positive_interactions["user_id"].map(user_to_inner).to_numpy(dtype=np.int32)
    )

    column_indices = (
        positive_interactions["anime_id"]
        .map(item_features.anime_to_inner)
        .to_numpy(dtype=np.int32)
    )

    weights = np.ones(
        len(positive_interactions),
        dtype=np.float32,
    )

    user_item_matrix = csr_matrix(
        (
            weights,
            (
                row_indices,
                column_indices,
            ),
        ),
        shape=(
            len(raw_user_ids),
            len(item_features.raw_anime_ids),
        ),
        dtype=np.float32,
    )

    user_profile_matrix = user_item_matrix @ item_features.item_feature_matrix

    user_profile_matrix = normalize(
        user_profile_matrix,
        norm="l2",
        axis=1,
        copy=False,
    ).tocsr()

    return ContentUserProfiles(
        user_profile_matrix=user_profile_matrix,
        raw_user_ids=raw_user_ids,
        user_to_inner=user_to_inner,
    )
