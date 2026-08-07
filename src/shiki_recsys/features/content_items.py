from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder


@dataclass(frozen=True)
class ContentItemFeatures:
    """Хранит content-представление каталога."""

    item_feature_matrix: csr_matrix
    raw_anime_ids: np.ndarray
    anime_to_inner: dict[int, int]


def _bucket_episodes(value: Any) -> str:
    """Преобразует число эпизодов в категориальный диапазон."""

    if pd.isna(value) or value <= 0:
        return "unknown"
    if value == 1:
        return "1"
    if value <= 6:
        return "2-6"
    if value <= 13:
        return "7-13"
    if value <= 26:
        return "14-26"
    if value <= 52:
        return "27-52"
    if value <= 100:
        return "53-100"
    return "101+"


def _bucket_duration(value: Any) -> str:
    """Преобразует длительность в категориальный диапазон."""

    if pd.isna(value) or value <= 0:
        return "unknown"
    if value <= 5:
        return "1-5"
    if value <= 15:
        return "6-15"
    if value <= 30:
        return "16-30"
    if value <= 60:
        return "31-60"
    if value <= 120:
        return "61-120"
    return "121+"


def _build_binary_feature_matrix(
    catalog: pd.DataFrame,
) -> csr_matrix:
    """Строит бинарную матрицу content-признаков аниме."""

    genre_encoder = MultiLabelBinarizer(sparse_output=True)
    studio_encoder = MultiLabelBinarizer(sparse_output=True)

    genre_matrix = genre_encoder.fit_transform(
        catalog["genres"],
    ).astype(np.float32)

    studio_matrix = studio_encoder.fit_transform(
        catalog["studios"],
    ).astype(np.float32)

    categorical_data = pd.DataFrame(
        {
            "kind": catalog["kind"].fillna("unknown"),
            "rating": catalog["rating"].fillna("unknown"),
            "episodes_bucket": catalog["episodes"].map(_bucket_episodes),
            "duration_bucket": catalog["duration"].map(_bucket_duration),
        }
    )

    categorical_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float32,
    )

    categorical_matrix = categorical_encoder.fit_transform(
        categorical_data,
    )

    return hstack(
        [
            genre_matrix,
            studio_matrix,
            categorical_matrix,
        ],
        format="csr",
        dtype=np.float32,
    )


def build_content_item_features(
    catalog: pd.DataFrame,
) -> ContentItemFeatures:
    """
    Формирует TF-IDF content-признаки аниме.

    Args:
        catalog: Подготовленный общий каталог аниме.

    Returns:
        Матрицу item-признаков и соответствие anime_id
        внутренним индексам.

    Raises:
        ValueError: Если каталог пуст или не содержит
            необходимых столбцов.
    """
    required_columns = {
        "anime_id",
        "genres",
        "studios",
        "kind",
        "rating",
        "episodes",
        "duration",
    }
    missing_columns = required_columns.difference(catalog.columns)

    if missing_columns:
        raise ValueError(f"В catalog отсутствуют столбцы: {sorted(missing_columns)}.")

    if catalog.empty:
        raise ValueError("catalog не должен быть пустым.")

    binary_matrix = _build_binary_feature_matrix(catalog)

    transformer = TfidfTransformer(
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    )

    item_feature_matrix = transformer.fit_transform(
        binary_matrix,
    ).astype(np.float32)

    raw_anime_ids = catalog["anime_id"].to_numpy(
        dtype=np.int64,
        copy=True,
    )

    anime_to_inner = {
        int(anime_id): int(inner_anime_id)
        for inner_anime_id, anime_id in enumerate(raw_anime_ids)
    }

    return ContentItemFeatures(
        item_feature_matrix=item_feature_matrix,
        raw_anime_ids=raw_anime_ids,
        anime_to_inner=anime_to_inner,
    )
