import numpy as np
import pandas as pd


def build_als_signed_interactions(
    train_interactions: pd.DataFrame,
    *,
    rating_8_10_confidence: float,
    watching_confidence: float,
    rewatching_confidence: float,
    completed_confidence: float,
    planned_confidence: float,
    on_hold_confidence: float,
    rating_4_5_confidence: float,
    rating_1_3_confidence: float,
) -> pd.DataFrame:
    """
    Формирует обучающие взаимодействия implicit ALS.

    Args:
        train_interactions: Подготовленные взаимодействия
            из train-части.
        rating_8_10_confidence: Вес оценок от 8 до 10.
        watching_confidence: Вес статуса watching без оценки.
        rewatching_confidence: Вес статуса rewatching без оценки.
        completed_confidence: Вес статуса completed без оценки.
        planned_confidence: Вес статуса planned без оценки.
        on_hold_confidence: Вес статуса on_hold без оценки.
        rating_4_5_confidence: Вес оценок от 4 до 5.
        rating_1_3_confidence: Вес оценок от 1 до 3.

    Returns:
        Взаимодействия с ненулевым ALS-confidence.

    Raises:
        ValueError: Если входные данные или веса сигналов
            некорректны либо после назначения весов
            не осталось подходящих взаимодействий.
    """
    required_columns = {
        "user_id",
        "anime_id",
        "rating",
        "status",
    }
    missing_columns = required_columns.difference(train_interactions.columns)

    if missing_columns:
        raise ValueError(
            f"В train_interactions отсутствуют столбцы: {sorted(missing_columns)}."
        )

    if train_interactions.empty:
        raise ValueError("train_interactions не должен быть пустым.")

    confidence_values = {
        "rating_8_10_confidence": rating_8_10_confidence,
        "watching_confidence": watching_confidence,
        "rewatching_confidence": rewatching_confidence,
        "completed_confidence": completed_confidence,
        "planned_confidence": planned_confidence,
        "on_hold_confidence": on_hold_confidence,
        "rating_4_5_confidence": rating_4_5_confidence,
        "rating_1_3_confidence": rating_1_3_confidence,
    }

    for parameter_name, parameter_value in confidence_values.items():
        if not np.isfinite(parameter_value):
            raise ValueError(f"{parameter_name} должен быть конечным числом.")

    non_negative_confidence_names = (
        "watching_confidence",
        "rewatching_confidence",
        "completed_confidence",
        "planned_confidence",
        "on_hold_confidence",
    )

    for parameter_name in non_negative_confidence_names:
        if confidence_values[parameter_name] < 0:
            raise ValueError(f"{parameter_name} не может быть отрицательным.")

    if rating_8_10_confidence <= 0:
        raise ValueError("rating_8_10_confidence должен быть больше 0.")

    if rating_4_5_confidence > 0:
        raise ValueError("rating_4_5_confidence не может быть положительным.")

    if rating_1_3_confidence > 0:
        raise ValueError("rating_1_3_confidence не может быть положительным.")

    interactions = train_interactions.loc[
        :,
        [
            "user_id",
            "anime_id",
            "rating",
            "status",
        ],
    ].copy()

    interactions["confidence"] = np.float32(0.0)

    interactions.loc[
        interactions["rating"].between(8, 10),
        "confidence",
    ] = np.float32(rating_8_10_confidence)

    interactions.loc[
        interactions["rating"].between(4, 5),
        "confidence",
    ] = np.float32(rating_4_5_confidence)

    interactions.loc[
        interactions["rating"].between(1, 3),
        "confidence",
    ] = np.float32(rating_1_3_confidence)

    without_explicit_rating = interactions["rating"].eq(0)

    status_confidences = {
        "watching": watching_confidence,
        "rewatching": rewatching_confidence,
        "completed": completed_confidence,
        "planned": planned_confidence,
        "on_hold": on_hold_confidence,
    }

    for status_name, status_confidence in status_confidences.items():
        interactions.loc[
            (without_explicit_rating & interactions["status"].eq(status_name)),
            "confidence",
        ] = np.float32(status_confidence)

    signed_interactions = (
        interactions.loc[
            interactions["confidence"] != 0,
            [
                "user_id",
                "anime_id",
                "confidence",
            ],
        ]
        .sort_values(
            [
                "user_id",
                "anime_id",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if signed_interactions.empty:
        raise ValueError("После назначения ALS-confidence не осталось взаимодействий.")

    if not (signed_interactions["confidence"] > 0).any():
        raise ValueError("ALS-взаимодействия не содержат положительных сигналов.")

    signed_interactions["user_id"] = signed_interactions["user_id"].astype("int64")
    signed_interactions["anime_id"] = signed_interactions["anime_id"].astype("int64")
    signed_interactions["confidence"] = signed_interactions["confidence"].astype(
        "float32"
    )

    return signed_interactions
