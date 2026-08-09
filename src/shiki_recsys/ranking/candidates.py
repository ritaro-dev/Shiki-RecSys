from collections.abc import Mapping

import numpy as np
import pandas as pd

from shiki_recsys.retrievers.common import RetrieverName


def build_ranker_candidate_features(
    user_id: int,
    anime_ids: np.ndarray,
    retriever_candidates: Mapping[RetrieverName, pd.DataFrame],
    retriever_scores: Mapping[RetrieverName, np.ndarray],
) -> pd.DataFrame:
    """
    Формирует признаки кандидатов для ranker-а.

    Args:
        user_id: Идентификатор пользователя.
        anime_ids: Anime общего candidate set.
        retriever_candidates: Top-K кандидаты retriever-ов.
        retriever_scores: Scores retriever-ов для candidate set.

    Returns:
        Таблицу кандидатов с признаками retriever-ов.
    """
    features = pd.DataFrame(
        {
            "anime_id": pd.Series(
                anime_ids,
                dtype="int64",
            ),
        }
    )

    rank_columns = []
    from_columns = []
    rr_columns = []

    for source in RetrieverName:
        source_name = source.value

        score_column = f"score_{source_name}"
        rank_column = f"rank_{source_name}"
        from_column = f"from_{source_name}"
        rr_column = f"rr_{source_name}"

        rank_columns.append(rank_column)
        from_columns.append(from_column)
        rr_columns.append(rr_column)

        features[score_column] = retriever_scores[source]

        source_features = (
            retriever_candidates[source]
            .loc[
                :,
                [
                    "anime_id",
                    "source_rank",
                ],
            ]
            .rename(
                columns={
                    "source_rank": rank_column,
                }
            )
        )

        source_features[rank_column] = source_features[rank_column].astype("float64")
        source_features[from_column] = 1
        source_features[rr_column] = 1.0 / source_features[rank_column]

        features = features.merge(
            source_features,
            on="anime_id",
            how="left",
        )

        features[from_column] = features[from_column].fillna(0).astype("int8")
        features[rr_column] = features[rr_column].fillna(0.0)

    features["retriever_count"] = features[from_columns].sum(axis=1).astype("int8")
    features["best_rank"] = features[rank_columns].min(axis=1)
    features["mean_rank_present"] = features[rank_columns].mean(axis=1)
    features["rr_sum"] = features[rr_columns].sum(axis=1)
    features["rr_max"] = features[rr_columns].max(axis=1)

    features.insert(
        0,
        "user_id",
        pd.Series(
            user_id,
            index=features.index,
            dtype="int64",
        ),
    )

    return features.sort_values(
        [
            "best_rank",
            "rr_sum",
            "anime_id",
        ],
        ascending=[
            True,
            False,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)
