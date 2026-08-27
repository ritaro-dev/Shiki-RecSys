from collections.abc import Collection, Mapping
from dataclasses import dataclass

import pandas as pd

from shiki_recsys.evaluation.metrics import recall_at_k
from shiki_recsys.retrievers.common import RetrieverName
from shiki_recsys.retrievers.generation import build_candidate_union

RetrieverCandidatesByUser = Mapping[
    int,
    Mapping[RetrieverName, pd.DataFrame],
]


@dataclass(frozen=True)
class RetrieverContribution:
    """Store one retriever's contribution to candidate recall."""

    source: RetrieverName
    full_recall: float
    recall_without_source: float
    marginal_recall: float
    unique_relevant_hits: int


def candidate_union_recall(
    candidates_by_user: RetrieverCandidatesByUser,
    positive_items_by_user: Mapping[int, Collection[int]],
    *,
    candidate_count: int,
) -> float:
    """
    Calculate recall of the combined retriever candidate pool.

    Args:
        candidates_by_user: Retriever candidates grouped by user and source.
        positive_items_by_user: Positive target items for each user.
        candidate_count: Number of candidates taken from each source.

    Returns:
        Mean recall of the combined candidate pool.
    """
    recommendations_by_user: dict[int, list[int]] = {}

    for user_id in positive_items_by_user:
        retriever_candidates = candidates_by_user.get(
            user_id,
            {},
        )

        top_candidates = {
            source: candidates.head(candidate_count)
            for source, candidates in retriever_candidates.items()
        }

        candidate_union = build_candidate_union(top_candidates)

        recommendations_by_user[user_id] = candidate_union.tolist()

    evaluation_k = max(
        (len(recommendations) for recommendations in recommendations_by_user.values()),
        default=1,
    )

    result = recall_at_k(
        recommendations_by_user,
        positive_items_by_user,
        k=max(evaluation_k, 1),
    )

    return result.value


def analyze_retriever_contributions(
    candidates_by_user: RetrieverCandidatesByUser,
    positive_items_by_user: Mapping[int, Collection[int]],
    *,
    candidate_count: int,
) -> list[RetrieverContribution]:
    """
    Measure each retriever's marginal and unique contribution.

    Args:
        candidates_by_user: Retriever candidates grouped by user and source.
        positive_items_by_user: Positive target items for each user.
        candidate_count: Number of candidates taken from each source.

    Returns:
        Contribution statistics for each retriever.
    """
    sources = [
        source
        for source in RetrieverName
        if any(
            source in retriever_candidates
            for retriever_candidates in candidates_by_user.values()
        )
    ]

    full_recall = candidate_union_recall(
        candidates_by_user,
        positive_items_by_user,
        candidate_count=candidate_count,
    )

    relevant_hits_by_source: dict[
        RetrieverName,
        set[tuple[int, int]],
    ] = {}

    for source in sources:
        hits: set[tuple[int, int]] = set()

        for user_id, positive_items in positive_items_by_user.items():
            candidates = candidates_by_user.get(
                user_id,
                {},
            ).get(source)

            if candidates is None:
                continue

            recommended_items = set(
                candidates["anime_id"].head(candidate_count).astype(int)
            )

            hits.update(
                (user_id, anime_id)
                for anime_id in recommended_items.intersection(positive_items)
            )

        relevant_hits_by_source[source] = hits

    results: list[RetrieverContribution] = []

    for source in sources:
        remaining_candidates = {
            user_id: {
                other_source: candidates
                for other_source, candidates in retriever_candidates.items()
                if other_source != source
            }
            for user_id, retriever_candidates in candidates_by_user.items()
        }

        recall_without_source = candidate_union_recall(
            remaining_candidates,
            positive_items_by_user,
            candidate_count=candidate_count,
        )

        other_hits = set().union(
            *(
                hits
                for other_source, hits in relevant_hits_by_source.items()
                if other_source != source
            )
        )

        unique_relevant_hits = len(
            relevant_hits_by_source[source].difference(other_hits)
        )

        results.append(
            RetrieverContribution(
                source=source,
                full_recall=full_recall,
                recall_without_source=recall_without_source,
                marginal_recall=full_recall - recall_without_source,
                unique_relevant_hits=unique_relevant_hits,
            )
        )

    return results


def candidate_jaccard_matrix(
    candidates_by_user: RetrieverCandidatesByUser,
    *,
    candidate_count: int,
) -> pd.DataFrame:
    """
    Calculate pairwise Jaccard overlap of candidate user-item pairs.

    Args:
        candidates_by_user: Retriever candidates grouped by user and source.
        candidate_count: Number of candidates taken from each source.

    Returns:
        Symmetric Jaccard similarity matrix between retrievers.
    """
    sources = [
        source
        for source in RetrieverName
        if any(
            source in retriever_candidates
            for retriever_candidates in candidates_by_user.values()
        )
    ]

    candidate_pairs_by_source: dict[
        RetrieverName,
        set[tuple[int, int]],
    ] = {}

    for source in sources:
        pairs: set[tuple[int, int]] = set()

        for user_id, retriever_candidates in candidates_by_user.items():
            candidates = retriever_candidates.get(source)

            if candidates is None:
                continue

            pairs.update(
                (user_id, int(anime_id))
                for anime_id in candidates["anime_id"].head(candidate_count)
            )

        candidate_pairs_by_source[source] = pairs

    source_names = [source.value for source in sources]

    matrix = pd.DataFrame(
        index=source_names,
        columns=source_names,
        dtype=float,
    )

    for left_source in sources:
        for right_source in sources:
            left_pairs = candidate_pairs_by_source[left_source]
            right_pairs = candidate_pairs_by_source[right_source]

            union = left_pairs.union(right_pairs)

            matrix.loc[
                left_source.value,
                right_source.value,
            ] = len(left_pairs.intersection(right_pairs)) / len(union) if union else 1.0

    return matrix
