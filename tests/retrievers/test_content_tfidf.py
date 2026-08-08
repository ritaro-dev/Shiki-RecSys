import numpy as np
import pytest
from scipy.sparse import csr_matrix

from shiki_recsys.features.content_items import ContentItemFeatures
from shiki_recsys.features.content_users import ContentUserProfiles
from shiki_recsys.retrievers.common import RetrieverName
from shiki_recsys.retrievers.content_tfidf import ContentTFIDFRetriever


def _build_item_features() -> ContentItemFeatures:
    return ContentItemFeatures(
        item_feature_matrix=csr_matrix(
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1 / np.sqrt(2), 1 / np.sqrt(2)],
                ],
                dtype=np.float32,
            )
        ),
        raw_anime_ids=np.array(
            [30, 10, 20],
            dtype=np.int64,
        ),
        anime_to_inner={
            30: 0,
            10: 1,
            20: 2,
        },
    )


def _build_user_profiles() -> ContentUserProfiles:
    return ContentUserProfiles(
        user_profile_matrix=csr_matrix(
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            )
        ),
        raw_user_ids=np.array(
            [1, 2],
            dtype=np.int64,
        ),
        user_to_inner={
            1: 0,
            2: 1,
        },
    )


@pytest.fixture
def fitted_retriever() -> ContentTFIDFRetriever:
    return ContentTFIDFRetriever().fit(
        _build_item_features(),
        _build_user_profiles(),
    )


def test_supported_anime_ids_rejects_call_before_fit() -> None:
    retriever = ContentTFIDFRetriever()

    with pytest.raises(
        RuntimeError,
        match="ещё не обучен",
    ):
        _ = retriever.supported_anime_ids


def test_retrieve_rejects_call_before_fit() -> None:
    retriever = ContentTFIDFRetriever()

    with pytest.raises(
        RuntimeError,
        match="ещё не обучен",
    ):
        retriever.retrieve(
            user_id=1,
        )


def test_fit_rejects_incompatible_feature_dimensions() -> None:
    item_features = _build_item_features()

    user_profiles = ContentUserProfiles(
        user_profile_matrix=csr_matrix(
            np.ones(
                (1, 3),
                dtype=np.float32,
            )
        ),
        raw_user_ids=np.array(
            [1],
            dtype=np.int64,
        ),
        user_to_inner={
            1: 0,
        },
    )

    with pytest.raises(
        ValueError,
        match="не совпадают",
    ):
        ContentTFIDFRetriever().fit(
            item_features,
            user_profiles,
        )


def test_fit_sets_supported_anime_ids(
    fitted_retriever: ContentTFIDFRetriever,
) -> None:
    assert fitted_retriever.supported_anime_ids == frozenset(
        {
            10,
            20,
            30,
        }
    )


def test_fit_accepts_empty_user_profiles() -> None:
    item_features = _build_item_features()

    user_profiles = ContentUserProfiles(
        user_profile_matrix=csr_matrix(
            (0, 2),
            dtype=np.float32,
        ),
        raw_user_ids=np.array(
            [],
            dtype=np.int64,
        ),
        user_to_inner={},
    )

    retriever = ContentTFIDFRetriever().fit(
        item_features,
        user_profiles,
    )

    candidates = retriever.retrieve(
        user_id=1,
    )

    assert candidates.empty
    assert retriever.supported_anime_ids == frozenset(
        {
            10,
            20,
            30,
        }
    )


def test_retrieve_returns_ranked_candidates_for_known_user(
    fitted_retriever: ContentTFIDFRetriever,
) -> None:
    candidates = fitted_retriever.retrieve(
        user_id=1,
    )

    assert candidates.columns.tolist() == [
        "anime_id",
        "score",
        "source",
        "source_rank",
    ]

    assert candidates["anime_id"].tolist() == [
        30,
        20,
        10,
    ]

    np.testing.assert_allclose(
        candidates["score"],
        [
            1.0,
            1 / np.sqrt(2),
            0.0,
        ],
        rtol=1e-6,
        atol=1e-7,
    )

    assert candidates["source"].tolist() == [
        RetrieverName.CONTENT_TFIDF.value,
        RetrieverName.CONTENT_TFIDF.value,
        RetrieverName.CONTENT_TFIDF.value,
    ]

    assert candidates["source_rank"].tolist() == [
        1,
        2,
        3,
    ]

    assert candidates["anime_id"].dtype == "int64"
    assert candidates["score"].dtype == "float64"
    assert candidates["source"].dtype == "string"
    assert candidates["source_rank"].dtype == "int32"


def test_retrieve_limits_candidate_count(
    fitted_retriever: ContentTFIDFRetriever,
) -> None:
    candidates = fitted_retriever.retrieve(
        user_id=1,
        candidate_count=2,
    )

    assert candidates["anime_id"].tolist() == [
        30,
        20,
    ]

    assert candidates["source_rank"].tolist() == [
        1,
        2,
    ]


@pytest.mark.parametrize(
    "candidate_count",
    [
        0,
        -1,
    ],
)
def test_retrieve_rejects_invalid_candidate_count(
    fitted_retriever: ContentTFIDFRetriever,
    candidate_count: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="candidate_count",
    ):
        fitted_retriever.retrieve(
            user_id=1,
            candidate_count=candidate_count,
        )


def test_retrieve_returns_typed_empty_frame_for_unknown_user(
    fitted_retriever: ContentTFIDFRetriever,
) -> None:
    candidates = fitted_retriever.retrieve(
        user_id=999,
    )

    assert candidates.empty

    assert candidates.columns.tolist() == [
        "anime_id",
        "score",
        "source",
        "source_rank",
    ]

    assert candidates["anime_id"].dtype == "int64"
    assert candidates["score"].dtype == "float64"
    assert candidates["source"].dtype == "string"
    assert candidates["source_rank"].dtype == "int32"
