import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from shiki_recsys.config.training import RankerConfig, TrainingConfig
from shiki_recsys.evaluation.model import ModelEvaluationResult
from shiki_recsys.features.content_items import ContentItemFeatures
from shiki_recsys.features.content_users import ContentUserProfiles
from shiki_recsys.inference.artifact_loader import (
    load_current_model_artifacts,
    load_model_artifacts,
)
from shiki_recsys.inference.model_bundle import ModelBundle
from shiki_recsys.model_artifacts import (
    ArtifactInferenceConfig,
    ArtifactMetadata,
)
from shiki_recsys.ranking.catboost import CatBoostRankerModel
from shiki_recsys.retrievers.content_tfidf import ContentTFIDFRetriever
from shiki_recsys.retrievers.explicit_svd import ExplicitSVDRetriever
from shiki_recsys.retrievers.implicit_als import ImplicitALSRetriever
from shiki_recsys.retrievers.popularity import PopularityRetriever
from shiki_recsys.training.artifact_writer import write_model_artifacts


def _write_artifact_version(
    artifacts_dir: Path,
    *,
    directory_version: str,
    metadata_version: str | None = None,
) -> Path:
    """Создаёт минимальную artifact-версию для теста."""
    version_dir = artifacts_dir / "versions" / directory_version
    version_dir.mkdir(parents=True)

    (version_dir / "model_bundle.joblib").write_bytes(b"bundle")
    (version_dir / "metadata.json").write_text(
        json.dumps(
            {
                "artifact_version": metadata_version or directory_version,
                "created_at": "2026-08-10T12:00:00+00:00",
                "inference": {
                    "retrieval_k": 100,
                    "positive_rating_threshold": 8,
                    "max_positive_items": 50,
                },
            }
        ),
        encoding="utf-8",
    )

    return version_dir


def test_load_model_artifacts_loads_specific_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Проверяет загрузку конкретной artifact-версии."""
    version_dir = _write_artifact_version(
        tmp_path,
        directory_version="v1",
    )
    bundle = Mock(spec=ModelBundle)

    load_mock = Mock(return_value=bundle)
    monkeypatch.setattr(
        "shiki_recsys.inference.artifact_loader.joblib.load",
        load_mock,
    )

    loaded_bundle, metadata = load_model_artifacts(
        artifacts_dir=tmp_path,
        artifact_version="v1",
    )

    assert loaded_bundle is bundle
    assert metadata.artifact_version == "v1"
    assert metadata.created_at == datetime(
        2026,
        8,
        10,
        12,
        0,
        tzinfo=UTC,
    )
    assert metadata.inference == ArtifactInferenceConfig(
        retrieval_k=100,
        positive_rating_threshold=8,
        max_positive_items=50,
    )

    load_mock.assert_called_once_with(version_dir / "model_bundle.joblib")


def test_load_current_model_artifacts_resolves_current_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Проверяет загрузку версии из current."""
    _write_artifact_version(
        tmp_path,
        directory_version="v2",
    )
    (tmp_path / "current").write_text(
        "v2\n",
        encoding="utf-8",
    )

    bundle = Mock(spec=ModelBundle)
    monkeypatch.setattr(
        "shiki_recsys.inference.artifact_loader.joblib.load",
        Mock(return_value=bundle),
    )

    loaded_bundle, metadata = load_current_model_artifacts(
        artifacts_dir=tmp_path,
    )

    assert loaded_bundle is bundle
    assert metadata.artifact_version == "v2"


def test_load_model_artifacts_rejects_metadata_version_mismatch(
    tmp_path: Path,
) -> None:
    """Проверяет согласованность директории и metadata."""
    _write_artifact_version(
        tmp_path,
        directory_version="v1",
        metadata_version="v2",
    )

    with pytest.raises(
        ValueError,
        match="не соответствует",
    ):
        load_model_artifacts(
            artifacts_dir=tmp_path,
            artifact_version="v1",
        )


def test_model_bundle_survives_artifact_round_trip(
    tmp_path: Path,
) -> None:
    """Verify serialization and inference of a real model bundle."""
    explicit_interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "anime_id": [10, 20, 10, 20],
            "rating": [9.0, 8.0, 7.0, 6.0],
        }
    )

    popularity = PopularityRetriever(
        relevance_threshold=8,
    ).fit(explicit_interactions)

    explicit_svd = ExplicitSVDRetriever(
        min_item_explicit_ratings=1,
        n_factors=2,
        n_epochs=2,
        biased=True,
        learning_rate=0.01,
        regularization=0.1,
        init_mean=0.0,
        init_std_dev=0.1,
        random_state=42,
    ).fit(explicit_interactions)

    signed_interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "anime_id": [10, 20, 10, 20],
            "confidence": pd.Series(
                [2.0, -1.0, 1.0, 2.0],
                dtype="float32",
            ),
        }
    )

    implicit_als = ImplicitALSRetriever(
        factors=2,
        regularization=0.1,
        alpha=1.0,
        iterations=2,
        random_state=42,
    ).fit(signed_interactions)

    item_features = ContentItemFeatures(
        item_feature_matrix=csr_matrix(
            np.eye(
                2,
                dtype=np.float32,
            )
        ),
        raw_anime_ids=np.array(
            [10, 20],
            dtype=np.int64,
        ),
        anime_to_inner={
            10: 0,
            20: 1,
        },
    )

    user_profiles = ContentUserProfiles(
        user_profile_matrix=csr_matrix(
            np.eye(
                2,
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

    content_tfidf = ContentTFIDFRetriever().fit(
        item_features,
        user_profiles,
    )

    ranker = CatBoostRankerModel(
        RankerConfig(
            iterations=2,
            depth=2,
            learning_rate=0.1,
            l2_leaf_reg=1.0,
        ),
        random_seed=42,
    )

    ranker_data = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "anime_id": [10, 20, 10, 20],
            "feature": [1.0, 0.0, 0.0, 1.0],
            "target": [1, 0, 0, 1],
        }
    )
    ranker.fit(ranker_data)

    bundle = ModelBundle(
        popularity=popularity,
        explicit_svd=explicit_svd,
        implicit_als=implicit_als,
        content_tfidf=content_tfidf,
        ranker=ranker,
    )

    metadata = ArtifactMetadata(
        artifact_version="v1",
        created_at=datetime(
            2026,
            8,
            10,
            12,
            0,
            tzinfo=UTC,
        ),
        inference=ArtifactInferenceConfig(
            retrieval_k=100,
            positive_rating_threshold=8,
            max_positive_items=50,
        ),
    )

    training_config = Mock(spec=TrainingConfig)
    training_config.model_dump.return_value = {
        "random_seed": 42,
    }

    evaluation = ModelEvaluationResult(
        ranking_k=20,
        recall_at_k=0.2,
        ndcg_at_k=0.25,
        evaluated_users=2,
    )

    write_model_artifacts(
        artifacts_dir=tmp_path,
        bundle=bundle,
        metadata=metadata,
        training_config=training_config,
        evaluation=evaluation,
    )

    loaded_bundle, loaded_metadata = load_model_artifacts(
        artifacts_dir=tmp_path,
        artifact_version="v1",
    )

    assert loaded_metadata == metadata

    assert (
        loaded_bundle.popularity.retrieve(
            candidate_count=1,
        ).shape[0]
        == 1
    )

    assert loaded_bundle.explicit_svd.supports_user(1)
    assert loaded_bundle.implicit_als.supports_user(1)
    assert loaded_bundle.content_tfidf.supports_user(1)

    predictions = loaded_bundle.ranker.predict(
        pd.DataFrame(
            {
                "feature": [1.0, 0.0],
            }
        )
    )

    assert predictions.shape == (2,)
    assert np.isfinite(predictions).all()
