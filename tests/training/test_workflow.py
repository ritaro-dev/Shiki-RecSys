from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
from pandas.testing import assert_frame_equal

import shiki_recsys.training.workflow as workflow_module
from shiki_recsys.evaluation.model import ModelEvaluationResult
from shiki_recsys.training.workflow import run_training_workflow


def test_run_training_workflow_writes_production_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Verify the complete offline training workflow."""
    interactions = pd.DataFrame(
        {
            "user_id": [1],
            "anime_id": [10],
        }
    )
    catalog = pd.DataFrame(
        {
            "anime_id": [10],
        }
    )

    config = SimpleNamespace(
        candidate_generation=SimpleNamespace(
            retrieval_k=100,
        ),
        target=SimpleNamespace(
            positive_rating_threshold=8,
        ),
        retrievers=SimpleNamespace(
            content_tfidf=SimpleNamespace(
                max_positive_items=50,
            )
        ),
    )

    evaluation_bundle = MagicMock()
    split = MagicMock()

    evaluation_training = SimpleNamespace(
        bundle=evaluation_bundle,
        split=split,
    )

    evaluation = ModelEvaluationResult(
        ranking_k=20,
        recall_at_k=0.31,
        ndcg_at_k=0.24,
        evaluated_users=742,
    )

    production_bundle = MagicMock()
    artifact_path = tmp_path / "versions" / "20260817T120000Z"

    train_evaluation = MagicMock(
        return_value=evaluation_training,
    )
    evaluate = MagicMock(return_value=evaluation)
    train_production = MagicMock(
        return_value=production_bundle,
    )
    write_artifacts = MagicMock(
        return_value=artifact_path,
    )

    monkeypatch.setattr(
        workflow_module,
        "train_evaluation_bundle",
        train_evaluation,
    )
    monkeypatch.setattr(
        workflow_module,
        "evaluate_model_bundle",
        evaluate,
    )
    monkeypatch.setattr(
        workflow_module,
        "train_production_bundle",
        train_production,
    )
    monkeypatch.setattr(
        workflow_module,
        "write_model_artifacts",
        write_artifacts,
    )

    created_at = datetime(
        2026,
        8,
        17,
        12,
        0,
        tzinfo=UTC,
    )

    result = run_training_workflow(
        interactions=interactions,
        catalog=catalog,
        config=config,
        artifacts_dir=tmp_path,
        artifact_version="20260817T120000Z",
        created_at=created_at,
    )

    train_evaluation.assert_called_once()

    evaluation_call = train_evaluation.call_args

    assert_frame_equal(
        evaluation_call.kwargs["interactions"],
        interactions,
    )
    assert_frame_equal(
        evaluation_call.kwargs["catalog"],
        catalog,
    )
    assert evaluation_call.kwargs["config"] is config

    evaluate.assert_called_once_with(
        bundle=evaluation_bundle,
        split=split,
        config=config,
    )

    train_production.assert_called_once()

    production_call = train_production.call_args

    assert production_call.kwargs["split"] is split
    assert_frame_equal(
        production_call.kwargs["catalog"],
        catalog,
    )
    assert production_call.kwargs["config"] is config

    write_artifacts.assert_called_once()

    artifact_call = write_artifacts.call_args

    assert artifact_call.kwargs["artifacts_dir"] == tmp_path
    assert artifact_call.kwargs["bundle"] is production_bundle
    assert artifact_call.kwargs["training_config"] is config
    assert artifact_call.kwargs["evaluation"] is evaluation

    metadata = artifact_call.kwargs["metadata"]

    assert metadata.artifact_version == "20260817T120000Z"
    assert metadata.created_at == created_at
    assert metadata.inference.retrieval_k == 100
    assert metadata.inference.positive_rating_threshold == 8
    assert metadata.inference.max_positive_items == 50

    assert result.artifact_path == artifact_path
    assert result.evaluation is evaluation
