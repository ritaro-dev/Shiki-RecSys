import math
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from shiki_recsys.config.training import (
    load_training_config,
)


def _write_config(
    path: Path,
    content: str,
) -> None:
    path.write_text(
        content,
        encoding="utf-8",
    )


def _build_valid_config() -> dict[str, Any]:
    return {
        "random_seed": 42,
        "dataset": {
            "split": {
                "validation_fraction": 0.1,
                "test_fraction": 0.1,
                "min_interactions_per_user": 10,
            },
        },
        "target": {
            "positive_rating_threshold": 8,
        },
        "retrievers": {
            "explicit_svd": {
                "min_item_explicit_ratings": 5,
                "n_factors": 64,
                "n_epochs": 15,
                "biased": True,
                "learning_rate": 0.015,
                "regularization": 0.10,
                "init_mean": 0.0,
                "init_std_dev": 0.1,
            },
            "implicit_als": {
                "signal_confidences": {
                    "rating_8_10": 2.0,
                    "watching": 1.0,
                    "rewatching": 1.0,
                    "completed": 0.5,
                    "planned": 0.5,
                    "on_hold": 0.0,
                    "rating_4_5": -1.0,
                    "rating_1_3": -2.0,
                },
                "factors": 96,
                "regularization": 0.1,
                "alpha": 1.0,
                "iterations": 40,
            },
            "content_tfidf": {
                "max_positive_items": 50,
            },
        },
        "candidate_generation": {
            "retrieval_k": 100,
        },
        "ranker": {
            "iterations": 1183,
            "depth": 7,
            "learning_rate": 0.05,
            "l2_leaf_reg": 5.0,
        },
        "evaluation": {
            "ranking_k": 20,
        },
    }


def test_load_training_config_returns_typed_config(
    tmp_path: Path,
):
    config_path = tmp_path / "training.yaml"
    config_content = _build_valid_config()

    _write_config(
        config_path,
        yaml.safe_dump(
            config_content,
            sort_keys=False,
        ),
    )

    config = load_training_config(config_path)

    assert config.random_seed == 42

    assert config.dataset.split.validation_fraction == 0.1
    assert config.dataset.split.test_fraction == 0.1
    assert config.dataset.split.min_interactions_per_user == 10

    assert config.target.positive_rating_threshold == 8

    explicit_svd_config = config.retrievers.explicit_svd

    assert explicit_svd_config.min_item_explicit_ratings == 5
    assert explicit_svd_config.n_factors == 64
    assert explicit_svd_config.n_epochs == 15
    assert explicit_svd_config.biased is True
    assert explicit_svd_config.learning_rate == 0.015
    assert explicit_svd_config.regularization == 0.10
    assert explicit_svd_config.init_mean == 0.0
    assert explicit_svd_config.init_std_dev == 0.1

    implicit_als_config = config.retrievers.implicit_als
    signal_confidences = implicit_als_config.signal_confidences

    assert signal_confidences.rating_8_10 == 2.0
    assert signal_confidences.watching == 1.0
    assert signal_confidences.rewatching == 1.0
    assert signal_confidences.completed == 0.5
    assert signal_confidences.planned == 0.5
    assert signal_confidences.on_hold == 0.0
    assert signal_confidences.rating_4_5 == -1.0
    assert signal_confidences.rating_1_3 == -2.0

    assert implicit_als_config.factors == 96
    assert implicit_als_config.regularization == 0.1
    assert implicit_als_config.alpha == 1.0
    assert implicit_als_config.iterations == 40

    content_tfidf_config = config.retrievers.content_tfidf

    assert content_tfidf_config.max_positive_items == 50

    assert config.candidate_generation.retrieval_k == 100

    ranker_config = config.ranker

    assert ranker_config.iterations == 1183
    assert ranker_config.depth == 7
    assert ranker_config.learning_rate == 0.05
    assert ranker_config.l2_leaf_reg == 5.0

    assert config.evaluation.ranking_k == 20


def test_load_training_config_rejects_invalid_fraction_sum(
    tmp_path: Path,
):
    config_path = tmp_path / "training.yaml"
    config_content = _build_valid_config()

    config_content["dataset"]["split"]["validation_fraction"] = 0.6
    config_content["dataset"]["split"]["test_fraction"] = 0.4

    _write_config(
        config_path,
        yaml.safe_dump(
            config_content,
            sort_keys=False,
        ),
    )

    with pytest.raises(
        ValidationError,
        match="должна быть меньше 1",
    ):
        load_training_config(config_path)


def test_load_training_config_rejects_unknown_fields(
    tmp_path: Path,
):
    config_path = tmp_path / "training.yaml"
    config_content = _build_valid_config()

    config_content["unknown_parameter"] = 100

    _write_config(
        config_path,
        yaml.safe_dump(
            config_content,
            sort_keys=False,
        ),
    )

    with pytest.raises(
        ValidationError,
        match="unknown_parameter",
    ):
        load_training_config(config_path)


@pytest.mark.parametrize(
    (
        "min_interactions_per_user",
        "min_item_explicit_ratings",
        "max_positive_items",
    ),
    [
        (0, 5, 50),
        (-1, 5, 50),
        (10, 0, 50),
        (10, -1, 50),
        (10, 5, 0),
        (10, 5, -1),
    ],
)
def test_load_training_config_rejects_non_positive_count_parameters(
    tmp_path: Path,
    min_interactions_per_user: int,
    min_item_explicit_ratings: int,
    max_positive_items: int,
):
    config_path = tmp_path / "training.yaml"
    config_content = _build_valid_config()

    config_content["dataset"]["split"]["min_interactions_per_user"] = (
        min_interactions_per_user
    )
    config_content["retrievers"]["explicit_svd"]["min_item_explicit_ratings"] = (
        min_item_explicit_ratings
    )
    config_content["retrievers"]["content_tfidf"]["max_positive_items"] = (
        max_positive_items
    )

    _write_config(
        config_path,
        yaml.safe_dump(
            config_content,
            sort_keys=False,
        ),
    )

    with pytest.raises(ValidationError):
        load_training_config(config_path)


def test_load_training_config_rejects_invalid_yaml(
    tmp_path: Path,
):
    config_path = tmp_path / "training.yaml"

    _write_config(
        config_path,
        """
dataset:
  split:
    validation_fraction: [
""",
    )

    with pytest.raises(yaml.YAMLError):
        load_training_config(config_path)


@pytest.mark.parametrize(
    "positive_rating_threshold",
    [
        0,
        -1,
        10.1,
        ".nan",
        ".inf",
    ],
)
def test_load_training_config_rejects_invalid_positive_rating_threshold(
    tmp_path: Path,
    positive_rating_threshold: float | str,
) -> None:
    config_path = tmp_path / "training.yaml"
    config_content = _build_valid_config()

    config_content["target"]["positive_rating_threshold"] = positive_rating_threshold

    _write_config(
        config_path,
        yaml.safe_dump(
            config_content,
            sort_keys=False,
        ),
    )

    with pytest.raises(ValidationError):
        load_training_config(config_path)


@pytest.mark.parametrize(
    (
        "parameter_name",
        "invalid_value",
    ),
    [
        ("n_factors", 0),
        ("n_epochs", 0),
        ("learning_rate", 0),
        ("learning_rate", math.inf),
        ("regularization", -0.1),
        ("regularization", math.nan),
        ("init_mean", math.inf),
        ("init_std_dev", 0),
        ("init_std_dev", math.nan),
    ],
)
def test_load_training_config_rejects_invalid_svd_parameters(
    tmp_path: Path,
    parameter_name: str,
    invalid_value: float,
) -> None:
    config_path = tmp_path / "training.yaml"
    config_content = _build_valid_config()

    config_content["retrievers"]["explicit_svd"][parameter_name] = invalid_value

    _write_config(
        config_path,
        yaml.safe_dump(
            config_content,
            sort_keys=False,
        ),
    )

    with pytest.raises(ValidationError):
        load_training_config(config_path)


@pytest.mark.parametrize(
    (
        "parameter_group",
        "parameter_name",
        "invalid_value",
    ),
    [
        ("model", "factors", 0),
        ("model", "regularization", -0.1),
        ("model", "regularization", math.nan),
        ("model", "alpha", 0),
        ("model", "alpha", math.inf),
        ("model", "iterations", 0),
        ("signals", "rating_8_10", 0),
        ("signals", "rating_8_10", math.inf),
        ("signals", "watching", -0.1),
        ("signals", "rewatching", -0.1),
        ("signals", "completed", -0.1),
        ("signals", "planned", -0.1),
        ("signals", "on_hold", -0.1),
        ("signals", "rating_4_5", 0.1),
        ("signals", "rating_1_3", 0.1),
    ],
)
def test_load_training_config_rejects_invalid_als_parameters(
    tmp_path: Path,
    parameter_group: str,
    parameter_name: str,
    invalid_value: float,
) -> None:
    config_path = tmp_path / "training.yaml"
    config_content = _build_valid_config()

    if parameter_group == "signals":
        config_content["retrievers"]["implicit_als"]["signal_confidences"][
            parameter_name
        ] = invalid_value
    else:
        config_content["retrievers"]["implicit_als"][parameter_name] = invalid_value

    _write_config(
        config_path,
        yaml.safe_dump(
            config_content,
            sort_keys=False,
        ),
    )

    with pytest.raises(ValidationError):
        load_training_config(config_path)


@pytest.mark.parametrize(
    "ranking_k",
    [0, -1],
)
def test_load_training_config_rejects_invalid_ranking_k(
    tmp_path: Path,
    ranking_k: int,
) -> None:
    """Reject non-positive ranking evaluation cutoffs."""
    config_path = tmp_path / "training.yaml"
    config_content = _build_valid_config()

    config_content["evaluation"]["ranking_k"] = ranking_k

    _write_config(
        config_path,
        yaml.safe_dump(
            config_content,
            sort_keys=False,
        ),
    )

    with pytest.raises(ValidationError):
        load_training_config(config_path)
