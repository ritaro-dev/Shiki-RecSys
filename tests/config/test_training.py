import math
from pathlib import Path

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


def test_load_training_config_returns_typed_config(
    tmp_path: Path,
):
    config_path = tmp_path / "training.yaml"

    _write_config(
        config_path,
        """
random_seed: 42

dataset:
  split:
    validation_fraction: 0.1
    test_fraction: 0.1
    min_interactions_per_user: 10

target:
  positive_rating_threshold: 8

retrievers:
  explicit_svd:
    min_item_explicit_ratings: 5
    n_factors: 64
    n_epochs: 15
    biased: true
    learning_rate: 0.015
    regularization: 0.10
    init_mean: 0.0
    init_std_dev: 0.1

  implicit_als:
    signal_confidences:
      rating_8_10: 2.0
      watching: 1.0
      rewatching: 1.0
      completed: 0.5
      planned: 0.5
      on_hold: 0.0
      rating_4_5: -1.0
      rating_1_3: -2.0
    factors: 96
    regularization: 0.1
    alpha: 1.0
    iterations: 40
""",
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


def test_load_training_config_rejects_invalid_fraction_sum(
    tmp_path: Path,
):
    config_path = tmp_path / "training.yaml"

    _write_config(
        config_path,
        """
random_seed: 42

dataset:
  split:
    validation_fraction: 0.6
    test_fraction: 0.4
    min_interactions_per_user: 10

target:
  positive_rating_threshold: 8

retrievers:
  explicit_svd:
    min_item_explicit_ratings: 5
    n_factors: 64
    n_epochs: 15
    biased: true
    learning_rate: 0.015
    regularization: 0.10
    init_mean: 0.0
    init_std_dev: 0.1

  implicit_als:
    signal_confidences:
      rating_8_10: 2.0
      watching: 1.0
      rewatching: 1.0
      completed: 0.5
      planned: 0.5
      on_hold: 0.0
      rating_4_5: -1.0
      rating_1_3: -2.0
    factors: 96
    regularization: 0.1
    alpha: 1.0
    iterations: 40
""",
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

    _write_config(
        config_path,
        """
random_seed: 42
unknown_parameter: 100

dataset:
  split:
    validation_fraction: 0.1
    test_fraction: 0.1
    min_interactions_per_user: 10

target:
  positive_rating_threshold: 8

retrievers:
  explicit_svd:
    min_item_explicit_ratings: 5
    n_factors: 64
    n_epochs: 15
    biased: true
    learning_rate: 0.015
    regularization: 0.10
    init_mean: 0.0
    init_std_dev: 0.1

  implicit_als:
    signal_confidences:
      rating_8_10: 2.0
      watching: 1.0
      rewatching: 1.0
      completed: 0.5
      planned: 0.5
      on_hold: 0.0
      rating_4_5: -1.0
      rating_1_3: -2.0
    factors: 96
    regularization: 0.1
    alpha: 1.0
    iterations: 40
""",
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
    ),
    [
        (0, 5),
        (-1, 5),
        (10, 0),
        (10, -1),
    ],
)
def test_load_training_config_rejects_invalid_minimums(
    tmp_path: Path,
    min_interactions_per_user: int,
    min_item_explicit_ratings: int,
):
    config_path = tmp_path / "training.yaml"

    _write_config(
        config_path,
        f"""
random_seed: 42

dataset:
  split:
    validation_fraction: 0.1
    test_fraction: 0.1
    min_interactions_per_user: {min_interactions_per_user}

target:
  positive_rating_threshold: 8

retrievers:
  explicit_svd:
    min_item_explicit_ratings: {min_item_explicit_ratings}
    n_factors: 64
    n_epochs: 15
    biased: true
    learning_rate: 0.015
    regularization: 0.10
    init_mean: 0.0
    init_std_dev: 0.1

  implicit_als:
    signal_confidences:
      rating_8_10: 2.0
      watching: 1.0
      rewatching: 1.0
      completed: 0.5
      planned: 0.5
      on_hold: 0.0
      rating_4_5: -1.0
      rating_1_3: -2.0
    factors: 96
    regularization: 0.1
    alpha: 1.0
    iterations: 40
""",
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

    _write_config(
        config_path,
        f"""
random_seed: 42

dataset:
  split:
    validation_fraction: 0.1
    test_fraction: 0.1
    min_interactions_per_user: 10

target:
  positive_rating_threshold: {positive_rating_threshold}

retrievers:
  explicit_svd:
    min_item_explicit_ratings: 5
    n_factors: 64
    n_epochs: 15
    biased: true
    learning_rate: 0.015
    regularization: 0.10
    init_mean: 0.0
    init_std_dev: 0.1

  implicit_als:
    signal_confidences:
      rating_8_10: 2.0
      watching: 1.0
      rewatching: 1.0
      completed: 0.5
      planned: 0.5
      on_hold: 0.0
      rating_4_5: -1.0
      rating_1_3: -2.0
    factors: 96
    regularization: 0.1
    alpha: 1.0
    iterations: 40
""",
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

    explicit_svd_config: dict[
        str,
        int | float | bool,
    ] = {
        "min_item_explicit_ratings": 5,
        "n_factors": 64,
        "n_epochs": 15,
        "biased": True,
        "learning_rate": 0.015,
        "regularization": 0.10,
        "init_mean": 0.0,
        "init_std_dev": 0.1,
    }

    explicit_svd_config[parameter_name] = invalid_value

    config_content = {
        "random_seed": 42,
        "dataset": {
            "split": {
                "validation_fraction": 0.1,
                "test_fraction": 0.1,
                "min_interactions_per_user": 10,
            }
        },
        "target": {
            "positive_rating_threshold": 8,
        },
        "retrievers": {
            "explicit_svd": explicit_svd_config,
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
        },
    }

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

    signal_confidences: dict[str, float] = {
        "rating_8_10": 2.0,
        "watching": 1.0,
        "rewatching": 1.0,
        "completed": 0.5,
        "planned": 0.5,
        "on_hold": 0.0,
        "rating_4_5": -1.0,
        "rating_1_3": -2.0,
    }

    implicit_als_config: dict[
        str,
        int | float | dict[str, float],
    ] = {
        "signal_confidences": signal_confidences,
        "factors": 96,
        "regularization": 0.1,
        "alpha": 1.0,
        "iterations": 40,
    }

    if parameter_group == "signals":
        signal_confidences[parameter_name] = float(invalid_value)
    else:
        implicit_als_config[parameter_name] = invalid_value

    config_content = {
        "random_seed": 42,
        "dataset": {
            "split": {
                "validation_fraction": 0.1,
                "test_fraction": 0.1,
                "min_interactions_per_user": 10,
            }
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
            "implicit_als": implicit_als_config,
        },
    }

    _write_config(
        config_path,
        yaml.safe_dump(
            config_content,
            sort_keys=False,
        ),
    )

    with pytest.raises(ValidationError):
        load_training_config(config_path)
