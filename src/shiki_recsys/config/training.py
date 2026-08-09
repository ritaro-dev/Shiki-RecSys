from pathlib import Path

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class SplitConfig(BaseModel):
    """Хранит параметры хронологического разделения данных."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    validation_fraction: float = Field(
        gt=0,
        lt=1,
    )
    test_fraction: float = Field(
        gt=0,
        lt=1,
    )
    min_interactions_per_user: int = Field(
        gt=0,
    )

    @model_validator(mode="after")
    def validate_fraction_sum(self) -> "SplitConfig":
        """
        Проверяет сумму отложенных долей.

        Returns:
            Проверенную конфигурацию разделения.

        Raises:
            ValueError: Если для обучающей части не остаётся данных.
        """

        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError(
                "Сумма validation_fraction и test_fraction должна быть меньше 1."
            )

        return self


class DatasetConfig(BaseModel):
    """Хранит параметры подготовки общего датасета."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    split: SplitConfig


class TargetConfig(BaseModel):
    """Хранит параметры целевой задачи."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    positive_rating_threshold: float = Field(
        gt=0,
        le=10,
        allow_inf_nan=False,
    )


class ExplicitSVDConfig(BaseModel):
    """Хранит параметры explicit SVD retriever."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    min_item_explicit_ratings: int = Field(
        gt=0,
    )
    n_factors: int = Field(
        gt=0,
    )
    n_epochs: int = Field(
        gt=0,
    )
    biased: bool
    learning_rate: float = Field(
        gt=0,
        allow_inf_nan=False,
    )
    regularization: float = Field(
        ge=0,
        allow_inf_nan=False,
    )
    init_mean: float = Field(
        allow_inf_nan=False,
    )
    init_std_dev: float = Field(
        gt=0,
        allow_inf_nan=False,
    )


class ALSSignalConfidencesConfig(BaseModel):
    """Хранит веса пользовательских сигналов для implicit ALS."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    rating_8_10: float = Field(
        gt=0,
        allow_inf_nan=False,
    )
    watching: float = Field(
        ge=0,
        allow_inf_nan=False,
    )
    rewatching: float = Field(
        ge=0,
        allow_inf_nan=False,
    )
    completed: float = Field(
        ge=0,
        allow_inf_nan=False,
    )
    planned: float = Field(
        ge=0,
        allow_inf_nan=False,
    )
    on_hold: float = Field(
        ge=0,
        allow_inf_nan=False,
    )
    rating_4_5: float = Field(
        le=0,
        allow_inf_nan=False,
    )
    rating_1_3: float = Field(
        le=0,
        allow_inf_nan=False,
    )


class ImplicitALSConfig(BaseModel):
    """Хранит параметры implicit ALS retriever."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    signal_confidences: ALSSignalConfidencesConfig
    factors: int = Field(
        gt=0,
    )
    regularization: float = Field(
        ge=0,
        allow_inf_nan=False,
    )
    alpha: float = Field(
        gt=0,
        allow_inf_nan=False,
    )
    iterations: int = Field(
        gt=0,
    )


class ContentTFIDFConfig(BaseModel):
    """Содержит параметры TF-IDF content retriever."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    max_positive_items: int = Field(gt=0)


class RetrieversConfig(BaseModel):
    """Хранит конфигурации retriever-моделей."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    explicit_svd: ExplicitSVDConfig
    implicit_als: ImplicitALSConfig
    content_tfidf: ContentTFIDFConfig


class CandidateGenerationConfig(BaseModel):
    """Хранит параметры генерации кандидатов."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    retrieval_k: int = Field(gt=0)


class TrainingConfig(BaseModel):
    """Хранит полную конфигурацию обучения системы."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    random_seed: int
    dataset: DatasetConfig
    target: TargetConfig
    retrievers: RetrieversConfig
    candidate_generation: CandidateGenerationConfig


def load_training_config(
    path: str | Path,
) -> TrainingConfig:
    """
    Загружает конфигурацию обучения из YAML-файла.

    Args:
        path: Путь к YAML-файлу конфигурации.

    Returns:
        Проверенную типизированную конфигурацию обучения.

    Raises:
        OSError: Если файл невозможно прочитать.
        yaml.YAMLError: Если YAML имеет некорректный синтаксис.
        pydantic.ValidationError: Если структура или значения
            конфигурации некорректны.
    """

    config_path = Path(path)

    with config_path.open(
        mode="r",
        encoding="utf-8",
    ) as config_file:
        raw_config = yaml.safe_load(config_file)

    return TrainingConfig.model_validate(raw_config)
