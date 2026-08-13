from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ArtifactInferenceConfig:
    """Хранит inference-параметры конкретной версии моделей."""

    retrieval_k: int
    positive_rating_threshold: float
    max_positive_items: int


@dataclass(frozen=True)
class ArtifactMetadata:
    """Хранит metadata версии model artifacts."""

    artifact_version: str
    created_at: datetime
    inference: ArtifactInferenceConfig
