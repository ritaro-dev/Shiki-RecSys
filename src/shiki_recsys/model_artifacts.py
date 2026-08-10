from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ArtifactMetadata:
    """Хранит metadata версии model artifacts."""

    artifact_version: str
    created_at: datetime
