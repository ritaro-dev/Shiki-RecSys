from dataclasses import dataclass

from shiki_recsys.config.inference import RecommendationServingConfig
from shiki_recsys.inference.model_bundle import ModelBundle
from shiki_recsys.model_artifacts import ArtifactMetadata


@dataclass(frozen=True)
class InferenceState:
    """
    Store long-lived resources required for recommendation inference.

    Attributes:
        bundle: Loaded inference model bundle.
        metadata: Metadata of the loaded model artifact.
        serving_config: Recommendation serving configuration.
    """

    bundle: ModelBundle
    metadata: ArtifactMetadata
    serving_config: RecommendationServingConfig
