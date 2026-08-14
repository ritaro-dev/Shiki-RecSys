from pydantic import BaseModel, ConfigDict, Field

from shiki_recsys.config.settings import Settings


class RecommendationServingConfig(BaseModel):
    """Store recommendation serving parameters."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    top_k: int = Field(gt=0)
    min_positive_items: int = Field(gt=0)


def build_recommendation_serving_config(
    settings: Settings,
) -> RecommendationServingConfig:
    """
    Build recommendation serving configuration.

    Args:
        settings: Application settings.

    Returns:
        Validated recommendation serving configuration.
    """
    return RecommendationServingConfig(
        top_k=settings.recommendation_top_k,
        min_positive_items=settings.recommendation_min_positive_items,
    )
