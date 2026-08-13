from pydantic import BaseModel, ConfigDict, Field


class RecommendationServingConfig(BaseModel):
    """Хранит параметры recommendation serving."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    top_k: int = Field(gt=0)
    min_positive_items: int = Field(gt=0)
