from pydantic import BaseModel, Field

from shiki_recsys.inference.user_state import UserState


class RecommendationItemResponse(BaseModel):
    """Represent a single recommended anime."""

    anime_id: int = Field(
        gt=0,
        description="Shikimori anime ID.",
    )
    display_name: str = Field(
        description="Russian anime title when available, otherwise the default title.",
    )
    rank: int = Field(
        gt=0,
        description="Position in the recommendation list.",
    )


class RecommendationsResponse(BaseModel):
    """Represent a recommendation result for a user."""

    user_id: int = Field(
        gt=0,
        description="Shikimori user ID.",
    )
    state: UserState
    recommendations: list[RecommendationItemResponse]
