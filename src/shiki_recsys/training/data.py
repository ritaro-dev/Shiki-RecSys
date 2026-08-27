from dataclasses import dataclass

import pandas as pd
from sqlalchemy.orm import Session

from shiki_recsys.database.repositories.anime_repository import AnimeRepository
from shiki_recsys.database.repositories.user_rate_svd_repository import (
    UserRateSVDRepository,
)
from shiki_recsys.preprocessing.catalog import prepare_catalog
from shiki_recsys.preprocessing.interactions import prepare_interactions


@dataclass(frozen=True)
class TrainingData:
    """Store prepared data used for model training and experiments."""

    interactions: pd.DataFrame
    catalog: pd.DataFrame


def load_training_data(session: Session) -> TrainingData:
    """
    Load and prepare model training data from the database.

    Args:
        session: Active database session.

    Returns:
        Prepared interactions and anime catalog.
    """
    interaction_rows = UserRateSVDRepository().get_all(session)
    catalog_rows = AnimeRepository().get_all(session)

    return TrainingData(
        interactions=prepare_interactions(interaction_rows),
        catalog=prepare_catalog(catalog_rows),
    )
