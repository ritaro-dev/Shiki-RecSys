from unittest.mock import MagicMock, patch

import pandas as pd

from shiki_recsys.training.data import TrainingData, load_training_data


def test_load_training_data_loads_and_prepares_database_rows() -> None:
    session = MagicMock()

    interaction_rows = [{"user_id": 1}]
    catalog_rows = [{"id": 10}]

    interactions = pd.DataFrame(
        {
            "user_id": [1],
            "anime_id": [10],
        }
    )
    catalog = pd.DataFrame(
        {
            "anime_id": [10],
        }
    )

    with (
        patch(
            "shiki_recsys.training.data.UserRateSVDRepository"
        ) as rates_repository_class,
        patch("shiki_recsys.training.data.AnimeRepository") as anime_repository_class,
        patch(
            "shiki_recsys.training.data.prepare_interactions",
            return_value=interactions,
        ) as prepare_interactions,
        patch(
            "shiki_recsys.training.data.prepare_catalog",
            return_value=catalog,
        ) as prepare_catalog,
    ):
        rates_repository_class.return_value.get_all.return_value = interaction_rows
        anime_repository_class.return_value.get_all.return_value = catalog_rows

        result = load_training_data(session)

    assert result == TrainingData(
        interactions=interactions,
        catalog=catalog,
    )

    rates_repository_class.return_value.get_all.assert_called_once_with(session)
    anime_repository_class.return_value.get_all.assert_called_once_with(session)
    prepare_interactions.assert_called_once_with(interaction_rows)
    prepare_catalog.assert_called_once_with(catalog_rows)
