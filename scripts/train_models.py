import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

from shiki_recsys.config.settings import get_settings
from shiki_recsys.config.training import load_training_config
from shiki_recsys.database.repositories.anime_repository import (
    AnimeRepository,
)
from shiki_recsys.database.repositories.user_rate_svd_repository import (
    UserRateSVDRepository,
)
from shiki_recsys.database.session import (
    create_database_engine,
    create_session_factory,
)
from shiki_recsys.preprocessing.catalog import prepare_catalog
from shiki_recsys.preprocessing.interactions import (
    prepare_interactions,
)
from shiki_recsys.training.workflow import run_training_workflow

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse training command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train recommendation models.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the training configuration.",
    )

    return parser.parse_args()


def main() -> None:
    """Train and persist a new recommendation model artifact."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()

    settings = get_settings()
    config = load_training_config(args.config)

    engine = create_database_engine(settings)
    session_factory = create_session_factory(engine)

    anime_repository = AnimeRepository()
    rates_repository = UserRateSVDRepository()

    try:
        logger.info("Loading training data from PostgreSQL.")

        with session_factory() as session:
            interaction_rows = rates_repository.get_all(session)
            catalog_rows = anime_repository.get_all(session)

        interactions = prepare_interactions(interaction_rows)
        catalog = prepare_catalog(catalog_rows)

        logger.info(
            "Loaded %s interactions for %s users and %s anime.",
            len(interactions),
            interactions["user_id"].nunique(),
            len(catalog),
        )

        created_at = datetime.now(UTC)
        artifact_version = created_at.strftime(
            "%Y%m%dT%H%M%SZ",
        )

        logger.info(
            "Starting training run %s.",
            artifact_version,
        )

        result = run_training_workflow(
            interactions=interactions,
            catalog=catalog,
            config=config,
            artifacts_dir=settings.artifacts_dir,
            artifact_version=artifact_version,
            created_at=created_at,
        )

        evaluation = result.evaluation

        logger.info(
            "Evaluation complete: Recall@%s=%.4f, NDCG@%s=%.4f, users=%s.",
            evaluation.ranking_k,
            evaluation.recall_at_k,
            evaluation.ranking_k,
            evaluation.ndcg_at_k,
            evaluation.evaluated_users,
        )

        logger.info(
            "Artifact written to %s.",
            result.artifact_path,
        )
        logger.info("Artifact was not promoted automatically.")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
