from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from shiki_recsys.api.exception_handlers import (
    register_exception_handlers,
)
from shiki_recsys.api.routers.health import (
    router as health_router,
)
from shiki_recsys.api.routers.users import (
    router as users_router,
)
from shiki_recsys.config.inference import (
    build_recommendation_serving_config,
)
from shiki_recsys.config.settings import get_settings
from shiki_recsys.database.session import (
    create_database_engine,
    create_session_factory,
)
from shiki_recsys.inference.artifact_loader import (
    load_current_model_artifacts,
)
from shiki_recsys.inference.runtime import InferenceState
from shiki_recsys.integrations.shikimori.client import (
    ShikimoriClient,
)
from shiki_recsys.integrations.shikimori.rate_limiter import (
    ShikimoriRateLimiter,
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncGenerator[None, None]:
    """
    Manage application-wide resources.

    Args:
        app: FastAPI application instance.

    Yields:
        Control while application resources are available.
    """

    settings = get_settings()
    serving_config = build_recommendation_serving_config(settings)

    engine = create_database_engine(settings)
    session_factory = create_session_factory(engine)

    bundle, metadata = load_current_model_artifacts(
        artifacts_dir=settings.artifacts_dir,
    )

    inference = InferenceState(
        bundle=bundle,
        metadata=metadata,
        serving_config=serving_config,
    )

    limiter = ShikimoriRateLimiter(
        min_interval_seconds=(settings.shikimori_min_interval_seconds),
    )

    shikimori_client = ShikimoriClient(
        graphql_url=settings.shikimori_graphql_url,
        user_agent=settings.shikimori_user_agent,
        limiter=limiter,
        timeout_seconds=(settings.shikimori_timeout_seconds),
        max_retries=settings.shikimori_max_retries,
    )

    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.shikimori_client = shikimori_client
    app.state.inference = inference

    try:
        yield
    finally:
        shikimori_client.close()
        engine.dispose()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application.
    """

    app = FastAPI(
        title="Shiki Recsys API",
        version="0.1.0",
        lifespan=lifespan,
    )

    register_exception_handlers(app)

    app.include_router(health_router)
    app.include_router(users_router)

    return app


app = create_app()
