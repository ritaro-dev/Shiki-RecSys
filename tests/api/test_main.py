from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import shiki_recsys.api.main as main_module


class FakeEngine:
    def __init__(self):
        self.disposed = False

    def dispose(self):
        self.disposed = True


class FakeShikimoriClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_lifespan_initializes_inference_state(monkeypatch):
    settings = SimpleNamespace(
        artifacts_dir=Path("artifacts"),
        shikimori_graphql_url="https://example.com/graphql",
        shikimori_user_agent="test-agent",
        shikimori_min_interval_seconds=0.8,
        shikimori_timeout_seconds=15.0,
        shikimori_max_retries=5,
        recommendation_top_k=20,
        recommendation_min_positive_items=5,
    )
    engine = FakeEngine()
    session_factory = object()
    bundle = object()
    metadata = object()
    shikimori_client = FakeShikimoriClient()

    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: settings,
    )
    monkeypatch.setattr(
        main_module,
        "create_database_engine",
        lambda settings: engine,
    )
    monkeypatch.setattr(
        main_module,
        "create_session_factory",
        lambda engine: session_factory,
    )
    monkeypatch.setattr(
        main_module,
        "load_current_model_artifacts",
        lambda *, artifacts_dir: (bundle, metadata),
    )
    monkeypatch.setattr(
        main_module,
        "ShikimoriClient",
        lambda **kwargs: shikimori_client,
    )

    app = main_module.create_app()

    with TestClient(app):
        assert app.state.engine is engine
        assert app.state.session_factory is session_factory
        assert app.state.shikimori_client is shikimori_client
        assert app.state.inference.bundle is bundle
        assert app.state.inference.metadata is metadata

        assert engine.disposed is False
        assert shikimori_client.closed is False

        assert app.state.inference.serving_config.top_k == 20
        assert app.state.inference.serving_config.min_positive_items == 5

    assert engine.disposed is True
    assert shikimori_client.closed is True
