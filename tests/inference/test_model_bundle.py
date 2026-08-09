from unittest.mock import Mock

import pytest

from shiki_recsys.inference.model_bundle import ModelBundle
from shiki_recsys.ranking.catboost import CatBoostRankerModel
from shiki_recsys.retrievers.content_tfidf import ContentTFIDFRetriever
from shiki_recsys.retrievers.explicit_svd import ExplicitSVDRetriever
from shiki_recsys.retrievers.implicit_als import ImplicitALSRetriever
from shiki_recsys.retrievers.popularity import PopularityRetriever


@pytest.mark.parametrize(
    ("svd_support", "als_support", "content_support", "expected"),
    [
        (False, False, False, False),
        (True, False, False, True),
        (False, True, False, True),
        (False, False, True, True),
    ],
)
def test_model_bundle_checks_personal_user_support(
    svd_support: bool,
    als_support: bool,
    content_support: bool,
    expected: bool,
) -> None:
    """Проверяет поддержку пользователя personal retriever-ами."""
    explicit_svd = Mock(spec=ExplicitSVDRetriever)
    explicit_svd.supports_user.return_value = svd_support

    implicit_als = Mock(spec=ImplicitALSRetriever)
    implicit_als.supports_user.return_value = als_support

    content_tfidf = Mock(spec=ContentTFIDFRetriever)
    content_tfidf.supports_user.return_value = content_support

    bundle = ModelBundle(
        popularity=Mock(spec=PopularityRetriever),
        explicit_svd=explicit_svd,
        implicit_als=implicit_als,
        content_tfidf=content_tfidf,
        ranker=Mock(spec=CatBoostRankerModel),
    )

    assert bundle.supports_personal_user(123) is expected
