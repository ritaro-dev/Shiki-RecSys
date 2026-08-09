from dataclasses import dataclass

from shiki_recsys.ranking.catboost import CatBoostRankerModel
from shiki_recsys.retrievers.content_tfidf import ContentTFIDFRetriever
from shiki_recsys.retrievers.explicit_svd import ExplicitSVDRetriever
from shiki_recsys.retrievers.implicit_als import ImplicitALSRetriever
from shiki_recsys.retrievers.popularity import PopularityRetriever


@dataclass(frozen=True)
class ModelBundle:
    """Хранит модели одного inference artifact."""

    popularity: PopularityRetriever
    explicit_svd: ExplicitSVDRetriever
    implicit_als: ImplicitALSRetriever
    content_tfidf: ContentTFIDFRetriever
    ranker: CatBoostRankerModel

    def supports_personal_user(self, user_id: int) -> bool:
        """
        Проверяет поддержку пользователя персональными retriever-ами.

        Args:
            user_id: Идентификатор пользователя.

        Returns:
            True, если пользователя поддерживает хотя бы один retriever.
        """
        return (
            self.explicit_svd.supports_user(user_id)
            or self.implicit_als.supports_user(user_id)
            or self.content_tfidf.supports_user(user_id)
        )
