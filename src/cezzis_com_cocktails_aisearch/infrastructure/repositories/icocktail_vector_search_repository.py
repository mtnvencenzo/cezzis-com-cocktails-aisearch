from abc import ABC, abstractmethod

from qdrant_client.http.models import Filter

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailSearchModel


class ICocktailVectorSearchRepository(ABC):
    @abstractmethod
    async def search_vectors(self, free_text: str, query_filter: Filter | None = None) -> list[CocktailSearchModel]:
        pass

    @abstractmethod
    async def get_all_cocktails(self) -> list[CocktailSearchModel]:
        pass
