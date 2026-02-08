from abc import ABC, abstractmethod

from qdrant_client.http.models import Filter

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_description_chunk import (
    CocktailDescriptionChunk,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_keywords import (
    CocktailSearchKeywords,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailSearchModel


class ICocktailVectorRepository(ABC):
    @abstractmethod
    async def delete_vectors(self, cocktail_id: str) -> None:
        pass

    @abstractmethod
    async def store_vectors(
        self,
        cocktail_id: str,
        chunks: list[CocktailDescriptionChunk],
        cocktail_model: CocktailSearchModel,
        cocktail_keywords: CocktailSearchKeywords | None = None,
    ) -> None:
        pass

    @abstractmethod
    async def search_vectors(self, free_text: str, query_filter: Filter | None = None) -> list[CocktailSearchModel]:
        pass

    @abstractmethod
    async def get_all_cocktails(self) -> list[CocktailSearchModel]:
        pass
