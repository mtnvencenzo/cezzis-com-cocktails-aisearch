from abc import ABC, abstractmethod

from qdrant_client.http.models import Filter

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_description_chunk import (
    CocktailDescriptionChunk,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_keywords import (
    CocktailKeywords,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailModel


class ICocktailVectorRepository(ABC):
    @abstractmethod
    async def delete_vectors(self, cocktail_id: str) -> None:
        pass

    @abstractmethod
    async def store_vectors(
        self,
        cocktail_id: str,
        chunks: list[CocktailDescriptionChunk],
        cocktail_model: CocktailModel,
        cocktail_keywords: CocktailKeywords | None = None,
    ) -> None:
        pass

    @abstractmethod
    async def search_vectors(self, free_text: str, query_filter: Filter | None = None) -> list[CocktailModel]:
        pass

    @abstractmethod
    async def get_all_cocktails(self) -> list[CocktailModel]:
        pass
