from abc import ABC, abstractmethod

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_description_chunk import (
    CocktailDescriptionChunk,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_keywords import (
    CocktailSearchKeywords,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailSearchModel


class ICocktailVectorEmbeddingRepository(ABC):
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
