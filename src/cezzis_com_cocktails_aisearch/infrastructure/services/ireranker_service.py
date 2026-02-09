from abc import ABC, abstractmethod

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailSearchModel


class IRerankerService(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        cocktails: list[CocktailSearchModel],
        top_k: int = 10,
    ) -> list[CocktailSearchModel]:
        """Rerank cocktail search results using a cross-encoder model via TEI.

        Takes the original query and a list of candidate cocktails, sends them
        to the TEI /rerank endpoint, and returns cocktails reordered by
        cross-encoder relevance score.

        Args:
            query: The original search query text.
            cocktails: Candidate cocktails from the initial vector search.
            top_k: Maximum number of results to return after reranking.

        Returns:
            Reordered list of cocktails with updated reranker scores.
        """
        pass
