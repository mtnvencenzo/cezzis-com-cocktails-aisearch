from pydantic import BaseModel, Field

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_vector_search_result import (
    CocktailVectorSearchResult,
)


class CocktailSearchStatistics(BaseModel):
    total_score: float = Field(..., description="Total score of the search result")
    hit_results: list[CocktailVectorSearchResult] = Field([], description="List of hit results with their scores")
