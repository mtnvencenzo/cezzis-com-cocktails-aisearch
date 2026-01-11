from pydantic import BaseModel, ConfigDict, Field

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_vector_search_result import (
    CocktailVectorSearchResult,
)


class CocktailSearchStatistics(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_score": 3.72682033,
                "hit_results": [
                    {"score": 0.5158496},
                    {"score": 0.4789201},
                    {"score": 0.4500000},
                ],
            }
        }
    )

    total_score: float = Field(..., description="Total score of the search result")
    hit_results: list[CocktailVectorSearchResult] = Field([], description="List of hit results with their scores")
