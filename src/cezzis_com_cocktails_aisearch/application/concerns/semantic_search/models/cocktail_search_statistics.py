from pydantic import BaseModel, ConfigDict, Field

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_vector_search_result import (
    CocktailVectorSearchResult,
)


class CocktailSearchStatistics(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_score": 3.72682033,
                "max_score": 0.5158496,
                "avg_score": 0.4815899,
                "weighted_score": 0.6260669,
                "hit_count": 3,
                "hit_results": [
                    {"score": 0.5158496},
                    {"score": 0.4789201},
                    {"score": 0.4500000},
                ],
            }
        }
    )

    total_score: float = Field(..., description="Sum of all hit scores")
    max_score: float = Field(0.0, description="Highest individual chunk score")
    avg_score: float = Field(0.0, description="Average score across all hits")
    weighted_score: float = Field(0.0, description="Weighted score combining avg with hit count boost")
    hit_count: int = Field(0, description="Number of matching chunks")
    hit_results: list[CocktailVectorSearchResult] = Field([], description="List of hit results with their scores")
