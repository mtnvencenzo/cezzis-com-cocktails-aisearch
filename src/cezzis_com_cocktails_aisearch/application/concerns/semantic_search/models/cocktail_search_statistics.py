from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_vector_search_result import (
    CocktailVectorSearchResult,
)


class CocktailSearchStatistics(BaseModel):
    """Model representing the search statistics for a cocktail search operation, including score metrics and hit details."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    total_score: float = Field(..., description="Sum of all hit scores", examples=[3.72682033])
    max_score: float = Field(0.0, description="Highest individual chunk score", examples=[0.5158496])
    avg_score: float = Field(0.0, description="Average score across all hits", examples=[0.4815899])
    weighted_score: float = Field(
        0.0, description="Weighted score combining avg with hit count boost", examples=[0.6260669]
    )
    hit_count: int = Field(0, description="Number of matching chunks", examples=[3])
    hit_results: list[CocktailVectorSearchResult] = Field(
        [],
        description="List of hit results with their scores",
        examples=[
            {"score": 0.5158496},
            {"score": 0.4789201},
            {"score": 0.4500000},
        ],
    )
