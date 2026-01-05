from pydantic import BaseModel, Field


class CocktailVectorSearchResult(BaseModel):
    score: float = Field(..., description="Score of the search result")
