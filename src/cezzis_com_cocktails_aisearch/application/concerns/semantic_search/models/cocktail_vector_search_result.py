from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CocktailVectorSearchResult(BaseModel):
    """Model representing an individual search result from a cocktail vector search, including the score and associated metadata."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    score: float = Field(..., description="Score of the search result", examples=[0.5158496])
