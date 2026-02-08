from typing import List

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailModel


class CocktailsSearchRs(BaseModel):
    """Model representing the response structure for a cocktail search operation, containing a list of matching cocktails."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    items: List[CocktailModel] = Field(..., description="List of cocktails returned from the search")
