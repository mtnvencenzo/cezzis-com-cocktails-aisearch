from typing import List

from pydantic import BaseModel, Field

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailModel


class CocktailsSearchRs(BaseModel):
    items: List[CocktailModel] = Field(..., description="List of cocktails returned from the search")
