from typing import List, Optional
from pydantic import BaseModel

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailModel

class CocktailsRs(BaseModel):
    items: List[CocktailModel]