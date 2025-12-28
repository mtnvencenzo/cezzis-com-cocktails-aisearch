from typing import List
from pydantic import BaseModel

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.glassware_type_model import GlasswareTypeModel
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.ingredient_model import IngredientModel

class CocktailModel(BaseModel):
    id: str
    title: str
    descriptiveTitle: str
    rating: int
    ingredients: List[IngredientModel]
    isIba: bool
    serves: int
    prepTimeMinutes: int
    searchTiles: List[str]
    glassware: List[GlasswareTypeModel]