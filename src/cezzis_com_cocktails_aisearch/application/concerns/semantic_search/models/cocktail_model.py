from typing import List

from pydantic import BaseModel, Field

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.glassware_type_model import (
    GlasswareTypeModel,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.ingredient_model import IngredientModel


class CocktailModel(BaseModel):
    id: str = Field(..., description="Unique identifier for the cocktail")
    title: str = Field(..., description="Title of the cocktail")
    descriptiveTitle: str = Field(..., description="Descriptive title of the cocktail")
    rating: int = Field(..., description="Rating of the cocktail")
    ingredients: List[IngredientModel] = Field(..., description="List of ingredients in the cocktail")
    isIba: bool = Field(..., description="Indicates if the cocktail is an IBA official cocktail")
    serves: int = Field(..., description="Number of servings")
    prepTimeMinutes: int = Field(..., description="Preparation time in minutes")
    searchTiles: List[str] = Field(..., description="Search tiles associated with the cocktail")
    glassware: List[GlasswareTypeModel] = Field(..., description="List of glassware types used for the cocktail")
