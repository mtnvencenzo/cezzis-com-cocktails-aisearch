from typing import List

from pydantic import BaseModel, Field

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_search_statistics import (
    CocktailSearchStatistics,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.glassware_type_model import (
    GlasswareTypeModel,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.ingredient_model import IngredientModel


class CocktailModel(BaseModel):
    id: str = Field(..., description="Unique identifier for the cocktail", examples=["margarita"])
    title: str = Field(..., description="Title of the cocktail", examples=["Margarita"])
    descriptiveTitle: str = Field(
        ..., description="Descriptive title of the cocktail", examples=["The Margarita: A Refreshing Tequila Classic"]
    )
    rating: float = Field(..., description="Rating of the cocktail", examples=[5.0])
    ingredients: List[IngredientModel] = Field(..., description="List of ingredients in the cocktail")
    isIba: bool = Field(..., description="Indicates if the cocktail is an IBA official cocktail", examples=[True])
    serves: int = Field(..., description="Number of servings", examples=[1])
    prepTimeMinutes: int = Field(..., description="Preparation time in minutes", examples=[5])
    searchTiles: List[str] = Field(
        ...,
        description="Search tiles associated with the cocktail",
        examples=[["http://localhost:7179/api/v1/images/traditional-margarita-cocktail-300x300.webp"]],
    )
    glassware: List[GlasswareTypeModel] = Field(
        ..., description="List of glassware types used for the cocktail", examples=[["rocks", "coupe", "cocktailGlass"]]
    )
    search_statistics: CocktailSearchStatistics = Field(
        CocktailSearchStatistics(total_score=0.0, hit_results=[]),
        description="Search statistics for the cocktail",
    )
