from typing import List

from pydantic import BaseModel, ConfigDict, Field

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailModel
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.glassware_type_model import (
    GlasswareTypeModel,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.ingredient_model import IngredientModel


class CocktailEmbeddingModel(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "margarita",
                "title": "Margarita",
                "descriptiveTitle": "The Margarita: A Refreshing Tequila Classic",
                "rating": 5.0,
                "ingredients": [],
                "isIba": True,
                "serves": 1,
                "prepTimeMinutes": 5,
                "searchTiles": ["http://localhost:7179/api/v1/images/traditional-margarita-cocktail-300x300.webp"],
                "glassware": ["rocks", "coupe", "cocktailGlass"],
            }
        }
    )

    id: str = Field(..., description="Unique identifier for the cocktail")
    title: str = Field(..., description="Title of the cocktail")
    descriptiveTitle: str = Field(..., description="Descriptive title of the cocktail")
    rating: float = Field(..., description="Rating of the cocktail")
    ingredients: List[IngredientModel] = Field(..., description="List of ingredients in the cocktail")
    isIba: bool = Field(..., description="Indicates if the cocktail is an IBA official cocktail")
    serves: int = Field(..., description="Number of servings")
    prepTimeMinutes: int = Field(..., description="Preparation time in minutes")
    searchTiles: List[str] = Field(..., description="Search tiles associated with the cocktail")
    glassware: List[GlasswareTypeModel] = Field(..., description="List of glassware types used for the cocktail")

    def to_cocktail_model(self) -> "CocktailModel":
        return CocktailModel(
            id=self.id,
            title=self.title,
            descriptiveTitle=self.descriptiveTitle,
            rating=self.rating,
            ingredients=self.ingredients,
            isIba=self.isIba,
            serves=self.serves,
            prepTimeMinutes=self.prepTimeMinutes,
            searchTiles=self.searchTiles,
            glassware=self.glassware,
        )
