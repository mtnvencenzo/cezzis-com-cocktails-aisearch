from typing import List

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailModel
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.glassware_type_model import (
    GlasswareTypeModel,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.ingredient_model import IngredientModel


class CocktailEmbeddingModel(BaseModel):
    """Model representing the cocktail embedding data structure used for vector search and storage."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    id: str = Field(
        ...,
        description="Unique identifier for the cocktail",
        examples=["old-fashioned"],
    )
    title: str = Field(..., description="Title of the cocktail", examples=["Old Fashioned"])
    descriptive_title: str = Field(
        ...,
        description="Descriptive title of the cocktail",
        examples=["The Old Fashioned: A Timeless Whiskey Cocktail"],
    )
    rating: float = Field(..., description="Rating of the cocktail", examples=[4.5])
    ingredients: List[IngredientModel] = Field(..., description="List of ingredients in the cocktail")
    is_iba: bool = Field(..., description="Indicates if the cocktail is an IBA official cocktail", examples=[True])
    serves: int = Field(..., description="Number of servings", examples=[1])
    prep_time_minutes: int = Field(..., description="Preparation time in minutes", examples=[5])
    search_tiles: List[str] = Field(
        ...,
        description="Search tiles associated with the cocktail",
        examples=["http://localhost:7179/api/v1/images/old-fashioned-cocktail-300x300.webp"],
    )
    glassware: List[GlasswareTypeModel] = Field(
        ..., description="List of glassware types used for the cocktail", examples=["rocks", "coupe", "cocktailGlass"]
    )

    def to_cocktail_model(self) -> "CocktailModel":
        return CocktailModel(
            id=self.id,
            title=self.title,
            descriptive_title=self.descriptive_title,
            rating=self.rating,
            ingredients=self.ingredients,
            is_iba=self.is_iba,
            serves=self.serves,
            prep_time_minutes=self.prep_time_minutes,
            search_tiles=self.search_tiles,
            glassware=self.glassware,
        )
