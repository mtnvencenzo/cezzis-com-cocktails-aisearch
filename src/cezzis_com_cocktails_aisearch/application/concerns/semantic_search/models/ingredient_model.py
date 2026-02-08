from typing import List

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.ingredient_application_type_model import (
    CocktailSearchIngredientApplicationTypeModel,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.ingredient_requirment_type_model import (
    CocktailSearchIngredientRequirementTypeModel,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.ingredient_type_model import (
    CocktailSearchIngredientTypeModel,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.preparation_type_model import (
    CocktailSearchPreparationTypeModel,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.uofm_type_model import (
    CocktailSearchUofMTypeModel,
)


# Ingredient model using the enum
class CocktailSearchIngredientModel(BaseModel):
    """Model representing an ingredient used in a cocktail, including its name, unit of measure, requirement type, preparation, and suggestions."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    name: str = Field(..., description="Name of the ingredient", examples=["Blanco Tequila"])
    uoM: CocktailSearchUofMTypeModel = Field(..., description="Unit of Measure for the ingredient", examples=["ounces"])
    requirement: CocktailSearchIngredientRequirementTypeModel = Field(
        ..., description="Requirement type for the ingredient", examples=["required"]
    )
    display: str = Field(..., description="Display string for the ingredient", examples=["1 1/2 oz Blanco Tequila"])
    units: float = Field(..., description="Quantity of the ingredient", examples=[1.5])
    preparation: CocktailSearchPreparationTypeModel = Field(
        ..., description="Preparation type for the ingredient", examples=["chilled"]
    )
    suggestions: str = Field(
        ...,
        description="Suggestions for the ingredient",
        examples=["Use a good quality Blanco Tequila for the best flavor"],
    )
    types: List[CocktailSearchIngredientTypeModel] = Field(
        ..., description="List of ingredient types", examples=[["spirit"]]
    )
    applications: List[CocktailSearchIngredientApplicationTypeModel] = Field(
        ..., description="List of ingredient applications", examples=[["base"]]
    )
