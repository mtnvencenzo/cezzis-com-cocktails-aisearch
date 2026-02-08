from typing import List

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.ingredient_application_type_model import (
    IngredientApplicationTypeModel,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.ingredient_requirment_type_model import (
    IngredientRequirementTypeModel,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.ingredient_type_model import (
    IngredientTypeModel,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.preparation_type_model import (
    PreparationTypeModel,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.uofm_type_model import UofMTypeModel


# Ingredient model using the enum
class IngredientModel(BaseModel):
    """Model representing an ingredient used in a cocktail, including its name, unit of measure, requirement type, preparation, and suggestions."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    name: str = Field(..., description="Name of the ingredient", examples=["Blanco Tequila"])
    uoM: UofMTypeModel = Field(..., description="Unit of Measure for the ingredient", examples=["ounces"])
    requirement: IngredientRequirementTypeModel = Field(
        ..., description="Requirement type for the ingredient", examples=["required"]
    )
    display: str = Field(..., description="Display string for the ingredient", examples=["1 1/2 oz Blanco Tequila"])
    units: float = Field(..., description="Quantity of the ingredient", examples=[1.5])
    preparation: PreparationTypeModel = Field(
        ..., description="Preparation type for the ingredient", examples=["chilled"]
    )
    suggestions: str = Field(
        ...,
        description="Suggestions for the ingredient",
        examples=["Use a good quality Blanco Tequila for the best flavor"],
    )
    types: List[IngredientTypeModel] = Field(..., description="List of ingredient types", examples=[["spirit"]])
    applications: List[IngredientApplicationTypeModel] = Field(
        ..., description="List of ingredient applications", examples=[["base"]]
    )
