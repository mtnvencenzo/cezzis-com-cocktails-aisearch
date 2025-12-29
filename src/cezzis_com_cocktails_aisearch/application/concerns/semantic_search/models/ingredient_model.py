from typing import List

from pydantic import BaseModel, Field

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
    name: str = Field(..., description="Name of the ingredient")
    uoM: UofMTypeModel = Field(..., description="Unit of Measure for the ingredient")
    requirement: IngredientRequirementTypeModel = Field(..., description="Requirement type for the ingredient")
    display: str = Field(..., description="Display string for the ingredient")
    units: float = Field(..., description="Quantity of the ingredient")
    preparation: PreparationTypeModel = Field(..., description="Preparation type for the ingredient")
    suggestions: str = Field(..., description="Suggestions for the ingredient")
    types: List[IngredientTypeModel] = Field(..., description="List of ingredient types")
    applications: List[IngredientApplicationTypeModel] = Field(..., description="List of ingredient applications")
