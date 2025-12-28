from typing import List

from pydantic import BaseModel

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
    name: str
    uoM: UofMTypeModel
    requirement: List[IngredientRequirementTypeModel]
    display: str
    units: float
    preparation: PreparationTypeModel
    suggestions: str
    types: List[IngredientTypeModel]
    applications: List[str]
