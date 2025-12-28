from enum import Enum


class IngredientRequirementTypeModel(str, Enum):
    none = "None"
    optional = "Optional"
    required = "Required"