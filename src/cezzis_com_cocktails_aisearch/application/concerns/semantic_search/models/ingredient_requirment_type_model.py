from enum import Enum


class IngredientRequirementTypeModel(str, Enum):
    none = "none"
    optional = "optional"
    required = "required"