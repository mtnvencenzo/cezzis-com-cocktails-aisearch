from enum import Enum


class IngredientRequirementTypeModel(str, Enum):
    """The requirement types for cocktail ingredients."""

    none = "none"
    optional = "optional"
    required = "required"
