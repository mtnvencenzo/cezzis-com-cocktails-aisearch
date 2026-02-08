from enum import Enum


class IngredientApplicationTypeModel(str, Enum):
    """The application types for cocktail ingredients."""

    Base = "base"
    Additional = "additional"
    Garnishment = "garnishment"
    Muddle = "muddle"
