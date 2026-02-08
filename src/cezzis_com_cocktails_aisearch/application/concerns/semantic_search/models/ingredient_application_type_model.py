from enum import Enum


class CocktailSearchIngredientApplicationTypeModel(str, Enum):
    """The application types for cocktail ingredients."""

    Base = "base"
    Additional = "additional"
    Garnishment = "garnishment"
    Muddle = "muddle"
