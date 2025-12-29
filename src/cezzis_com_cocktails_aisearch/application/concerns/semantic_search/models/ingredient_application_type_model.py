from enum import Enum


class IngredientApplicationTypeModel(str, Enum):
    Base = "base"
    Additional = "additional"
    Garnishment = "garnishment"
    Muddle = "muddle"