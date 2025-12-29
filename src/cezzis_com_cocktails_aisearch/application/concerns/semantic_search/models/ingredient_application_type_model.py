from enum import Enum


class IngredientApplicationTypeModel(str, Enum):
    Base = "Base"
    Additional = "Additional"
    Garnishment = "Garnishment"
    Muddle = "Muddle"
