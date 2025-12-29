from enum import Enum


class IngredientApplicationModel(str, Enum):
    Base = "Base"
    Additional = "Additional"
    Garnishment = "Garnishment"
    Muddle = "Muddle"
