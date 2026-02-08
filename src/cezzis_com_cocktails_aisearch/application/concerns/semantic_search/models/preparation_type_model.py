from enum import Enum


class CocktailSearchPreparationTypeModel(str, Enum):
    """The preparation types for cocktail ingredients."""

    None_ = "none"
    Chilled = "chilled"
    FreshlySqueezed = "freshlySqueezed"
    PeeledAndJuiced = "peeledAndJuiced"
    FreshlyGrated = "freshlyGrated"
    Quartered = "quartered"
    FreshPressed = "freshPressed"
