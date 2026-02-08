from enum import Enum


class CocktailSearchUofMTypeModel(str, Enum):
    """The unit of measurment types for cocktail ingredients."""

    None_ = "none"
    Ounces = "ounces"
    Dashes = "dashes"
    Tablespoon = "tablespoon"
    Topoff = "topoff"
    Item = "item"
    Teaspoon = "teaspoon"
    ToTaste = "toTaste"
    Barspoon = "barspoon"
    Cups = "cups"
    Splash = "splash"
    Discretion = "discretion"
