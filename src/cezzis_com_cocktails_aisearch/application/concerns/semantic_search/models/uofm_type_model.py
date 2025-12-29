from enum import Enum


class UofMTypeModel(str, Enum):
    """The unit of measurment types for cocktail ingredients."""

    None_ = "None"
    Ounces = "Ounces"
    Dashes = "Dashes"
    Tablespoon = "Tablespoon"
    Topoff = "Topoff"
    Item = "Item"
    Teaspoon = "Teaspoon"
    ToTaste = "ToTaste"
    Barspoon = "Barspoon"
    Cups = "Cups"
    Splash = "Splash"
    Discretion = "Discretion"
