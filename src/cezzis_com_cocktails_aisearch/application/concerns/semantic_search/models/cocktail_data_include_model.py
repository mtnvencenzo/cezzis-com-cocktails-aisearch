from enum import Enum


class CocktailDataIncludeModel(str, Enum):
    """The types of cocktail data to include in the search results."""

    mainImages = "mainImages"
    searchTiles = "searchTiles"
    descriptiveTitle = "descriptiveTitle"
