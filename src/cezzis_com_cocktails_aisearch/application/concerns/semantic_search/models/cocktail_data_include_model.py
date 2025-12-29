from enum import Enum


class CocktailDataIncludeModel(str, Enum):
    mainImages = "mainImages"
    searchTiles = "searchTiles"
    descriptiveTitle = "descriptiveTitle"
