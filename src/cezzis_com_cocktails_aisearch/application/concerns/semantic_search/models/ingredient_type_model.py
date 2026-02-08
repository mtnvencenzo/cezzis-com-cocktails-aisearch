from enum import Enum


class CocktailSearchIngredientTypeModel(str, Enum):
    """The types of ingredients used in cocktails."""

    Herb = "herb"
    Fruit = "fruit"
    Juice = "juice"
    Bitters = "bitters"
    Syrup = "syrup"
    Protein = "protein"
    Flowers = "flowers"
    Sauce = "sauce"
    Vegetable = "vegetable"
    Dilution = "dilution"
    Beer = "beer"
    Spirit = "spirit"
    Liqueur = "liqueur"
    Wine = "wine"
    Champagne = "champagne"
