from enum import Enum


class IngredientTypeModel(str, Enum):
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