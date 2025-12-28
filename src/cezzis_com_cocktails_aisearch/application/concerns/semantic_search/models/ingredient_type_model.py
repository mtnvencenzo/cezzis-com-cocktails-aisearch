from enum import Enum


class IngredientTypeModel(str, Enum):
    Herb = "Herb"
    Fruit = "Fruit"
    Juice = "Juice"
    Bitters = "Bitters"
    Syrup = "Syrup"
    Protein = "Protein"
    Flowers = "Flowers"
    Sauce = "Sauce"
    Vegetable = "Vegetable"
    Dilution = "Dilution"
    Beer = "Beer"
    Spirit = "Spirit"
    Liqueur = "Liqueur"
    Wine = "Wine"
    Champagne = "Champagne"
