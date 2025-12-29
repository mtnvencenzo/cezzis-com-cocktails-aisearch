from enum import Enum


class PreparationTypeModel(str, Enum):
    None_ = "none"
    Chilled = "chilled"
    FreshlySqueezed = "freshlySqueezed"
    PeeledAndJuiced = "peeledAndJuiced"
    FreshlyGrated = "freshlyGrated"
    Quartered = "quartered"
    FreshPressed = "freshPressed"
