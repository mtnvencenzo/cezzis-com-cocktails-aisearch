from enum import Enum


class PreparationTypeModel(str, Enum):
    None_ = "None"
    Chilled = "Chilled"
    FreshlySqueezed = "FreshlySqueezed"
    PeeledAndJuiced = "PeeledAndJuiced"
    FreshlyGrated = "FreshlyGrated"
    Quartered = "Quartered"
    FreshPressed = "FreshPressed"
