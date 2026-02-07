from typing import List

from pydantic import BaseModel, Field


class CocktailKeywords(BaseModel):
    """Keyword metadata for Qdrant payload filtering. Not returned in search responses."""

    keywordsBaseSpirit: List[str] = Field(
        default_factory=list, description="Base spirit keywords (e.g. gin, rum, bourbon)"
    )
    keywordsSpiritSubtype: List[str] = Field(
        default_factory=list, description="Spirit subtype keywords (e.g. aged-rum, islay-scotch)"
    )
    keywordsFlavorProfile: List[str] = Field(
        default_factory=list, description="Flavor profile keywords (e.g. bitter, sweet, citrus)"
    )
    keywordsCocktailFamily: List[str] = Field(
        default_factory=list, description="Cocktail family keywords (e.g. sour, tiki, negroni)"
    )
    keywordsTechnique: List[str] = Field(
        default_factory=list, description="Technique keywords (e.g. shaken, stirred, built)"
    )
    keywordsStrength: str = Field(default="", description="Strength keyword (light, medium, strong)")
    keywordsTemperature: str = Field(default="", description="Temperature keyword (cold, frozen, warm)")
    keywordsSeason: List[str] = Field(
        default_factory=list, description="Season keywords (e.g. summer, winter, all-season)"
    )
    keywordsOccasion: List[str] = Field(
        default_factory=list, description="Occasion keywords (e.g. aperitif, party, brunch)"
    )
    keywordsMood: List[str] = Field(
        default_factory=list, description="Mood keywords (e.g. sophisticated, fun, refreshing)"
    )
    keywordsSearchTerms: List[str] = Field(default_factory=list, description="Additional search terms")
