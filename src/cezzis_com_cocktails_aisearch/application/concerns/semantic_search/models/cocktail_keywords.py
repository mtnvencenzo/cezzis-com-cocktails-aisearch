from typing import List

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CocktailSearchKeywords(BaseModel):
    """Keyword metadata for Qdrant payload filtering. Not returned in search responses."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    keywords_base_spirit: List[str] = Field(
        default_factory=list,
        description="Base spirit keywords (e.g. gin, rum, bourbon)",
        examples=[["gin"], ["rum"], ["bourbon"]],
    )
    keywords_spirit_subtype: List[str] = Field(
        default_factory=list,
        description="Spirit subtype keywords (e.g. aged-rum, islay-scotch)",
        examples=[["aged-rum"], ["islay-scotch"]],
    )
    keywords_flavor_profile: List[str] = Field(
        default_factory=list,
        description="Flavor profile keywords (e.g. bitter, sweet, citrus)",
        examples=[["bitter"], ["sweet"], ["citrus"]],
    )
    keywords_cocktail_family: List[str] = Field(
        default_factory=list,
        description="Cocktail family keywords (e.g. sour, tiki, negroni)",
        examples=[["sour"], ["tiki"], ["negroni"]],
    )
    keywords_technique: List[str] = Field(
        default_factory=list,
        description="Technique keywords (e.g. shaken, stirred, built)",
        examples=[["shaken"], ["stirred"], ["built"]],
    )
    keywords_strength: str = Field(
        default="", description="Strength keyword (light, medium, strong)", examples=["light", "medium", "strong"]
    )
    keywords_temperature: str = Field(
        default="", description="Temperature keyword (cold, frozen, warm)", examples=["cold", "frozen", "warm"]
    )
    keywords_season: List[str] = Field(
        default_factory=list,
        description="Season keywords (e.g. summer, winter, all-season)",
        examples=[["summer"], ["winter"], ["all-season"]],
    )
    keywords_occasion: List[str] = Field(
        default_factory=list,
        description="Occasion keywords (e.g. aperitif, party, brunch)",
        examples=[["aperitif"], ["party"], ["brunch"]],
    )
    keywords_mood: List[str] = Field(
        default_factory=list,
        description="Mood keywords (e.g. sophisticated, fun, refreshing)",
        examples=[["sophisticated"], ["fun"], ["refreshing"]],
    )
    keywords_search_terms: List[str] = Field(
        default_factory=list, description="Additional search terms", examples=[["classic"], ["modern"], ["trending"]]
    )
