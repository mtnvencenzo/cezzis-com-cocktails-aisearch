from uuid import NAMESPACE_DNS, uuid5

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CocktailDescriptionChunk(BaseModel):
    """Model representing a chunk of cocktail description text, categorized for embedding and vector search."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    category: str = Field(
        ...,
        description="Category of the description chunk",
        examples=["desc", "ingredients", "preparation"],
    )
    content: str = Field(
        ...,
        description="Textual content of the description chunk",
        examples=[
            "A refreshing cocktail made with rum, lime juice, and mint leaves.",
            "Ingredients: 2 oz white rum, 1 oz lime juice, 2 tsp sugar, mint leaves, soda water.",
            "Preparation: Muddle mint leaves with sugar and lime juice. Add rum and ice, then top with soda water. Garnish with a sprig of mint.",
        ],
    )

    def to_uuid(self) -> str:
        return str(uuid5(NAMESPACE_DNS, f"{self.category}-{self.content}"))
