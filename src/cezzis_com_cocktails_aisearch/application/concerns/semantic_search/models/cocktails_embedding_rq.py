from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_description_chunk import (
    CocktailDescriptionChunk,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_embedding_model import (
    CocktailEmbeddingModel,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_keywords import (
    CocktailKeywords,
)


class CocktailEmbeddingRq(BaseModel):
    """Request model for embedding cocktail description chunks into the vector database."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    content_chunks: list[CocktailDescriptionChunk] = Field(..., description="List of text chunks to be embedded")
    cocktail_embedding_model: CocktailEmbeddingModel = Field(
        ..., description="Cocktail embedding model associated with the chunks"
    )
    cocktail_keywords: CocktailKeywords = Field(
        default_factory=CocktailKeywords, description="Keyword metadata for Qdrant payload filtering"
    )
