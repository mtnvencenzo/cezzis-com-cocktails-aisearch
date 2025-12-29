from pydantic import BaseModel, Field

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_description_chunk import (
    CocktailDescriptionChunk,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailModel


class CocktailEmbeddingRq(BaseModel):
    """
    Request model for embedding cocktail description chunks into the vector database.
    """

    content_chunks: list[CocktailDescriptionChunk] = Field(..., description="List of text chunks to be embedded")
    cocktail_model: CocktailModel = Field(..., description="Cocktail model associated with the chunks")
