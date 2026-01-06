import logging

from injector import inject
from mediatr import GenericQuery, Mediator

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_description_chunk import (
    CocktailDescriptionChunk,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailModel
from cezzis_com_cocktails_aisearch.infrastructure.repositories.icocktail_vector_repository import (
    ICocktailVectorRepository,
)


class CocktailEmbeddingCommand(GenericQuery[bool]):
    @inject
    def __init__(self, chunks: list[CocktailDescriptionChunk], cocktail_model: CocktailModel):
        self.chunks = chunks
        self.cocktail_model = cocktail_model


@Mediator.behavior
class CocktailEmbeddingCommandValidator:
    def handle(self, command: CocktailEmbeddingCommand, next) -> None:
        if not command.cocktail_model or not command.cocktail_model.id:
            raise ValueError("Invalid cocktail model provided for embedding processing")

        if not command.chunks or len(command.chunks) == 0:
            raise ValueError("No chunks provided for embedding processing")

        chunks_to_embed = [chunk for chunk in command.chunks if chunk.content.strip() != ""]

        if not chunks_to_embed or len(chunks_to_embed) == 0:
            raise ValueError("No valid chunks to embed for cocktail, skipping embedding process")

        return next()


@Mediator.handler
class CocktailEmbeddingCommandHandler:
    @inject
    def __init__(self, cocktail_vector_repository: ICocktailVectorRepository):
        self.cocktail_vector_repository = cocktail_vector_repository
        self.logger = logging.getLogger("cocktail_embedding_command_handler")

    async def handle(self, command: CocktailEmbeddingCommand) -> bool:
        assert command.cocktail_model is not None
        assert command.chunks is not None

        self.logger.info(
            msg="Processing cocktail embedding request",
            extra={
                "cocktail_id": command.cocktail_model.id,
            },
        )

        await self.cocktail_vector_repository.delete_vectors(command.cocktail_model.id)

        await self.cocktail_vector_repository.store_vectors(
            cocktail_id=command.cocktail_model.id,
            chunks=[chunk for chunk in command.chunks if chunk.content.strip() != ""],
            cocktail_model=command.cocktail_model,
        )

        self.logger.info(
            msg="Cocktail embedding stored in qdrant successfully",
            extra={
                "cocktail_id": command.cocktail_model.id,
            },
        )

        return True
