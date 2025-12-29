import logging

from injector import inject
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_qdrant import QdrantVectorStore
from mediatr import GenericQuery, Mediator
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_description_chunk import (
    CocktailDescriptionChunk,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailModel
from cezzis_com_cocktails_aisearch.domain.config.hugging_face_options import HuggingFaceOptions
from cezzis_com_cocktails_aisearch.domain.config.qdrant_options import QdrantOptions


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
    def __init__(
        self,
        hugging_face_options: HuggingFaceOptions,
        qdrant_client: QdrantClient,
        qdrant_options: QdrantOptions,
    ):
        self.hugging_face_options = hugging_face_options
        self.qdrant_client = qdrant_client
        self.qdrant_options = qdrant_options
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

        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.qdrant_options.collection_name,
            embedding=HuggingFaceEndpointEmbeddings(
                model=self.hugging_face_options.inference_model,  # http://localhost:8989 | sentence-transformers/all-mpnet-base-v2
                huggingfacehub_api_token=self.hugging_face_options.api_token,
                task="feature-extraction",
            ),
        )

        self.logger.info(
            msg="Deleting existing cocktail embedding vectors from qdrant",
            extra={
                "cocktail_id": command.cocktail_model.id,
            },
        )

        self.qdrant_client.delete(
            collection_name=self.qdrant_options.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="cocktail_id", match=MatchValue(value=command.cocktail_model.id))]
            ),
        )

        chunks_to_embed = [chunk for chunk in command.chunks if chunk.content.strip() != ""]

        self.logger.info(
            msg="Attempting to store cocktail embedding in qdrant",
            extra={
                "cocktail_id": command.cocktail_model.id,
            },
        )

        result = vector_store.add_texts(
            texts=[chunk.content for chunk in chunks_to_embed],
            metadatas=[
                {
                    "cocktail_id": command.cocktail_model.id,
                    "category": chunk.category,
                    "description": chunk.content,
                    "model": command.cocktail_model.model_dump_json(),
                }
                for chunk in chunks_to_embed
            ],
            ids=[chunks_to_embed[i].to_uuid() for i in range(len(chunks_to_embed))],
        )

        if len(result) == 0:
            raise ValueError("No embedding results returned from vector store")

        self.logger.info(
            msg="Cocktail embedding stored in qdrant successfully",
            extra={
                "cocktail_id": command.cocktail_model.id,
            },
        )

        return True
