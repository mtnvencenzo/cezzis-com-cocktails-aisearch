import logging

from injector import inject
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVector,
)

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_description_chunk import (
    CocktailDescriptionChunk,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_keywords import (
    CocktailSearchKeywords,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailSearchModel
from cezzis_com_cocktails_aisearch.domain.config.hugging_face_options import HuggingFaceOptions
from cezzis_com_cocktails_aisearch.domain.config.qdrant_options import QdrantOptions
from cezzis_com_cocktails_aisearch.infrastructure.repositories.icocktail_vector_embedding_repository import (
    ICocktailVectorEmbeddingRepository,
)
from cezzis_com_cocktails_aisearch.infrastructure.services.isplade_service import ISpladeService


class CocktailVectorEmbeddingRepository(ICocktailVectorEmbeddingRepository):
    @inject
    def __init__(
        self,
        hugging_face_options: HuggingFaceOptions,
        qdrant_client: QdrantClient,
        qdrant_options: QdrantOptions,
        splade_service: ISpladeService,
    ):
        self.hugging_face_options = hugging_face_options
        self.qdrant_client = qdrant_client
        self.qdrant_options = qdrant_options
        self.splade_service = splade_service
        self._embeddings = HuggingFaceEndpointEmbeddings(
            model=self.hugging_face_options.inference_model,
            huggingfacehub_api_token=self.hugging_face_options.api_token,
            task="feature-extraction",
        )
        self.logger = logging.getLogger("cocktail_vector_embedding_repository")

    async def delete_vectors(self, cocktail_id: str) -> None:
        self.logger.info(
            msg="Deleting existing cocktail embedding vectors from qdrant",
            extra={
                "cocktail_id": cocktail_id,
            },
        )

        self.qdrant_client.delete(
            wait=True,
            collection_name=self.qdrant_options.collection_name,
            points_selector=Filter(
                should=[
                    FieldCondition(key="cocktail_id", match=MatchValue(value=cocktail_id)),
                    FieldCondition(key="metadata.cocktail_id", match=MatchValue(value=cocktail_id)),
                ]
            ),
        )

    async def store_vectors(
        self,
        cocktail_id: str,
        chunks: list[CocktailDescriptionChunk],
        cocktail_model: CocktailSearchModel,
        cocktail_keywords: CocktailSearchKeywords | None = None,
    ) -> None:
        self.logger.info(
            msg="Attempting to store cocktail embedding in qdrant",
            extra={
                "cocktail_id": cocktail_id,
            },
        )

        keywords = cocktail_keywords or CocktailSearchKeywords()

        texts = [chunk.content for chunk in chunks]

        # Generate dense embeddings for all chunks
        dense_vectors = await self._embeddings.aembed_documents(texts)
        if not dense_vectors:
            raise ValueError("No dense embedding results returned from embedding model")

        # Generate sparse embeddings for all chunks via SPLADE
        sparse_vectors = await self.splade_service.encode_batch(texts)

        # Build PointStruct list with named vectors (dense + sparse)
        points: list[PointStruct] = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "cocktail_id": cocktail_id,
                "category": chunk.category,
                "description": chunk.content,
                "model": cocktail_model.model_dump_json(),
                "title": cocktail_model.title.lower(),
                "is_iba": cocktail_model.is_iba,
                "serves": cocktail_model.serves,
                "prep_time_minutes": cocktail_model.prep_time_minutes,
                "ingredient_count": len(cocktail_model.ingredients),
                "ingredient_names": [i_item.name.lower() for i_item in cocktail_model.ingredients if i_item.name],
                "ingredient_words": list(
                    {
                        word.lower()
                        for i_item in cocktail_model.ingredients
                        if i_item.name
                        for word in i_item.name.split()
                        if len(word) >= 3
                    }
                ),
                "glassware_values": [g.value for g in cocktail_model.glassware],
                "rating": cocktail_model.rating,
                "keywords_base_spirit": keywords.keywords_base_spirit,
                "keywords_spirit_subtype": keywords.keywords_spirit_subtype,
                "keywords_flavor_profile": keywords.keywords_flavor_profile,
                "keywords_cocktail_family": keywords.keywords_cocktail_family,
                "keywords_technique": keywords.keywords_technique,
                "keywords_strength": keywords.keywords_strength,
                "keywords_temperature": keywords.keywords_temperature,
                "keywords_season": keywords.keywords_season,
                "keywords_occasion": keywords.keywords_occasion,
                "keywords_mood": keywords.keywords_mood,
                "keywords_search_terms": keywords.keywords_search_terms,
            }

            sparse_indices, sparse_values = sparse_vectors[i] if i < len(sparse_vectors) else ([], [])

            named_vectors: dict = {
                "dense": dense_vectors[i],
            }

            if sparse_indices:
                named_vectors["sparse"] = SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                )

            points.append(
                PointStruct(
                    id=chunk.to_uuid(),
                    vector=named_vectors,
                    payload={"metadata": metadata},
                )
            )

        self.qdrant_client.upsert(
            collection_name=self.qdrant_options.collection_name,
            points=points,
            wait=True,
        )

        self.logger.info(
            msg="Stored cocktail vectors with named dense + sparse embeddings",
            extra={
                "cocktail_id": cocktail_id,
                "point_count": len(points),
            },
        )
