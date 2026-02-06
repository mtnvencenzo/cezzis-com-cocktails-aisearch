import asyncio
import logging

from injector import inject
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_description_chunk import (
    CocktailDescriptionChunk,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailModel
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_search_statistics import (
    CocktailSearchStatistics,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_vector_search_result import (
    CocktailVectorSearchResult,
)
from cezzis_com_cocktails_aisearch.domain.config.hugging_face_options import HuggingFaceOptions
from cezzis_com_cocktails_aisearch.domain.config.qdrant_options import QdrantOptions
from cezzis_com_cocktails_aisearch.infrastructure.repositories.icocktail_vector_repository import (
    ICocktailVectorRepository,
)


class CocktailVectorRepository(ICocktailVectorRepository):
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
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.qdrant_options.collection_name,
            embedding=HuggingFaceEndpointEmbeddings(
                model=self.hugging_face_options.inference_model,  # http://localhost:8989 | sentence-transformers/all-mpnet-base-v2
                huggingfacehub_api_token=self.hugging_face_options.api_token,
                task="feature-extraction",
            ),
        )
        self.logger = logging.getLogger("cocktail_vector_repository")
        self._cocktails_cache: list[CocktailModel] | None = None
        self._cache_lock = asyncio.Lock()

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
        self, cocktail_id: str, chunks: list[CocktailDescriptionChunk], cocktail_model: CocktailModel
    ) -> None:
        self.logger.info(
            msg="Attempting to store cocktail embedding in qdrant",
            extra={
                "cocktail_id": cocktail_id,
            },
        )

        result = self.vector_store.add_texts(
            texts=[chunk.content for chunk in chunks],
            metadatas=[
                {
                    "cocktail_id": cocktail_id,
                    "category": chunk.category,
                    "description": chunk.content,
                    "model": cocktail_model.model_dump_json(),
                }
                for chunk in chunks
            ],
            ids=[chunks[i].to_uuid() for i in range(len(chunks))],
        )

        if len(result) == 0:
            raise ValueError("No embedding results returned from vector store")

    async def search_vectors(self, free_text: str) -> list[CocktailModel]:
        hf_endpoint = HuggingFaceEndpointEmbeddings(
            model=self.hugging_face_options.inference_model,  # http://localhost:8989 | sentence-transformers/all-mpnet-base-v2
            huggingfacehub_api_token=self.hugging_face_options.api_token,
            task="feature-extraction",
        )

        query_vector = await hf_endpoint.aembed_query(free_text or "")

        search_results = self.qdrant_client.query_points(
            collection_name=self.qdrant_options.collection_name,
            limit=self.qdrant_options.semantic_search_limit,
            score_threshold=self.qdrant_options.semantic_search_score_threshold,
            query=query_vector,
            with_payload=True,
        )

        if len(query_vector) == 0:
            raise ValueError("Failed to generate embeddings for the provided text")

        cocktails: list[CocktailModel] = []

        # Sort points by score descending
        sorted_points = sorted(search_results.points, key=lambda p: getattr(p, "score", 0), reverse=True)
        seen_ids = set()

        for point in sorted_points:
            payload = point.payload if hasattr(point, "payload") else None
            if payload:
                metadata = payload.get("metadata")
                if metadata:
                    id = metadata.get("cocktail_id")
                    score = getattr(point, "score", 0)
                    if id and id not in seen_ids:
                        cocktailModel: CocktailModel = CocktailModel.model_validate_json(metadata.get("model"))
                        cocktailModel.search_statistics = CocktailSearchStatistics(
                            total_score=score,
                            max_score=score,
                            avg_score=score,
                            weighted_score=score,
                            hit_count=1,
                            hit_results=[CocktailVectorSearchResult(score=score)],
                        )
                        cocktails.append(cocktailModel)
                        seen_ids.add(id)
                    elif id:
                        # Update existing cocktail's search statistics if duplicate ID found
                        for existing_cocktail in cocktails:
                            if existing_cocktail.id == id:
                                if existing_cocktail.search_statistics is None:
                                    existing_cocktail.search_statistics = CocktailSearchStatistics(
                                        total_score=0.0,
                                        max_score=0.0,
                                        avg_score=0.0,
                                        weighted_score=0.0,
                                        hit_count=0,
                                        hit_results=[],
                                    )
                                stats = existing_cocktail.search_statistics
                                stats.total_score += score
                                stats.max_score = max(stats.max_score, score)
                                stats.hit_count += 1
                                stats.hit_results.append(CocktailVectorSearchResult(score=score))
                                break

        # Calculate final weighted scores for all cocktails
        for cocktail in cocktails:
            if cocktail.search_statistics and cocktail.search_statistics.hit_count > 0:
                stats = cocktail.search_statistics
                stats.avg_score = stats.total_score / stats.hit_count
                # Weighted score: average score boosted by hit count (capped at 5 hits for diminishing returns)
                # This rewards cocktails that match multiple chunks while still prioritizing high-scoring matches
                hit_boost = 1.0 + (0.1 * min(stats.hit_count - 1, 4))  # Up to 40% boost for 5+ hits
                stats.weighted_score = stats.avg_score * hit_boost

        return cocktails

    async def get_all_cocktails(self) -> list[CocktailModel]:
        # Return cached results if available
        if self._cocktails_cache is not None:
            self.logger.debug("Returning cached cocktails")
            return self._cocktails_cache

        # Use lock to prevent concurrent fetches
        async with self._cache_lock:
            # Double-check cache after acquiring lock (another request might have populated it)
            if self._cocktails_cache is not None:
                self.logger.debug("Returning cached cocktails (after lock)")
                return self._cocktails_cache

            self.logger.info(msg="Retrieving all cocktails from qdrant")

            cocktails_dict = {}
            next_offset = None

            while True:
                points, next_offset = self.qdrant_client.scroll(
                    collection_name=self.qdrant_options.collection_name,
                    limit=100,  # Smaller batch size
                    offset=next_offset,
                    with_payload=True,
                )

                for point in points:
                    payload = point.payload if hasattr(point, "payload") else None
                    if payload:
                        metadata = payload.get("metadata")
                        if metadata:
                            id = metadata.get("cocktail_id")
                            if id and id not in cocktails_dict:
                                cocktailModel: CocktailModel = CocktailModel.model_validate_json(metadata.get("model"))
                                cocktails_dict[id] = cocktailModel

                # Break if no more results
                if next_offset is None:
                    break

            # Cache the results
            self._cocktails_cache = list(cocktails_dict.values())
            self.logger.info(f"Cached {len(self._cocktails_cache)} cocktails")

            return self._cocktails_cache
