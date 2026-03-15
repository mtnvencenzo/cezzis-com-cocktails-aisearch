import asyncio
import logging
import math
from collections import OrderedDict

from injector import inject
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter,
    Fusion,
    FusionQuery,
    Prefetch,
    QueryResponse,
    SparseVector,
)

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailSearchModel
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_search_statistics import (
    CocktailSearchStatistics,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_vector_search_result import (
    CocktailVectorSearchResult,
)
from cezzis_com_cocktails_aisearch.domain.config.hugging_face_options import HuggingFaceOptions
from cezzis_com_cocktails_aisearch.domain.config.qdrant_options import QdrantOptions
from cezzis_com_cocktails_aisearch.infrastructure.repositories.icocktail_vector_search_repository import (
    ICocktailVectorSearchRepository,
)
from cezzis_com_cocktails_aisearch.infrastructure.services.isplade_service import ISpladeService


class CocktailVectorSearchRepository(ICocktailVectorSearchRepository):
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
        self.logger = logging.getLogger("cocktail_vector_search_repository")
        self._cocktails_cache: list[CocktailSearchModel] | None = None
        self._cache_lock = asyncio.Lock()
        self._embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._embedding_cache_max_size: int = 1024

    async def _get_cached_embedding(self, text: str) -> list[float]:
        """Get embedding from cache or generate and cache it."""
        cache_key = text.strip().lower()
        if cache_key in self._embedding_cache:
            # Move to end (most recently used)
            self._embedding_cache.move_to_end(cache_key)
            self.logger.debug(f"Embedding cache hit for: {cache_key[:50]}")
            return self._embedding_cache[cache_key]

        embedding = await self._embeddings.aembed_query(text)

        # Evict oldest entry if cache is full
        if len(self._embedding_cache) >= self._embedding_cache_max_size:
            self._embedding_cache.popitem(last=False)

        self._embedding_cache[cache_key] = embedding
        return embedding

    async def search_vectors(self, free_text: str, query_filter: Filter | None = None) -> list[CocktailSearchModel]:
        query_vector = await self._get_cached_embedding(free_text or "")

        if len(query_vector) == 0:
            raise ValueError("Failed to generate embeddings for the provided text")

        # Always use hybrid search (dense + sparse via RRF)
        search_results = await self._hybrid_search(free_text or "", query_vector, query_filter)

        cocktails: list[CocktailSearchModel] = []

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
                        cocktailModel: CocktailSearchModel = CocktailSearchModel.model_validate_json(
                            metadata.get("model")
                        )
                        cocktailModel.search_statistics = CocktailSearchStatistics(
                            total_score=score,
                            max_score=score,
                            avg_score=score,
                            weighted_score=score,
                            reranker_score=0.0,
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
                                        reranker_score=0.0,
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
        self._calculate_weighted_scores(cocktails)

        return cocktails

    def _dense_only_search(self, query_vector: list[float], query_filter: Filter | None) -> QueryResponse:
        """Perform dense-only vector search using Qdrant query_points."""
        return self.qdrant_client.query_points(
            collection_name=self.qdrant_options.collection_name,
            limit=self.qdrant_options.semantic_search_limit,
            score_threshold=self.qdrant_options.semantic_search_score_threshold,
            query=query_vector,
            using="dense",
            query_filter=query_filter,
            with_payload=True,
        )

    async def _hybrid_search(
        self, free_text: str, query_vector: list[float], query_filter: Filter | None
    ) -> QueryResponse:
        """Perform hybrid search combining dense + sparse vectors via RRF fusion.

        Uses Qdrant's prefetch mechanism to run dense and sparse searches in
        parallel, then fuses results using Reciprocal Rank Fusion (RRF).
        Falls back to dense-only search if SPLADE encoding fails.
        """
        try:
            sparse_indices, sparse_values = await self.splade_service.encode(free_text)
        except Exception:
            self.logger.warning("SPLADE encoding failed during hybrid search, falling back to dense-only")
            return self._dense_only_search(query_vector, query_filter)

        # If SPLADE returned empty results, fall back to dense-only
        if not sparse_indices:
            self.logger.debug("SPLADE returned empty sparse vector, falling back to dense-only")
            return self._dense_only_search(query_vector, query_filter)

        prefetch_limit = self.qdrant_options.semantic_search_prefetch_limit

        return self.qdrant_client.query_points(
            collection_name=self.qdrant_options.collection_name,
            prefetch=[
                Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                    using="sparse",
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=self.qdrant_options.semantic_search_limit,
            with_payload=True,
        )

    @staticmethod
    def _calculate_weighted_scores(cocktails: list[CocktailSearchModel]) -> None:
        """Calculate final weighted scores for all cocktails.

        Formula: max_score * 0.6 + avg_score * 0.3 + log(hit_count + 1) * 0.1
        """
        for cocktail in cocktails:
            if cocktail.search_statistics and cocktail.search_statistics.hit_count > 0:
                stats = cocktail.search_statistics
                stats.avg_score = stats.total_score / stats.hit_count
                # Improved weighted scoring formula:
                # - max_score (60%): strongest single-chunk match is the primary signal
                # - avg_score (30%): rewards consistent relevance across chunks
                # - log(hit_count) (10%): diminishing returns for breadth of matching
                # This prevents a cocktail with 5 low-scoring chunk hits (e.g., 0.3 each)
                # from outranking one with 2 high-scoring hits (e.g., 0.8 each)
                hit_breadth = math.log(stats.hit_count + 1)  # log(2)=0.69, log(6)=1.79
                stats.weighted_score = stats.max_score * 0.6 + stats.avg_score * 0.3 + hit_breadth * 0.1

    async def get_all_cocktails(self) -> list[CocktailSearchModel]:
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
                                cocktailModel: CocktailSearchModel = CocktailSearchModel.model_validate_json(
                                    metadata.get("model")
                                )
                                cocktails_dict[id] = cocktailModel

                # Break if no more results
                if next_offset is None:
                    break

            # Cache the results only if non-empty, so a temporarily empty
            # collection doesn't permanently poison the cache
            cocktails_list = list(cocktails_dict.values())
            if cocktails_list:
                self._cocktails_cache = cocktails_list
                self.logger.info(f"Cached {len(cocktails_list)} cocktails")
            else:
                self.logger.warning("No cocktails found to cache")

            return cocktails_list
