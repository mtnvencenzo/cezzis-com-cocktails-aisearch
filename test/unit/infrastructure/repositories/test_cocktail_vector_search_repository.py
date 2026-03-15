import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_search_repository import (
    CocktailVectorSearchRepository,
)


class TestCocktailVectorSearchRepository:
    """Test cases for CocktailVectorSearchRepository."""

    def _make_splade_service(self):
        """Create a mock SPLADE service."""
        mock = MagicMock()
        mock.encode = AsyncMock(return_value=([42, 100], [0.8, 0.5]))
        mock.encode_batch = AsyncMock(return_value=[([42, 100], [0.8, 0.5])])
        return mock

    @pytest.mark.anyio
    async def test_search_vectors_success(self):
        """Test successful vector search."""
        mock_hf_options = MagicMock()
        mock_hf_options.inference_model = "test-model"
        mock_hf_options.api_token = "test-token"

        mock_qdrant_client = MagicMock()
        mock_qdrant_options = MagicMock()
        mock_qdrant_options.collection_name = "test-collection"
        mock_qdrant_options.semantic_search_limit = 30
        mock_qdrant_options.semantic_search_prefetch_limit = 100
        mock_qdrant_options.semantic_search_score_threshold = 0.5

        # Create a proper CocktailModel JSON with all required fields
        cocktail_json = """{
            "id": "cocktail-123",
            "title": "Margarita",
            "descriptiveTitle": "Classic Margarita",
            "rating": 4.5,
            "ingredients": [],
            "isIba": true,
            "serves": 1,
            "prepTimeMinutes": 5,
            "searchTiles": ["tequila", "lime"],
            "glassware": []
        }"""

        mock_point = MagicMock()
        mock_point.score = 0.9
        mock_point.payload = {
            "metadata": {
                "cocktail_id": "cocktail-123",
                "model": cocktail_json,
                "keywords_search_terms": ["hangover cure", "brunch"],
            }
        }

        mock_search_results = MagicMock()
        mock_search_results.points = [mock_point]
        mock_qdrant_client.query_points = MagicMock(return_value=mock_search_results)

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_search_repository.HuggingFaceEndpointEmbeddings"
        ) as mock_hf_class:
            mock_embeddings = AsyncMock()
            mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
            mock_hf_class.return_value = mock_embeddings

            repo = CocktailVectorSearchRepository(
                hugging_face_options=mock_hf_options,
                qdrant_client=mock_qdrant_client,
                qdrant_options=mock_qdrant_options,
                splade_service=self._make_splade_service(),
            )

            result = await repo.search_vectors("tequila cocktails")

        assert len(result) == 1
        assert result[0].id == "cocktail-123"
        assert result[0].title == "Margarita"
        assert result[0].search_statistics is not None
        assert result[0].search_statistics.total_score == 0.9
        assert result[0].keywords_search_terms == ["hangover cure", "brunch"]

        # Verify hybrid search was used (prefetch + RRF)
        call_kwargs = mock_qdrant_client.query_points.call_args[1]
        from qdrant_client.http.models import Fusion, FusionQuery

        assert call_kwargs["query"] == FusionQuery(fusion=Fusion.RRF)
        assert len(call_kwargs["prefetch"]) == 2

    @pytest.mark.anyio
    async def test_search_vectors_with_filter(self):
        """Test that query_filter is passed through to Qdrant."""
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue

        mock_hf_options = MagicMock()
        mock_hf_options.inference_model = "test-model"
        mock_hf_options.api_token = "test-token"

        mock_qdrant_client = MagicMock()
        mock_qdrant_options = MagicMock()
        mock_qdrant_options.collection_name = "test-collection"
        mock_qdrant_options.semantic_search_limit = 30
        mock_qdrant_options.semantic_search_prefetch_limit = 100
        mock_qdrant_options.semantic_search_score_threshold = 0.5

        mock_search_results = MagicMock()
        mock_search_results.points = []
        mock_qdrant_client.query_points = MagicMock(return_value=mock_search_results)

        test_filter = Filter(must=[FieldCondition(key="metadata.is_iba", match=MatchValue(value=True))])

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_search_repository.HuggingFaceEndpointEmbeddings"
        ) as mock_hf_class:
            mock_embeddings = AsyncMock()
            mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
            mock_hf_class.return_value = mock_embeddings

            repo = CocktailVectorSearchRepository(
                hugging_face_options=mock_hf_options,
                qdrant_client=mock_qdrant_client,
                qdrant_options=mock_qdrant_options,
                splade_service=self._make_splade_service(),
            )

            await repo.search_vectors("iba cocktails", query_filter=test_filter)

        call_kwargs = mock_qdrant_client.query_points.call_args[1]
        # Filter should be on both prefetches
        assert call_kwargs["prefetch"][0].filter == test_filter
        assert call_kwargs["prefetch"][1].filter == test_filter

    @pytest.mark.anyio
    async def test_search_vectors_handles_duplicates(self):
        """Test that search_vectors aggregates duplicate cocktail results."""
        mock_hf_options = MagicMock()
        mock_hf_options.inference_model = "test-model"
        mock_hf_options.api_token = "test-token"

        mock_qdrant_client = MagicMock()
        mock_qdrant_options = MagicMock()
        mock_qdrant_options.collection_name = "test-collection"
        mock_qdrant_options.semantic_search_limit = 30
        mock_qdrant_options.semantic_search_prefetch_limit = 100
        mock_qdrant_options.semantic_search_score_threshold = 0.5

        # Create a proper CocktailModel JSON with all required fields
        cocktail_json = """{
            "id": "cocktail-123",
            "title": "Margarita",
            "descriptiveTitle": "Classic Margarita",
            "rating": 4.5,
            "ingredients": [],
            "isIba": true,
            "serves": 1,
            "prepTimeMinutes": 5,
            "searchTiles": ["tequila", "lime"],
            "glassware": []
        }"""

        # Two points with same cocktail_id
        mock_point1 = MagicMock()
        mock_point1.score = 0.9
        mock_point1.payload = {"metadata": {"cocktail_id": "cocktail-123", "model": cocktail_json}}

        mock_point2 = MagicMock()
        mock_point2.score = 0.8
        mock_point2.payload = {"metadata": {"cocktail_id": "cocktail-123", "model": cocktail_json}}

        mock_search_results = MagicMock()
        mock_search_results.points = [mock_point1, mock_point2]
        mock_qdrant_client.query_points = MagicMock(return_value=mock_search_results)

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_search_repository.HuggingFaceEndpointEmbeddings"
        ) as mock_hf_class:
            mock_embeddings = AsyncMock()
            mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
            mock_hf_class.return_value = mock_embeddings

            repo = CocktailVectorSearchRepository(
                hugging_face_options=mock_hf_options,
                qdrant_client=mock_qdrant_client,
                qdrant_options=mock_qdrant_options,
                splade_service=self._make_splade_service(),
            )

            result = await repo.search_vectors("tequila")

        # Should only return one cocktail with aggregated scores
        assert len(result) == 1
        assert result[0].id == "cocktail-123"
        assert result[0].title == "Margarita"
        assert result[0].search_statistics is not None
        assert result[0].search_statistics.total_score == pytest.approx(1.7, rel=1e-9)  # 0.9 + 0.8
        assert len(result[0].search_statistics.hit_results) == 2

    @pytest.mark.anyio
    async def test_search_vectors_weighted_score_formula(self):
        """Test that the improved weighted scoring formula is applied correctly.
        Formula: max_score * 0.6 + avg_score * 0.3 + log(hit_count + 1) * 0.1
        """
        mock_hf_options = MagicMock()
        mock_hf_options.inference_model = "test-model"
        mock_hf_options.api_token = "test-token"

        mock_qdrant_client = MagicMock()
        mock_qdrant_options = MagicMock()
        mock_qdrant_options.collection_name = "test-collection"
        mock_qdrant_options.semantic_search_limit = 30
        mock_qdrant_options.semantic_search_prefetch_limit = 100
        mock_qdrant_options.semantic_search_score_threshold = 0.0

        cocktail_json = """{
            "id": "cocktail-1",
            "title": "Test",
            "descriptiveTitle": "Test Desc",
            "rating": 4.0,
            "ingredients": [],
            "isIba": false,
            "serves": 1,
            "prepTimeMinutes": 5,
            "searchTiles": [],
            "glassware": []
        }"""

        # 3 chunks: scores 0.9, 0.7, 0.5
        mock_points = []
        for score in [0.9, 0.7, 0.5]:
            point = MagicMock()
            point.score = score
            point.payload = {"metadata": {"cocktail_id": "cocktail-1", "model": cocktail_json}}
            mock_points.append(point)

        mock_search_results = MagicMock()
        mock_search_results.points = mock_points
        mock_qdrant_client.query_points = MagicMock(return_value=mock_search_results)

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_search_repository.HuggingFaceEndpointEmbeddings"
        ) as mock_hf_class:
            mock_embeddings = AsyncMock()
            mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
            mock_hf_class.return_value = mock_embeddings

            repo = CocktailVectorSearchRepository(
                hugging_face_options=mock_hf_options,
                qdrant_client=mock_qdrant_client,
                qdrant_options=mock_qdrant_options,
                splade_service=self._make_splade_service(),
            )

            result = await repo.search_vectors("test query")

        assert len(result) == 1
        stats = result[0].search_statistics
        assert stats is not None
        assert stats.max_score == pytest.approx(0.9)
        assert stats.avg_score == pytest.approx(0.7)  # (0.9 + 0.7 + 0.5) / 3
        assert stats.hit_count == 3

        # Verify weighted score: max_score * 0.6 + avg_score * 0.3 + log(hit_count + 1) * 0.1
        expected_weighted = 0.9 * 0.6 + 0.7 * 0.3 + math.log(4) * 0.1
        assert stats.weighted_score == pytest.approx(expected_weighted, rel=1e-6)

    @pytest.mark.anyio
    async def test_search_vectors_single_hit_weighted_score(self):
        """Test weighted scoring formula with a single hit."""
        mock_hf_options = MagicMock()
        mock_hf_options.inference_model = "test-model"
        mock_hf_options.api_token = "test-token"

        mock_qdrant_client = MagicMock()
        mock_qdrant_options = MagicMock()
        mock_qdrant_options.collection_name = "test-collection"
        mock_qdrant_options.semantic_search_limit = 30
        mock_qdrant_options.semantic_search_prefetch_limit = 100
        mock_qdrant_options.semantic_search_score_threshold = 0.0

        cocktail_json = """{
            "id": "cocktail-1",
            "title": "Test",
            "descriptiveTitle": "Test Desc",
            "rating": 4.0,
            "ingredients": [],
            "isIba": false,
            "serves": 1,
            "prepTimeMinutes": 5,
            "searchTiles": [],
            "glassware": []
        }"""

        mock_point = MagicMock()
        mock_point.score = 0.85
        mock_point.payload = {"metadata": {"cocktail_id": "cocktail-1", "model": cocktail_json}}

        mock_search_results = MagicMock()
        mock_search_results.points = [mock_point]
        mock_qdrant_client.query_points = MagicMock(return_value=mock_search_results)

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_search_repository.HuggingFaceEndpointEmbeddings"
        ) as mock_hf_class:
            mock_embeddings = AsyncMock()
            mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
            mock_hf_class.return_value = mock_embeddings

            repo = CocktailVectorSearchRepository(
                hugging_face_options=mock_hf_options,
                qdrant_client=mock_qdrant_client,
                qdrant_options=mock_qdrant_options,
                splade_service=self._make_splade_service(),
            )

            result = await repo.search_vectors("test")

        stats = result[0].search_statistics
        # single hit: max=0.85, avg=0.85, hit_count=1
        expected_weighted = 0.85 * 0.6 + 0.85 * 0.3 + math.log(2) * 0.1
        assert stats.weighted_score == pytest.approx(expected_weighted, rel=1e-6)

    @pytest.mark.anyio
    async def test_embedding_cache_hit(self):
        """Test that repeated queries use cached embeddings instead of calling the API."""
        mock_hf_options = MagicMock()
        mock_hf_options.inference_model = "test-model"
        mock_hf_options.api_token = "test-token"

        mock_qdrant_client = MagicMock()
        mock_qdrant_options = MagicMock()
        mock_qdrant_options.collection_name = "test-collection"
        mock_qdrant_options.semantic_search_limit = 30
        mock_qdrant_options.semantic_search_prefetch_limit = 100
        mock_qdrant_options.semantic_search_score_threshold = 0.5

        mock_search_results = MagicMock()
        mock_search_results.points = []
        mock_qdrant_client.query_points = MagicMock(return_value=mock_search_results)

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_search_repository.HuggingFaceEndpointEmbeddings"
        ) as mock_hf_class:
            mock_embeddings = AsyncMock()
            mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
            mock_hf_class.return_value = mock_embeddings

            repo = CocktailVectorSearchRepository(
                hugging_face_options=mock_hf_options,
                qdrant_client=mock_qdrant_client,
                qdrant_options=mock_qdrant_options,
                splade_service=self._make_splade_service(),
            )

            # First call should generate embedding
            await repo.search_vectors("tequila cocktails")
            assert mock_embeddings.aembed_query.call_count == 1

            # Second call with same query should use cache
            await repo.search_vectors("tequila cocktails")
            assert mock_embeddings.aembed_query.call_count == 1  # NOT incremented

            # Different query should generate new embedding
            await repo.search_vectors("gin cocktails")
            assert mock_embeddings.aembed_query.call_count == 2

    @pytest.mark.anyio
    async def test_embedding_cache_case_insensitive(self):
        """Test that embedding cache keys are case-insensitive."""
        mock_hf_options = MagicMock()
        mock_hf_options.inference_model = "test-model"
        mock_hf_options.api_token = "test-token"

        mock_qdrant_client = MagicMock()
        mock_qdrant_options = MagicMock()
        mock_qdrant_options.collection_name = "test-collection"
        mock_qdrant_options.semantic_search_limit = 30
        mock_qdrant_options.semantic_search_prefetch_limit = 100
        mock_qdrant_options.semantic_search_score_threshold = 0.5

        mock_search_results = MagicMock()
        mock_search_results.points = []
        mock_qdrant_client.query_points = MagicMock(return_value=mock_search_results)

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_search_repository.HuggingFaceEndpointEmbeddings"
        ) as mock_hf_class:
            mock_embeddings = AsyncMock()
            mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
            mock_hf_class.return_value = mock_embeddings

            repo = CocktailVectorSearchRepository(
                hugging_face_options=mock_hf_options,
                qdrant_client=mock_qdrant_client,
                qdrant_options=mock_qdrant_options,
                splade_service=self._make_splade_service(),
            )

            await repo.search_vectors("Tequila Cocktails")
            await repo.search_vectors("tequila cocktails")
            # Both should resolve to same cache key
            assert mock_embeddings.aembed_query.call_count == 1

    @pytest.mark.anyio
    async def test_embedding_cache_eviction(self):
        """Test that oldest cached embeddings are evicted when cache is full."""
        mock_hf_options = MagicMock()
        mock_hf_options.inference_model = "test-model"
        mock_hf_options.api_token = "test-token"

        mock_qdrant_client = MagicMock()
        mock_qdrant_options = MagicMock()
        mock_qdrant_options.collection_name = "test-collection"
        mock_qdrant_options.semantic_search_limit = 30
        mock_qdrant_options.semantic_search_prefetch_limit = 100
        mock_qdrant_options.semantic_search_score_threshold = 0.5

        mock_search_results = MagicMock()
        mock_search_results.points = []
        mock_qdrant_client.query_points = MagicMock(return_value=mock_search_results)

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_search_repository.HuggingFaceEndpointEmbeddings"
        ) as mock_hf_class:
            mock_embeddings = AsyncMock()
            mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
            mock_hf_class.return_value = mock_embeddings

            repo = CocktailVectorSearchRepository(
                hugging_face_options=mock_hf_options,
                qdrant_client=mock_qdrant_client,
                qdrant_options=mock_qdrant_options,
                splade_service=self._make_splade_service(),
            )

            # Set a small cache size for testing
            repo._embedding_cache_max_size = 3

            # Fill cache with 3 entries
            await repo.search_vectors("query 1")
            await repo.search_vectors("query 2")
            await repo.search_vectors("query 3")
            assert mock_embeddings.aembed_query.call_count == 3
            assert len(repo._embedding_cache) == 3

            # Add a 4th should evict the oldest ("query 1")
            await repo.search_vectors("query 4")
            assert mock_embeddings.aembed_query.call_count == 4
            assert len(repo._embedding_cache) == 3
            assert "query 1" not in repo._embedding_cache
            assert "query 4" in repo._embedding_cache

    @pytest.mark.anyio
    async def test_search_vectors_hybrid_uses_prefetch_rrf(self):
        """Test that hybrid search uses prefetch + RRF fusion."""
        from qdrant_client.http.models import Fusion, FusionQuery, Prefetch, SparseVector

        mock_hf_options = MagicMock()
        mock_hf_options.inference_model = "test-model"
        mock_hf_options.api_token = "test-token"

        mock_qdrant_client = MagicMock()
        mock_qdrant_options = MagicMock()
        mock_qdrant_options.collection_name = "test-collection"
        mock_qdrant_options.semantic_search_limit = 30
        mock_qdrant_options.semantic_search_prefetch_limit = 100
        mock_qdrant_options.semantic_search_score_threshold = 0.5

        cocktail_json = """{
            "id": "cocktail-123",
            "title": "Margarita",
            "descriptiveTitle": "Classic Margarita",
            "rating": 4.5,
            "ingredients": [],
            "isIba": true,
            "serves": 1,
            "prepTimeMinutes": 5,
            "searchTiles": ["tequila", "lime"],
            "glassware": []
        }"""

        mock_point = MagicMock()
        mock_point.score = 0.9
        mock_point.payload = {"metadata": {"cocktail_id": "cocktail-123", "model": cocktail_json}}

        mock_search_results = MagicMock()
        mock_search_results.points = [mock_point]
        mock_qdrant_client.query_points = MagicMock(return_value=mock_search_results)

        mock_splade = self._make_splade_service()
        mock_splade.encode = AsyncMock(return_value=([42, 100], [0.8, 0.5]))

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_search_repository.HuggingFaceEndpointEmbeddings"
        ) as mock_hf_class:
            mock_embeddings = AsyncMock()
            mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
            mock_hf_class.return_value = mock_embeddings

            repo = CocktailVectorSearchRepository(
                hugging_face_options=mock_hf_options,
                qdrant_client=mock_qdrant_client,
                qdrant_options=mock_qdrant_options,
                splade_service=mock_splade,
            )

            result = await repo.search_vectors("tequila cocktails")

        assert len(result) == 1
        assert result[0].id == "cocktail-123"

        # Verify prefetch + RRF was used
        call_kwargs = mock_qdrant_client.query_points.call_args[1]
        assert call_kwargs["query"] == FusionQuery(fusion=Fusion.RRF)
        assert len(call_kwargs["prefetch"]) == 2

        # Verify dense prefetch uses prefetch_limit (100), not fusion limit (30)
        dense_prefetch = call_kwargs["prefetch"][0]
        assert dense_prefetch.using == "dense"
        assert dense_prefetch.query == [0.1, 0.2, 0.3]
        assert dense_prefetch.limit == 100

        # Verify sparse prefetch uses prefetch_limit (100)
        sparse_prefetch = call_kwargs["prefetch"][1]
        assert sparse_prefetch.using == "sparse"
        assert sparse_prefetch.limit == 100

        # Verify fusion limit is the main semantic_search_limit (30)
        assert call_kwargs["limit"] == 30

        # Verify SPLADE was called with the query text
        mock_splade.encode.assert_called_once_with("tequila cocktails")

    @pytest.mark.anyio
    async def test_search_vectors_hybrid_fallback_on_splade_failure(self):
        """Test that hybrid search falls back to dense-only when SPLADE fails."""
        mock_hf_options = MagicMock()
        mock_hf_options.inference_model = "test-model"
        mock_hf_options.api_token = "test-token"

        mock_qdrant_client = MagicMock()
        mock_qdrant_options = MagicMock()
        mock_qdrant_options.collection_name = "test-collection"
        mock_qdrant_options.semantic_search_limit = 30
        mock_qdrant_options.semantic_search_prefetch_limit = 100
        mock_qdrant_options.semantic_search_score_threshold = 0.5

        mock_search_results = MagicMock()
        mock_search_results.points = []
        mock_qdrant_client.query_points = MagicMock(return_value=mock_search_results)

        mock_splade = self._make_splade_service()
        mock_splade.encode = AsyncMock(side_effect=Exception("SPLADE down"))

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_search_repository.HuggingFaceEndpointEmbeddings"
        ) as mock_hf_class:
            mock_embeddings = AsyncMock()
            mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
            mock_hf_class.return_value = mock_embeddings

            repo = CocktailVectorSearchRepository(
                hugging_face_options=mock_hf_options,
                qdrant_client=mock_qdrant_client,
                qdrant_options=mock_qdrant_options,
                splade_service=mock_splade,
            )

            await repo.search_vectors("tequila cocktails")

        # Should have used dense-only search (no prefetch, direct query)
        call_kwargs = mock_qdrant_client.query_points.call_args[1]
        assert call_kwargs["query"] == [0.1, 0.2, 0.3]
        assert "prefetch" not in call_kwargs or call_kwargs.get("prefetch") is None

    @pytest.mark.anyio
    async def test_search_vectors_hybrid_fallback_on_empty_sparse(self):
        """Test that hybrid search falls back to dense-only when SPLADE returns empty."""
        mock_hf_options = MagicMock()
        mock_hf_options.inference_model = "test-model"
        mock_hf_options.api_token = "test-token"

        mock_qdrant_client = MagicMock()
        mock_qdrant_options = MagicMock()
        mock_qdrant_options.collection_name = "test-collection"
        mock_qdrant_options.semantic_search_limit = 30
        mock_qdrant_options.semantic_search_prefetch_limit = 100
        mock_qdrant_options.semantic_search_score_threshold = 0.5

        mock_search_results = MagicMock()
        mock_search_results.points = []
        mock_qdrant_client.query_points = MagicMock(return_value=mock_search_results)

        mock_splade = self._make_splade_service()
        mock_splade.encode = AsyncMock(return_value=([], []))

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_search_repository.HuggingFaceEndpointEmbeddings"
        ) as mock_hf_class:
            mock_embeddings = AsyncMock()
            mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
            mock_hf_class.return_value = mock_embeddings

            repo = CocktailVectorSearchRepository(
                hugging_face_options=mock_hf_options,
                qdrant_client=mock_qdrant_client,
                qdrant_options=mock_qdrant_options,
                splade_service=mock_splade,
            )

            await repo.search_vectors("tequila cocktails")

        # Should have used dense-only search
        call_kwargs = mock_qdrant_client.query_points.call_args[1]
        assert call_kwargs["query"] == [0.1, 0.2, 0.3]

    @pytest.mark.anyio
    async def test_search_vectors_hybrid_passes_filter_to_both_prefetches(self):
        """Test that query filter is passed to both dense and sparse prefetches."""
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue

        mock_hf_options = MagicMock()
        mock_hf_options.inference_model = "test-model"
        mock_hf_options.api_token = "test-token"

        mock_qdrant_client = MagicMock()
        mock_qdrant_options = MagicMock()
        mock_qdrant_options.collection_name = "test-collection"
        mock_qdrant_options.semantic_search_limit = 30
        mock_qdrant_options.semantic_search_prefetch_limit = 100
        mock_qdrant_options.semantic_search_score_threshold = 0.5

        mock_search_results = MagicMock()
        mock_search_results.points = []
        mock_qdrant_client.query_points = MagicMock(return_value=mock_search_results)

        test_filter = Filter(must=[FieldCondition(key="metadata.is_iba", match=MatchValue(value=True))])

        mock_splade = self._make_splade_service()
        mock_splade.encode = AsyncMock(return_value=([10], [0.9]))

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_search_repository.HuggingFaceEndpointEmbeddings"
        ) as mock_hf_class:
            mock_embeddings = AsyncMock()
            mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
            mock_hf_class.return_value = mock_embeddings

            repo = CocktailVectorSearchRepository(
                hugging_face_options=mock_hf_options,
                qdrant_client=mock_qdrant_client,
                qdrant_options=mock_qdrant_options,
                splade_service=mock_splade,
            )

            await repo.search_vectors("iba cocktails", query_filter=test_filter)

        call_kwargs = mock_qdrant_client.query_points.call_args[1]
        # Both prefetches should have the filter
        assert call_kwargs["prefetch"][0].filter == test_filter
        assert call_kwargs["prefetch"][1].filter == test_filter

    @pytest.mark.anyio
    async def test_search_vectors_hybrid_prefetch_limit_differs_from_fusion_limit(self):
        """Test that prefetch branches use prefetch_limit while fusion output uses semantic_search_limit.

        This ensures the retrieval stage casts a wider net (e.g., 150 per branch)
        while the RRF fusion output is limited (e.g., 40), feeding a manageable
        number of candidates to the reranker.
        """
        mock_hf_options = MagicMock()
        mock_hf_options.inference_model = "test-model"
        mock_hf_options.api_token = "test-token"

        mock_qdrant_client = MagicMock()
        mock_qdrant_options = MagicMock()
        mock_qdrant_options.collection_name = "test-collection"
        mock_qdrant_options.semantic_search_limit = 40
        mock_qdrant_options.semantic_search_prefetch_limit = 150
        mock_qdrant_options.semantic_search_score_threshold = 0.0

        mock_search_results = MagicMock()
        mock_search_results.points = []
        mock_qdrant_client.query_points = MagicMock(return_value=mock_search_results)

        mock_splade = self._make_splade_service()
        mock_splade.encode = AsyncMock(return_value=([10, 20], [0.9, 0.4]))

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_search_repository.HuggingFaceEndpointEmbeddings"
        ) as mock_hf_class:
            mock_embeddings = AsyncMock()
            mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
            mock_hf_class.return_value = mock_embeddings

            repo = CocktailVectorSearchRepository(
                hugging_face_options=mock_hf_options,
                qdrant_client=mock_qdrant_client,
                qdrant_options=mock_qdrant_options,
                splade_service=mock_splade,
            )

            await repo.search_vectors("cocktails with berries")

        call_kwargs = mock_qdrant_client.query_points.call_args[1]

        # Prefetch branches should use the larger prefetch_limit
        assert call_kwargs["prefetch"][0].limit == 150
        assert call_kwargs["prefetch"][1].limit == 150

        # RRF fusion output should use the smaller semantic_search_limit
        assert call_kwargs["limit"] == 40
