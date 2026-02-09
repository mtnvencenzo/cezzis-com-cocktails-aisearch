import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import create_test_cocktail_model

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_description_chunk import (
    CocktailDescriptionChunk,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailSearchModel
from cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_repository import (
    CocktailVectorRepository,
)


class TestCocktailVectorRepository:
    """Test cases for CocktailVectorRepository."""

    def test_init(self):
        """Test repository initialization."""
        mock_hf_options = MagicMock()
        mock_hf_options.inference_model = "test-model"
        mock_hf_options.api_token = "test-token"

        mock_qdrant_client = MagicMock()
        mock_qdrant_options = MagicMock()
        mock_qdrant_options.collection_name = "test-collection"

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_repository.QdrantVectorStore"
        ):
            repo = CocktailVectorRepository(
                hugging_face_options=mock_hf_options,
                qdrant_client=mock_qdrant_client,
                qdrant_options=mock_qdrant_options,
            )

        assert repo.hugging_face_options == mock_hf_options
        assert repo.qdrant_client == mock_qdrant_client
        assert repo.qdrant_options == mock_qdrant_options
        assert repo.vector_store is not None

    @pytest.mark.anyio
    async def test_delete_vectors_success(self):
        """Test successful deletion of vectors."""
        mock_hf_options = MagicMock()
        mock_hf_options.inference_model = "test-model"
        mock_hf_options.api_token = "test-token"

        mock_qdrant_client = MagicMock()
        mock_qdrant_options = MagicMock()
        mock_qdrant_options.collection_name = "test-collection"

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_repository.QdrantVectorStore"
        ):
            repo = CocktailVectorRepository(
                hugging_face_options=mock_hf_options,
                qdrant_client=mock_qdrant_client,
                qdrant_options=mock_qdrant_options,
            )

        await repo.delete_vectors("cocktail-123")

        mock_qdrant_client.delete.assert_called_once()
        call_kwargs = mock_qdrant_client.delete.call_args[1]
        assert call_kwargs["collection_name"] == "test-collection"
        assert call_kwargs["wait"] is True

    @pytest.mark.anyio
    async def test_store_vectors_success(self):
        """Test successful storage of vectors with enriched metadata."""
        mock_hf_options = MagicMock()
        mock_hf_options.inference_model = "test-model"
        mock_hf_options.api_token = "test-token"

        mock_qdrant_client = MagicMock()
        mock_qdrant_options = MagicMock()
        mock_qdrant_options.collection_name = "test-collection"

        mock_vector_store = MagicMock()
        mock_vector_store.add_texts = MagicMock(return_value=["id1", "id2"])

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_repository.QdrantVectorStore",
            return_value=mock_vector_store,
        ):
            repo = CocktailVectorRepository(
                hugging_face_options=mock_hf_options,
                qdrant_client=mock_qdrant_client,
                qdrant_options=mock_qdrant_options,
            )

        cocktail_model = create_test_cocktail_model("cocktail-123", "Test Cocktail")
        chunks = [
            CocktailDescriptionChunk(content="Description 1", category="desc"),
            CocktailDescriptionChunk(content="Description 2", category="ingredients"),
        ]

        await repo.store_vectors("cocktail-123", chunks, cocktail_model)

        mock_vector_store.add_texts.assert_called_once()

        # Verify enriched metadata fields are stored
        call_kwargs = mock_vector_store.add_texts.call_args[1]
        metadatas = call_kwargs["metadatas"]
        assert len(metadatas) == 2
        for metadata in metadatas:
            assert metadata["cocktail_id"] == "cocktail-123"
            assert metadata["title"] == "test cocktail"
            assert "is_iba" in metadata
            assert "serves" in metadata
            assert "prep_time_minutes" in metadata
            assert "ingredient_count" in metadata
            assert "ingredient_names" in metadata
            assert "ingredient_words" in metadata
            assert "glassware_values" in metadata
            assert "rating" in metadata
            assert "keywords_base_spirit" in metadata
            assert "keywords_spirit_subtype" in metadata
            assert "keywords_flavor_profile" in metadata
            assert "keywords_cocktail_family" in metadata
            assert "keywords_technique" in metadata
            assert "keywords_strength" in metadata
            assert "keywords_temperature" in metadata
            assert "keywords_season" in metadata
            assert "keywords_occasion" in metadata
            assert "keywords_mood" in metadata
            assert "keywords_search_terms" in metadata

    @pytest.mark.anyio
    async def test_store_vectors_raises_on_empty_result(self):
        """Test that store_vectors raises error when no results returned."""
        mock_hf_options = MagicMock()
        mock_hf_options.inference_model = "test-model"
        mock_hf_options.api_token = "test-token"

        mock_qdrant_client = MagicMock()
        mock_qdrant_options = MagicMock()
        mock_qdrant_options.collection_name = "test-collection"

        mock_vector_store = MagicMock()
        mock_vector_store.add_texts = MagicMock(return_value=[])

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_repository.QdrantVectorStore",
            return_value=mock_vector_store,
        ):
            repo = CocktailVectorRepository(
                hugging_face_options=mock_hf_options,
                qdrant_client=mock_qdrant_client,
                qdrant_options=mock_qdrant_options,
            )

        cocktail_model = create_test_cocktail_model("cocktail-123", "Test Cocktail")
        chunks = [CocktailDescriptionChunk(content="Test", category="desc")]

        with pytest.raises(ValueError, match="No embedding results returned"):
            await repo.store_vectors("cocktail-123", chunks, cocktail_model)

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
        mock_point.payload = {"metadata": {"cocktail_id": "cocktail-123", "model": cocktail_json}}

        mock_search_results = MagicMock()
        mock_search_results.points = [mock_point]
        mock_qdrant_client.query_points = MagicMock(return_value=mock_search_results)

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_repository.QdrantVectorStore"
        ):
            with patch(
                "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_repository.HuggingFaceEndpointEmbeddings"
            ) as mock_hf_class:
                mock_embeddings = AsyncMock()
                mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
                mock_hf_class.return_value = mock_embeddings

                repo = CocktailVectorRepository(
                    hugging_face_options=mock_hf_options,
                    qdrant_client=mock_qdrant_client,
                    qdrant_options=mock_qdrant_options,
                )

                result = await repo.search_vectors("tequila cocktails")

        assert len(result) == 1
        assert result[0].id == "cocktail-123"
        assert result[0].title == "Margarita"
        assert result[0].search_statistics is not None
        assert result[0].search_statistics.total_score == 0.9

        # Verify query_filter=None is used by default
        call_kwargs = mock_qdrant_client.query_points.call_args[1]
        assert call_kwargs["query_filter"] is None

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
        mock_qdrant_options.semantic_search_score_threshold = 0.5

        mock_search_results = MagicMock()
        mock_search_results.points = []
        mock_qdrant_client.query_points = MagicMock(return_value=mock_search_results)

        test_filter = Filter(must=[FieldCondition(key="metadata.is_iba", match=MatchValue(value=True))])

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_repository.QdrantVectorStore"
        ):
            with patch(
                "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_repository.HuggingFaceEndpointEmbeddings"
            ) as mock_hf_class:
                mock_embeddings = AsyncMock()
                mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
                mock_hf_class.return_value = mock_embeddings

                repo = CocktailVectorRepository(
                    hugging_face_options=mock_hf_options,
                    qdrant_client=mock_qdrant_client,
                    qdrant_options=mock_qdrant_options,
                )

                await repo.search_vectors("iba cocktails", query_filter=test_filter)

        call_kwargs = mock_qdrant_client.query_points.call_args[1]
        assert call_kwargs["query_filter"] == test_filter

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
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_repository.QdrantVectorStore"
        ):
            with patch(
                "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_repository.HuggingFaceEndpointEmbeddings"
            ) as mock_hf_class:
                mock_embeddings = AsyncMock()
                mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
                mock_hf_class.return_value = mock_embeddings

                repo = CocktailVectorRepository(
                    hugging_face_options=mock_hf_options,
                    qdrant_client=mock_qdrant_client,
                    qdrant_options=mock_qdrant_options,
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
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_repository.QdrantVectorStore"
        ):
            with patch(
                "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_repository.HuggingFaceEndpointEmbeddings"
            ) as mock_hf_class:
                mock_embeddings = AsyncMock()
                mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
                mock_hf_class.return_value = mock_embeddings

                repo = CocktailVectorRepository(
                    hugging_face_options=mock_hf_options,
                    qdrant_client=mock_qdrant_client,
                    qdrant_options=mock_qdrant_options,
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
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_repository.QdrantVectorStore"
        ):
            with patch(
                "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_repository.HuggingFaceEndpointEmbeddings"
            ) as mock_hf_class:
                mock_embeddings = AsyncMock()
                mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
                mock_hf_class.return_value = mock_embeddings

                repo = CocktailVectorRepository(
                    hugging_face_options=mock_hf_options,
                    qdrant_client=mock_qdrant_client,
                    qdrant_options=mock_qdrant_options,
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
        mock_qdrant_options.semantic_search_score_threshold = 0.5

        mock_search_results = MagicMock()
        mock_search_results.points = []
        mock_qdrant_client.query_points = MagicMock(return_value=mock_search_results)

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_repository.QdrantVectorStore"
        ):
            with patch(
                "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_repository.HuggingFaceEndpointEmbeddings"
            ) as mock_hf_class:
                mock_embeddings = AsyncMock()
                mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
                mock_hf_class.return_value = mock_embeddings

                repo = CocktailVectorRepository(
                    hugging_face_options=mock_hf_options,
                    qdrant_client=mock_qdrant_client,
                    qdrant_options=mock_qdrant_options,
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
        mock_qdrant_options.semantic_search_score_threshold = 0.5

        mock_search_results = MagicMock()
        mock_search_results.points = []
        mock_qdrant_client.query_points = MagicMock(return_value=mock_search_results)

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_repository.QdrantVectorStore"
        ):
            with patch(
                "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_repository.HuggingFaceEndpointEmbeddings"
            ) as mock_hf_class:
                mock_embeddings = AsyncMock()
                mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
                mock_hf_class.return_value = mock_embeddings

                repo = CocktailVectorRepository(
                    hugging_face_options=mock_hf_options,
                    qdrant_client=mock_qdrant_client,
                    qdrant_options=mock_qdrant_options,
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
        mock_qdrant_options.semantic_search_score_threshold = 0.5

        mock_search_results = MagicMock()
        mock_search_results.points = []
        mock_qdrant_client.query_points = MagicMock(return_value=mock_search_results)

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_repository.QdrantVectorStore"
        ):
            with patch(
                "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_repository.HuggingFaceEndpointEmbeddings"
            ) as mock_hf_class:
                mock_embeddings = AsyncMock()
                mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
                mock_hf_class.return_value = mock_embeddings

                repo = CocktailVectorRepository(
                    hugging_face_options=mock_hf_options,
                    qdrant_client=mock_qdrant_client,
                    qdrant_options=mock_qdrant_options,
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
