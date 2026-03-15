from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import create_test_cocktail_model

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_description_chunk import (
    CocktailDescriptionChunk,
)
from cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_embedding_repository import (
    CocktailVectorEmbeddingRepository,
)


class TestCocktailVectorEmbeddingRepository:
    """Test cases for CocktailVectorEmbeddingRepository."""

    def _make_splade_service(self):
        """Create a mock SPLADE service."""
        mock = MagicMock()
        mock.encode = AsyncMock(return_value=([42, 100], [0.8, 0.5]))
        mock.encode_batch = AsyncMock(return_value=[([42, 100], [0.8, 0.5])])
        return mock

    def test_init(self):
        """Test repository initialization."""
        mock_hf_options = MagicMock()
        mock_hf_options.inference_model = "test-model"
        mock_hf_options.api_token = "test-token"

        mock_qdrant_client = MagicMock()
        mock_qdrant_options = MagicMock()
        mock_qdrant_options.collection_name = "test-collection"

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_embedding_repository.HuggingFaceEndpointEmbeddings"
        ):
            repo = CocktailVectorEmbeddingRepository(
                hugging_face_options=mock_hf_options,
                qdrant_client=mock_qdrant_client,
                qdrant_options=mock_qdrant_options,
                splade_service=self._make_splade_service(),
            )

        assert repo.hugging_face_options == mock_hf_options
        assert repo.qdrant_client == mock_qdrant_client
        assert repo.qdrant_options == mock_qdrant_options
        assert repo.splade_service is not None

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
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_embedding_repository.HuggingFaceEndpointEmbeddings"
        ):
            repo = CocktailVectorEmbeddingRepository(
                hugging_face_options=mock_hf_options,
                qdrant_client=mock_qdrant_client,
                qdrant_options=mock_qdrant_options,
                splade_service=self._make_splade_service(),
            )

        await repo.delete_vectors("cocktail-123")

        mock_qdrant_client.delete.assert_called_once()
        call_kwargs = mock_qdrant_client.delete.call_args[1]
        assert call_kwargs["collection_name"] == "test-collection"
        assert call_kwargs["wait"] is True

    @pytest.mark.anyio
    async def test_store_vectors_success(self):
        """Test successful storage of vectors with named dense + sparse vectors."""
        mock_hf_options = MagicMock()
        mock_hf_options.inference_model = "test-model"
        mock_hf_options.api_token = "test-token"

        mock_qdrant_client = MagicMock()
        mock_qdrant_options = MagicMock()
        mock_qdrant_options.collection_name = "test-collection"

        mock_splade = self._make_splade_service()
        mock_splade.encode_batch = AsyncMock(
            return_value=[
                ([10, 20], [0.9, 0.4]),
                ([30], [0.7]),
            ]
        )

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_embedding_repository.HuggingFaceEndpointEmbeddings"
        ) as mock_hf_class:
            mock_embeddings = AsyncMock()
            mock_embeddings.aembed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            mock_hf_class.return_value = mock_embeddings

            repo = CocktailVectorEmbeddingRepository(
                hugging_face_options=mock_hf_options,
                qdrant_client=mock_qdrant_client,
                qdrant_options=mock_qdrant_options,
                splade_service=mock_splade,
            )

        cocktail_model = create_test_cocktail_model("cocktail-123", "Test Cocktail")
        chunks = [
            CocktailDescriptionChunk(content="Description 1", category="desc"),
            CocktailDescriptionChunk(content="Description 2", category="ingredients"),
        ]

        await repo.store_vectors("cocktail-123", chunks, cocktail_model)

        # Verify upsert was called with named vectors
        mock_qdrant_client.upsert.assert_called_once()
        call_kwargs = mock_qdrant_client.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "test-collection"
        assert call_kwargs["wait"] is True

        points = call_kwargs["points"]
        assert len(points) == 2

        # Verify first point has dense + sparse named vectors
        first_point = points[0]
        assert "dense" in first_point.vector
        assert first_point.vector["dense"] == [0.1, 0.2, 0.3]
        assert "sparse" in first_point.vector
        assert first_point.vector["sparse"].indices == [10, 20]
        assert first_point.vector["sparse"].values == [0.9, 0.4]

        # Verify metadata is stored correctly
        metadata = first_point.payload["metadata"]
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
        assert "keywords_search_words" in metadata

    @pytest.mark.anyio
    async def test_store_vectors_raises_on_empty_dense_embeddings(self):
        """Test that store_vectors raises error when no dense embeddings returned."""
        mock_hf_options = MagicMock()
        mock_hf_options.inference_model = "test-model"
        mock_hf_options.api_token = "test-token"

        mock_qdrant_client = MagicMock()
        mock_qdrant_options = MagicMock()
        mock_qdrant_options.collection_name = "test-collection"

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_embedding_repository.HuggingFaceEndpointEmbeddings"
        ) as mock_hf_class:
            mock_embeddings = AsyncMock()
            mock_embeddings.aembed_documents = AsyncMock(return_value=[])
            mock_hf_class.return_value = mock_embeddings

            repo = CocktailVectorEmbeddingRepository(
                hugging_face_options=mock_hf_options,
                qdrant_client=mock_qdrant_client,
                qdrant_options=mock_qdrant_options,
                splade_service=self._make_splade_service(),
            )

        cocktail_model = create_test_cocktail_model("cocktail-123", "Test Cocktail")
        chunks = [CocktailDescriptionChunk(content="Test", category="desc")]

        with pytest.raises(ValueError, match="No dense embedding results"):
            await repo.store_vectors("cocktail-123", chunks, cocktail_model)

    @pytest.mark.anyio
    async def test_store_vectors_without_sparse_vectors(self):
        """Test that store_vectors stores dense-only when SPLADE returns empty."""
        mock_hf_options = MagicMock()
        mock_hf_options.inference_model = "test-model"
        mock_hf_options.api_token = "test-token"

        mock_qdrant_client = MagicMock()
        mock_qdrant_options = MagicMock()
        mock_qdrant_options.collection_name = "test-collection"

        mock_splade = self._make_splade_service()
        mock_splade.encode_batch = AsyncMock(return_value=[([], [])])

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.repositories.cocktail_vector_embedding_repository.HuggingFaceEndpointEmbeddings"
        ) as mock_hf_class:
            mock_embeddings = AsyncMock()
            mock_embeddings.aembed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
            mock_hf_class.return_value = mock_embeddings

            repo = CocktailVectorEmbeddingRepository(
                hugging_face_options=mock_hf_options,
                qdrant_client=mock_qdrant_client,
                qdrant_options=mock_qdrant_options,
                splade_service=mock_splade,
            )

        cocktail_model = create_test_cocktail_model("cocktail-123", "Test Cocktail")
        chunks = [CocktailDescriptionChunk(content="Test", category="desc")]

        await repo.store_vectors("cocktail-123", chunks, cocktail_model)

        points = mock_qdrant_client.upsert.call_args[1]["points"]
        assert len(points) == 1
        # Only dense vector, no sparse
        assert "dense" in points[0].vector
        assert "sparse" not in points[0].vector
