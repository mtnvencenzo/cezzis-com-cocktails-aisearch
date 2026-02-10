import os
from unittest.mock import patch

import pytest

from cezzis_com_cocktails_aisearch.domain.config.qdrant_options import (
    QdrantOptions,
    clear_qdrant_options_cache,
    get_qdrant_options,
)


class TestQdrantOptions:
    """ "Test cases for QdrantOptions configuration."""

    def test_qdrant_options_init_with_defaults(self):
        """Test QdrantOptions initialization with default values."""
        with patch.dict(os.environ, {}, clear=True):
            options = QdrantOptions()

            assert options.host == ""
            assert options.port == 6333
            assert options.api_key is None
            assert options.collection_name == ""
            assert options.vector_size == 0
            assert options.use_https is True
            assert options.semantic_search_limit == 30
            assert options.semantic_search_prefetch_limit == 100
            assert options.semantic_search_score_threshold == 0.0
            assert options.semantic_search_total_score_threshold == 0.0

    def test_qdrant_options_init_with_env_vars(self):
        """Test QdrantOptions initialization with environment variables."""
        with patch.dict(
            os.environ,
            {
                "QDRANT_HOST": "qdrant.example.com",
                "QDRANT_PORT": "6334",
                "QDRANT_API_KEY": "test-api-key",
                "QDRANT_COLLECTION_NAME": "cocktails",
                "QDRANT_VECTOR_SIZE": "768",
                "QDRANT_USE_HTTPS": "false",
                "QDRANT_SEMANTIC_SEARCH_LIMIT": "50",
                "QDRANT_SEMANTIC_SEARCH_PREFETCH_LIMIT": "200",
                "QDRANT_SEMANTIC_SEARCH_SCORE_THRESHOLD": "0.7",
                "QDRANT_SEMANTIC_SEARCH_TOTAL_SCORE_THRESHOLD": "1.5",
            },
        ):
            options = QdrantOptions()

            assert options.host == "qdrant.example.com"
            assert options.port == 6334
            assert options.api_key == "test-api-key"
            assert options.collection_name == "cocktails"
            assert options.vector_size == 768
            assert options.use_https is False
            assert options.semantic_search_limit == 50
            assert options.semantic_search_prefetch_limit == 200
            assert options.semantic_search_score_threshold == 0.7
            assert options.semantic_search_total_score_threshold == 1.5

    def test_get_qdrant_options_raises_on_missing_host(self):
        """Test that get_qdrant_options raises ValueError when host is missing."""
        clear_qdrant_options_cache()

        with patch.dict(os.environ, {"QDRANT_HOST": "", "QDRANT_COLLECTION_NAME": "test", "QDRANT_VECTOR_SIZE": "768"}):
            with pytest.raises(ValueError, match="QDRANT_HOST"):
                get_qdrant_options()

    def test_get_qdrant_options_raises_on_invalid_vector_size(self):
        """Test that get_qdrant_options raises ValueError for invalid vector size."""
        clear_qdrant_options_cache()

        with patch.dict(
            os.environ, {"QDRANT_HOST": "localhost", "QDRANT_COLLECTION_NAME": "test", "QDRANT_VECTOR_SIZE": "0"}
        ):
            with pytest.raises(ValueError, match="QDRANT_VECTOR_SIZE"):
                get_qdrant_options()

    def test_get_qdrant_options_raises_on_negative_threshold(self):
        """Test that get_qdrant_options raises ValueError for negative threshold."""
        clear_qdrant_options_cache()

        with patch.dict(
            os.environ,
            {
                "QDRANT_HOST": "localhost",
                "QDRANT_COLLECTION_NAME": "test",
                "QDRANT_VECTOR_SIZE": "768",
                "QDRANT_SEMANTIC_SEARCH_SCORE_THRESHOLD": "-0.5",
            },
        ):
            with pytest.raises(ValueError, match="QDRANT_SEMANTIC_SEARCH_SCORE_THRESHOLD"):
                get_qdrant_options()

    def test_get_qdrant_options_raises_on_invalid_prefetch_limit(self):
        """Test that get_qdrant_options raises ValueError for invalid prefetch limit."""
        clear_qdrant_options_cache()

        with patch.dict(
            os.environ,
            {
                "QDRANT_HOST": "localhost",
                "QDRANT_COLLECTION_NAME": "test",
                "QDRANT_VECTOR_SIZE": "768",
                "QDRANT_SEMANTIC_SEARCH_PREFETCH_LIMIT": "0",
            },
        ):
            with pytest.raises(ValueError, match="QDRANT_SEMANTIC_SEARCH_PREFETCH_LIMIT"):
                get_qdrant_options()

    def test_get_qdrant_options_singleton(self):
        """Test that get_qdrant_options returns a singleton instance."""
        clear_qdrant_options_cache()

        with patch.dict(
            os.environ, {"QDRANT_HOST": "localhost", "QDRANT_COLLECTION_NAME": "test", "QDRANT_VECTOR_SIZE": "768"}
        ):
            options1 = get_qdrant_options()
            options2 = get_qdrant_options()

            assert options1 is options2

    def test_clear_qdrant_options_cache(self):
        """Test that clear_qdrant_options_cache resets the singleton."""
        clear_qdrant_options_cache()

        with patch.dict(
            os.environ, {"QDRANT_HOST": "host1", "QDRANT_COLLECTION_NAME": "collection1", "QDRANT_VECTOR_SIZE": "768"}
        ):
            options1 = get_qdrant_options()
            assert options1.host == "host1"

        clear_qdrant_options_cache()

        with patch.dict(
            os.environ, {"QDRANT_HOST": "host2", "QDRANT_COLLECTION_NAME": "collection2", "QDRANT_VECTOR_SIZE": "768"}
        ):
            options2 = get_qdrant_options()
            assert options2.host == "host2"
            assert options1 is not options2
