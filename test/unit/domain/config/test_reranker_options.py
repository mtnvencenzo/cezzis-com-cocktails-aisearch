import os
from unittest.mock import patch

import pytest

from cezzis_com_cocktails_aisearch.domain.config.reranker_options import (
    RerankerOptions,
    clear_reranker_options_cache,
    get_reranker_options,
)


class TestRerankerOptions:
    """Test cases for RerankerOptions configuration."""

    def test_reranker_options_init_with_defaults(self):
        """Test RerankerOptions initialization with default values."""
        with patch.dict(os.environ, {}, clear=True):
            options = RerankerOptions()

            assert options.endpoint == ""
            assert options.api_key == ""
            assert options.score_threshold == 0.0

    def test_reranker_options_init_with_env_vars(self):
        """Test RerankerOptions initialization with environment variables."""
        with patch.dict(
            os.environ,
            {
                "RERANKER_ENDPOINT": "http://localhost:8990",
                "RERANKER_API_KEY": "test-api-key-123",
                "RERANKER_SCORE_THRESHOLD": "0.5",
            },
        ):
            options = RerankerOptions()

            assert options.endpoint == "http://localhost:8990"
            assert options.api_key == "test-api-key-123"
            assert options.score_threshold == 0.5

    def test_get_reranker_options_raises_on_missing_endpoint(self):
        """Test that get_reranker_options raises ValueError when endpoint is missing."""
        clear_reranker_options_cache()

        with patch.dict(
            os.environ,
            {"RERANKER_ENDPOINT": ""},
        ):
            with pytest.raises(ValueError, match="RERANKER_ENDPOINT"):
                get_reranker_options()

    def test_get_reranker_options_with_endpoint_succeeds(self):
        """Test that get_reranker_options with endpoint succeeds."""
        clear_reranker_options_cache()

        with patch.dict(
            os.environ,
            {"RERANKER_ENDPOINT": "http://localhost:8990"},
        ):
            options = get_reranker_options()
            assert options.endpoint == "http://localhost:8990"

    def test_get_reranker_options_singleton(self):
        """Test that get_reranker_options returns a singleton instance."""
        clear_reranker_options_cache()

        with patch.dict(
            os.environ,
            {"RERANKER_ENDPOINT": "http://localhost:8990"},
        ):
            options1 = get_reranker_options()
            options2 = get_reranker_options()

            assert options1 is options2

    def test_clear_reranker_options_cache(self):
        """Test that clear_reranker_options_cache resets the singleton."""
        clear_reranker_options_cache()

        with patch.dict(
            os.environ,
            {"RERANKER_ENDPOINT": "http://first-endpoint"},
        ):
            options1 = get_reranker_options()
            assert options1.endpoint == "http://first-endpoint"

        clear_reranker_options_cache()

        with patch.dict(
            os.environ,
            {"RERANKER_ENDPOINT": "http://second-endpoint"},
        ):
            options2 = get_reranker_options()
            assert options2.endpoint == "http://second-endpoint"
            assert options1 is not options2
