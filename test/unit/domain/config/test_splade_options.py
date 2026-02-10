import os
from unittest.mock import patch

import pytest

from cezzis_com_cocktails_aisearch.domain.config.splade_options import (
    SpladeOptions,
    clear_splade_options_cache,
    get_splade_options,
)


class TestSpladeOptions:
    """Test cases for SpladeOptions configuration."""

    def test_splade_options_init_with_defaults(self):
        """Test SpladeOptions initialization with default values."""
        with patch.dict(os.environ, {}, clear=True):
            options = SpladeOptions()

            assert options.endpoint == ""
            assert options.api_key == ""

    def test_splade_options_init_with_env_vars(self):
        """Test SpladeOptions initialization with environment variables."""
        with patch.dict(
            os.environ,
            {
                "SPLADE_ENDPOINT": "http://localhost:8991",
                "SPLADE_API_KEY": "test-api-key-123",
            },
        ):
            options = SpladeOptions()

            assert options.endpoint == "http://localhost:8991"
            assert options.api_key == "test-api-key-123"

    def test_get_splade_options_raises_on_missing_endpoint(self):
        """Test that get_splade_options raises ValueError when endpoint is missing."""
        clear_splade_options_cache()

        with patch.dict(
            os.environ,
            {"SPLADE_ENDPOINT": ""},
        ):
            with pytest.raises(ValueError, match="SPLADE_ENDPOINT"):
                get_splade_options()

    def test_get_splade_options_with_endpoint_succeeds(self):
        """Test that get_splade_options with endpoint succeeds."""
        clear_splade_options_cache()

        with patch.dict(
            os.environ,
            {"SPLADE_ENDPOINT": "http://localhost:8991"},
        ):
            options = get_splade_options()
            assert options.endpoint == "http://localhost:8991"

    def test_get_splade_options_singleton(self):
        """Test that get_splade_options returns a singleton instance."""
        clear_splade_options_cache()

        with patch.dict(
            os.environ,
            {"SPLADE_ENDPOINT": "http://localhost:8991"},
        ):
            options1 = get_splade_options()
            options2 = get_splade_options()

            assert options1 is options2

    def test_clear_splade_options_cache(self):
        """Test that clear_splade_options_cache resets the singleton."""
        clear_splade_options_cache()

        with patch.dict(
            os.environ,
            {"SPLADE_ENDPOINT": "http://first-endpoint"},
        ):
            options1 = get_splade_options()
            assert options1.endpoint == "http://first-endpoint"

        clear_splade_options_cache()

        with patch.dict(
            os.environ,
            {"SPLADE_ENDPOINT": "http://second-endpoint"},
        ):
            options2 = get_splade_options()
            assert options2.endpoint == "http://second-endpoint"
            assert options1 is not options2
