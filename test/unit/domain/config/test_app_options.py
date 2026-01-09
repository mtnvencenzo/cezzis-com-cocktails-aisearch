import os
from unittest.mock import patch

import pytest

from cezzis_com_cocktails_aisearch.domain.config.app_options import (
    AppOptions,
    get_app_options,
)


class TestAppOptions:
    """Test cases for AppOptions configuration."""

    def test_app_options_init_with_defaults(self):
        """Test AppOptions initialization with default values."""
        with patch.dict(os.environ, {}, clear=True):
            options = AppOptions()

            assert options.allowed_origins == ""
            assert options.apim_host_key == ""

    def test_app_options_init_with_env_vars(self):
        """Test AppOptions initialization with environment variables."""
        with patch.dict(
            os.environ,
            {
                "COCKTAILS_AISEARCH_ALLOWED_ORIGINS": "http://localhost:3000,https://example.com",
                "APIM_HOST_KEY": "test-key-123",
            },
        ):
            options = AppOptions()

            assert options.allowed_origins == "http://localhost:3000,https://example.com"
            assert options.apim_host_key == "test-key-123"

    def test_get_app_options_singleton(self):
        """Test that get_app_options returns a singleton instance."""
        # Clear any existing singleton
        import cezzis_com_cocktails_aisearch.domain.config.app_options as app_options_module

        app_options_module._app_options = None

        with patch.dict(os.environ, {"APIM_HOST_KEY": "singleton-test"}):
            options1 = get_app_options()
            options2 = get_app_options()

            assert options1 is options2
            assert options1.apim_host_key == "singleton-test"

    def test_get_app_options_caching(self):
        """Test that get_app_options caches the instance."""
        import cezzis_com_cocktails_aisearch.domain.config.app_options as app_options_module

        app_options_module._app_options = None

        with patch.dict(os.environ, {"APIM_HOST_KEY": "cached-key"}):
            options1 = get_app_options()

        # Change env var after first call
        with patch.dict(os.environ, {"APIM_HOST_KEY": "new-key"}):
            options2 = get_app_options()

            # Should still return the cached instance with old value
            assert options1 is options2
            assert options2.apim_host_key == "cached-key"
