import os
from unittest.mock import patch

import pytest

from cezzis_com_cocktails_aisearch.domain.config.oauth_options import (
    OAuthOptions,
    clear_oauth_options_cache,
    get_oauth_options,
)


class TestOAuthOptions:
    """Test cases for OAuthOptions configuration."""

    def test_oauth_options_init_with_defaults(self):
        """Test OAuthOptions initialization with default values."""
        with patch.dict(os.environ, {}, clear=True):
            options = OAuthOptions()

            assert options.domain == ""
            assert options.api_audience == ""
            assert options.client_id == ""
            assert options.algorithms == ["RS256"]
            assert options.issuer == ""

    def test_oauth_options_init_with_env_vars(self):
        """Test OAuthOptions initialization with environment variables."""
        with patch.dict(
            os.environ,
            {
                "OAUTH_DOMAIN": "auth.example.com",
                "OAUTH_API_AUDIENCE": "api://cocktails",
                "OAUTH_CLIENT_ID": "test-client-id",
                "OAUTH_ISSUER": "https://auth.example.com/",
            },
        ):
            options = OAuthOptions()

            assert options.domain == "auth.example.com"
            assert options.api_audience == "api://cocktails"
            assert options.client_id == "test-client-id"
            assert options.issuer == "https://auth.example.com/"

    def test_get_oauth_options_singleton(self):
        """Test that get_oauth_options returns a singleton instance."""
        clear_oauth_options_cache()

        with patch.dict(os.environ, {"OAUTH_DOMAIN": "auth.test.com", "OAUTH_API_AUDIENCE": "api://test"}):
            options1 = get_oauth_options()
            options2 = get_oauth_options()

            assert options1 is options2
            assert options1.domain == "auth.test.com"

    def test_get_oauth_options_warnings_for_missing_config(self):
        """Test that warnings are logged for missing configuration."""
        clear_oauth_options_cache()

        with patch.dict(os.environ, {}, clear=True):
            with patch("cezzis_com_cocktails_aisearch.domain.config.oauth_options._logger") as mock_logger:
                _ = get_oauth_options()

                # Should have logged warnings for missing required configs
                assert mock_logger.warning.call_count >= 4

    def test_clear_oauth_options_cache(self):
        """Test that clear_oauth_options_cache resets the singleton."""
        clear_oauth_options_cache()

        with patch.dict(os.environ, {"OAUTH_DOMAIN": "domain1"}):
            options1 = get_oauth_options()
            assert options1.domain == "domain1"

        clear_oauth_options_cache()

        with patch.dict(os.environ, {"OAUTH_DOMAIN": "domain2"}):
            options2 = get_oauth_options()
            assert options2.domain == "domain2"
            assert options1 is not options2
