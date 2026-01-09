import os
from unittest.mock import patch

import pytest

from cezzis_com_cocktails_aisearch.domain.config.otel_options import (
    OTelOptions,
    get_otel_options,
)


class TestOTelOptions:
    """ "Test cases for OTelOptions configuration."""

    def test_otel_options_init_with_defaults(self):
        """Test OTelOptions initialization with default values."""
        with patch.dict(os.environ, {}, clear=True):
            options = OTelOptions()

            assert options.otel_exporter_otlp_endpoint == ""
            assert options.otel_service_name == ""
            assert options.otel_service_namespace == ""
            assert options.otel_otlp_exporter_auth_header == ""
            assert options.enable_console_logging is True
            assert options.enable_tracing is True
            assert options.enable_logging is True

    def test_otel_options_init_with_env_vars(self):
        """Test OTelOptions initialization with environment variables."""
        with patch.dict(
            os.environ,
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel:4318",
                "OTEL_SERVICE_NAME": "cocktails-api",
                "OTEL_SERVICE_NAMESPACE": "cezzis",
                "OTEL_OTLP_AUTH_HEADER": "Bearer token123",
                "OTEL_ENABLE_CONSOLE_LOGGING": "false",
                "OTEL_ENABLE_TRACING": "false",
                "OTEL_ENABLE_LOGGING": "false",
            },
        ):
            options = OTelOptions()

            assert options.otel_exporter_otlp_endpoint == "http://otel:4318"
            assert options.otel_service_name == "cocktails-api"
            assert options.otel_service_namespace == "cezzis"
            assert options.otel_otlp_exporter_auth_header == "Bearer token123"
            assert options.enable_console_logging is False
            assert options.enable_tracing is False
            assert options.enable_logging is False

    def test_get_otel_options_raises_on_missing_endpoint(self):
        """Test that get_otel_options raises ValueError when endpoint is missing."""
        import cezzis_com_cocktails_aisearch.domain.config.otel_options as otel_options_module

        otel_options_module._otel_options = None

        with patch.dict(
            os.environ,
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "",
                "OTEL_SERVICE_NAME": "test",
                "OTEL_SERVICE_NAMESPACE": "test",
                "OTEL_OTLP_AUTH_HEADER": "test",
            },
        ):
            with pytest.raises(ValueError, match="OTEL_EXPORTER_OTLP_ENDPOINT"):
                get_otel_options()

    def test_get_otel_options_raises_on_missing_service_name(self):
        """Test that get_otel_options raises ValueError when service name is missing."""
        import cezzis_com_cocktails_aisearch.domain.config.otel_options as otel_options_module

        otel_options_module._otel_options = None

        with patch.dict(
            os.environ,
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318",
                "OTEL_SERVICE_NAME": "",
                "OTEL_SERVICE_NAMESPACE": "test",
                "OTEL_OTLP_AUTH_HEADER": "test",
            },
        ):
            with pytest.raises(ValueError, match="OTEL_SERVICE_NAME"):
                get_otel_options()

    def test_get_otel_options_singleton(self):
        """Test that get_otel_options returns a singleton instance."""
        import cezzis_com_cocktails_aisearch.domain.config.otel_options as otel_options_module

        otel_options_module._otel_options = None

        with patch.dict(
            os.environ,
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318",
                "OTEL_SERVICE_NAME": "test-service",
                "OTEL_SERVICE_NAMESPACE": "test-namespace",
                "OTEL_OTLP_AUTH_HEADER": "Bearer test",
            },
        ):
            options1 = get_otel_options()
            options2 = get_otel_options()

            assert options1 is options2
            assert options1.otel_service_name == "test-service"
