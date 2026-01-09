import atexit
from unittest.mock import MagicMock, call, patch

import pytest

from cezzis_com_cocktails_aisearch.application.behaviors.otel.initialize_otel import initialize_opentelemetry


class TestInitializeOtel:
    """Test cases for initialize_opentelemetry function."""

    @patch("cezzis_com_cocktails_aisearch.application.behaviors.otel.initialize_otel.atexit.register")
    @patch("cezzis_com_cocktails_aisearch.application.behaviors.otel.initialize_otel.get_otel_options")
    @patch("cezzis_com_cocktails_aisearch.application.behaviors.otel.initialize_otel.initialize_otel")
    @patch("cezzis_com_cocktails_aisearch.application.behaviors.otel.initialize_otel.version")
    @patch("cezzis_com_cocktails_aisearch.application.behaviors.otel.initialize_otel.socket.gethostname")
    @patch("cezzis_com_cocktails_aisearch.application.behaviors.otel.initialize_otel.RequestsInstrumentor")
    @patch("cezzis_com_cocktails_aisearch.application.behaviors.otel.initialize_otel.HTTPXClientInstrumentor")
    def test_initialize_opentelemetry_success(
        self,
        mock_httpx_instrumentor,
        mock_requests_instrumentor,
        mock_hostname,
        mock_version,
        mock_initialize_otel,
        mock_get_options,
        mock_atexit_register,
    ):
        """Test successful OpenTelemetry initialization."""
        # Setup mocks
        mock_hostname.return_value = "test-host"
        mock_version.return_value = "1.0.0"

        mock_otel_options = MagicMock()
        mock_otel_options.otel_service_name = "test-service"
        mock_otel_options.otel_service_namespace = "test-namespace"
        mock_otel_options.otel_exporter_otlp_endpoint = "http://localhost:4318"
        mock_otel_options.otel_otlp_exporter_auth_header = "Bearer token"
        mock_otel_options.enable_logging = True
        mock_otel_options.enable_tracing = True
        mock_otel_options.enable_console_logging = True
        mock_get_options.return_value = mock_otel_options

        mock_requests_inst = MagicMock()
        mock_httpx_inst = MagicMock()
        mock_requests_instrumentor.return_value = mock_requests_inst
        mock_httpx_instrumentor.return_value = mock_httpx_inst

        with patch.dict("os.environ", {"ENV": "test"}):
            initialize_opentelemetry()

        # Verify atexit was registered
        from cezzis_otel import shutdown_otel

        mock_atexit_register.assert_called_once_with(shutdown_otel)

        # Verify get_otel_options was called
        mock_get_options.assert_called_once()

        # Verify initialize_otel was called with configure_tracing callback
        mock_initialize_otel.assert_called_once()
        call_kwargs = mock_initialize_otel.call_args[1]
        assert "configure_tracing" in call_kwargs
        assert callable(call_kwargs["configure_tracing"])

    @patch("cezzis_com_cocktails_aisearch.application.behaviors.otel.initialize_otel.atexit.register")
    @patch("cezzis_com_cocktails_aisearch.application.behaviors.otel.initialize_otel.get_otel_options")
    @patch("cezzis_com_cocktails_aisearch.application.behaviors.otel.initialize_otel.initialize_otel")
    @patch("cezzis_com_cocktails_aisearch.application.behaviors.otel.initialize_otel.version")
    @patch("cezzis_com_cocktails_aisearch.application.behaviors.otel.initialize_otel.socket.gethostname")
    @patch("cezzis_com_cocktails_aisearch.application.behaviors.otel.initialize_otel.RequestsInstrumentor")
    @patch("cezzis_com_cocktails_aisearch.application.behaviors.otel.initialize_otel.HTTPXClientInstrumentor")
    def test_initialize_opentelemetry_with_settings(
        self,
        mock_httpx_instrumentor,
        mock_requests_instrumentor,
        mock_hostname,
        mock_version,
        mock_initialize_otel,
        mock_get_options,
        mock_atexit_register,
    ):
        """Test that OTelSettings are properly configured."""
        mock_hostname.return_value = "test-host"
        mock_version.return_value = "1.0.0"

        mock_otel_options = MagicMock()
        mock_otel_options.otel_service_name = "cocktails-service"
        mock_otel_options.otel_service_namespace = "cezzis"
        mock_otel_options.otel_exporter_otlp_endpoint = "http://otel:4318"
        mock_otel_options.otel_otlp_exporter_auth_header = "Bearer secret"
        mock_otel_options.enable_logging = False
        mock_otel_options.enable_tracing = True
        mock_otel_options.enable_console_logging = False
        mock_get_options.return_value = mock_otel_options

        mock_requests_instrumentor.return_value = MagicMock()
        mock_httpx_instrumentor.return_value = MagicMock()

        with patch.dict("os.environ", {"ENV": "production"}):
            initialize_opentelemetry()

        # Verify initialize_otel was called with correct settings
        assert mock_initialize_otel.called
        call_args = mock_initialize_otel.call_args
        settings = call_args[1]["settings"]

        assert settings.service_name == "cocktails-service"
        assert settings.service_namespace == "cezzis"
        assert settings.otlp_exporter_endpoint == "http://otel:4318"
        assert settings.enable_logging is False
        assert settings.enable_tracing is True
        assert settings.enable_console_logging is False
