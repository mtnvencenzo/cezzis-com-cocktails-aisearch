import logging
import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OTelOptions(BaseSettings):
    """Opentelemetry settings loaded from environment variables and .env files.

    Attributes:
        otel_exporter_otlp_endpoint (str): OpenTelemetry OTLP exporter endpoint.
        otel_service_name (str): OpenTelemetry service name.
        otel_service_namespace (str): OpenTelemetry service namespace.
        otel_otlp_exporter_auth_header (str): OpenTelemetry OTLP exporter authorization header
        enable_console_logging (bool): Flag to enable console logging.
        enable_tracing (bool): Flag to enable tracing.
        enable_logging (bool): Flag to enable logging.
    """

    model_config = SettingsConfigDict(
        env_file=(".env", f".env.{os.environ.get('ENV')}"), env_file_encoding="utf-8", extra="allow"
    )

    otel_exporter_otlp_endpoint: str = Field(default="", validation_alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_service_name: str = Field(default="", validation_alias="OTEL_SERVICE_NAME")
    otel_service_namespace: str = Field(default="", validation_alias="OTEL_SERVICE_NAMESPACE")
    otel_otlp_exporter_auth_header: str = Field(default="", validation_alias="OTEL_OTLP_AUTH_HEADER")
    enable_console_logging: bool = Field(default=True, validation_alias="OTEL_ENABLE_CONSOLE_LOGGING")
    enable_tracing: bool = Field(default=True, validation_alias="OTEL_ENABLE_TRACING")
    enable_logging: bool = Field(default=True, validation_alias="OTEL_ENABLE_LOGGING")


_logger: logging.Logger = logging.getLogger("otel_options")

_otel_options: OTelOptions | None = None


def get_otel_options() -> OTelOptions:
    """Get the singleton instance of OTelOptions.

    Returns:
        OTelOptions: The OpenTelemetry options instance.
    """

    global _otel_options
    if _otel_options is None:
        _otel_options = OTelOptions()

        # Validate required configuration
        if not _otel_options.otel_exporter_otlp_endpoint:
            raise ValueError("OTEL_EXPORTER_OTLP_ENDPOINT environment variable is required")
        if not _otel_options.otel_service_name:
            raise ValueError("OTEL_SERVICE_NAME environment variable is required")
        if not _otel_options.otel_service_namespace:
            raise ValueError("OTEL_SERVICE_NAMESPACE environment variable is required")
        if not _otel_options.otel_otlp_exporter_auth_header:
            raise ValueError("OTEL_OTLP_AUTH_HEADER environment variable is required")

        _logger.info("OpenTelemetry options loaded successfully.")

    return _otel_options
