import atexit
import logging
import os
import socket
from importlib.metadata import version

from cezzis_otel import OTelSettings, initialize_otel, shutdown_otel
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

from cezzis_com_cocktails_aisearch.domain.config import get_otel_options


def initialize_opentelemetry() -> None:
    """Initialize OpenTelemetry tracing and logging for the application."""

    # Make sure to shutdown and gracefully flush the telemetry data on exit
    atexit.register(shutdown_otel)

    # Suppress urllib3 debug logs (used by OTLP exporter) to prevent self-logging
    logging.getLogger("urllib3").setLevel(logging.INFO)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

    otel_options = get_otel_options()

    initialize_otel(
        settings=OTelSettings(
            service_name=otel_options.otel_service_name,
            service_namespace=otel_options.otel_service_namespace,
            otlp_exporter_endpoint=otel_options.otel_exporter_otlp_endpoint,
            otlp_exporter_auth_header=otel_options.otel_otlp_exporter_auth_header,
            service_version=version("cezzis_com_cocktails_aisearch"),
            environment=os.environ.get("ENV", "unknown"),
            instance_id=socket.gethostname(),
            enable_logging=otel_options.enable_logging,
            enable_tracing=otel_options.enable_tracing,
            enable_console_logging=otel_options.enable_console_logging,
        ),
        resource_attributes={
            "app_name": otel_options.otel_service_name,
            "app_class": "api",
            "app_product": "cezzis.com",
            "app_product_segment": "backend",
            "app_unit": "cocktails",
            "app_env": os.environ.get("ENV", "unknown"),
        },
        configure_tracing=lambda _: (
            RequestsInstrumentor().instrument(),
            HTTPXClientInstrumentor().instrument(),
            None,
        )[-1],
    )

    logger = logging.getLogger("initialize_otel")
    logger.info("OpenTelemetry initialized successfully")
