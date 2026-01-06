import logging
import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppOptions(BaseSettings):
    """Application settings loaded from environment variables and .env files."""

    model_config = SettingsConfigDict(
        env_file=(".env", f".env.{os.environ.get('ENV')}"), env_file_encoding="utf-8", extra="allow"
    )

    allowed_origins: str = Field(default="", validation_alias="COCKTAILS_AISEARCH_ALLOWED_ORIGINS")
    apim_host_key: str = Field(default="", validation_alias="APIM_HOST_KEY")


_logger: logging.Logger = logging.getLogger("app_options")

_app_options: AppOptions | None = None


def get_app_options() -> AppOptions:
    """Get the singleton instance of AppOptions.
    Returns:
        AppOptions: The application options instance.
    """

    global _app_options
    if _app_options is None:
        _app_options = AppOptions()

        _logger.info("Application options loaded successfully.")

    return _app_options
