import logging
import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SpladeOptions(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", f".env.{os.environ.get('ENV')}"), env_file_encoding="utf-8", extra="allow"
    )

    endpoint: str = Field(default="", validation_alias="SPLADE_ENDPOINT")
    api_key: str = Field(default="", validation_alias="SPLADE_API_KEY")


_logger: logging.Logger = logging.getLogger("splade_options")

_splade_options: SpladeOptions | None = None


def get_splade_options() -> SpladeOptions:
    """Get the singleton instance of SpladeOptions.

    Returns:
        SpladeOptions: The SPLADE options instance.
    """
    global _splade_options
    if _splade_options is None:
        _splade_options = SpladeOptions()

        if not _splade_options.endpoint:
            raise ValueError("SPLADE_ENDPOINT environment variable is required")

        _logger.info(
            "SPLADE options loaded successfully.",
        )

    return _splade_options


def clear_splade_options_cache() -> None:
    """Clear the cached options instance. Useful for testing."""
    global _splade_options
    _splade_options = None
