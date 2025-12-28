import logging
import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class HuggingFaceOptions(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", f".env.{os.environ.get('ENV')}"), env_file_encoding="utf-8", extra="allow"
    )

    inference_model: str = Field(default="", validation_alias="HUGGINGFACE_INFERENCE_MODEL")
    api_token: str = Field(default="", validation_alias="HUGGINGFACE_API_TOKEN")


_logger: logging.Logger = logging.getLogger("hugging_face_options")

_hf_options: HuggingFaceOptions | None = None


def get_huggingface_options() -> HuggingFaceOptions:
    """Get the singleton instance of HuggingFaceOptions.

    Returns:
        HuggingFaceOptions: The HuggingFace options instance.
    """
    global _hf_options
    if _hf_options is None:
        _hf_options = HuggingFaceOptions()

        # Validate required configuration
        if not _hf_options.inference_model:
            raise ValueError("HUGGINGFACE_INFERENCE_MODEL environment variable is required")
        if not _hf_options.api_token:
            raise ValueError("HUGGINGFACE_API_TOKEN environment variable is required")

        _logger.info("HuggingFace options loaded successfully.")

    return _hf_options


def clear_huggingface_options_cache() -> None:
    """Clear the cached options instance. Useful for testing."""
    global _hf_options
    _hf_options = None
