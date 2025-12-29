import logging
import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantOptions(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", f".env.{os.environ.get('ENV')}"), env_file_encoding="utf-8", extra="allow"
    )

    host: str = Field(default="", validation_alias="QDRANT_HOST")
    port: int = Field(default=6333, validation_alias="QDRANT_PORT")
    api_key: str | None = Field(default=None, validation_alias="QDRANT_API_KEY")
    collection_name: str = Field(default="", validation_alias="QDRANT_COLLECTION_NAME")
    vector_size: int = Field(default=0, validation_alias="QDRANT_VECTOR_SIZE")
    use_https: bool = Field(default=True, validation_alias="QDRANT_USE_HTTPS")


_logger: logging.Logger = logging.getLogger("qdrant_options")

_qdrant_options: QdrantOptions | None = None


def get_qdrant_options() -> QdrantOptions:
    """Get the singleton instance of QdrantOptions.

    Returns:
        QdrantOptions: The Qdrant options instance.
    """
    global _qdrant_options
    if _qdrant_options is None:
        _qdrant_options = QdrantOptions()

        # Validate required configuration
        if not _qdrant_options.host:
            raise ValueError("QDRANT_HOST environment variable is required")
        if not _qdrant_options.port:
            raise ValueError("QDRANT_PORT environment variable is required")
        if not _qdrant_options.collection_name:
            raise ValueError("QDRANT_COLLECTION_NAME environment variable is required")
        if not _qdrant_options.vector_size or _qdrant_options.vector_size <= 0:
            raise ValueError("QDRANT_VECTOR_SIZE environment variable is required")

        _logger.info("Qdrant options loaded successfully.")

    return _qdrant_options


def clear_qdrant_options_cache() -> None:
    """Clear the cached options instance. Useful for testing."""
    global _qdrant_options
    _qdrant_options = None
