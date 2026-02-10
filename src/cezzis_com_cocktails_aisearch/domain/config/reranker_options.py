import logging
import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RerankerOptions(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", f".env.{os.environ.get('ENV')}"), env_file_encoding="utf-8", extra="allow"
    )

    endpoint: str = Field(default="", validation_alias="RERANKER_ENDPOINT")
    api_key: str = Field(default="", validation_alias="RERANKER_API_KEY")
    score_threshold: float = Field(default=0.0, validation_alias="RERANKER_SCORE_THRESHOLD")
    relative_score_cutoff: float = Field(default=0.0, validation_alias="RERANKER_RELATIVE_SCORE_CUTOFF")


_logger: logging.Logger = logging.getLogger("reranker_options")

_reranker_options: RerankerOptions | None = None


def get_reranker_options() -> RerankerOptions:
    """Get the singleton instance of RerankerOptions.

    Returns:
        RerankerOptions: The reranker options instance.
    """
    global _reranker_options
    if _reranker_options is None:
        _reranker_options = RerankerOptions()

        if not _reranker_options.endpoint:
            raise ValueError("RERANKER_ENDPOINT environment variable is required")
        if _reranker_options.relative_score_cutoff < 0.0 or _reranker_options.relative_score_cutoff > 1.0:
            raise ValueError("RERANKER_RELATIVE_SCORE_CUTOFF must be between 0.0 and 1.0")

        _logger.info(
            "Reranker options loaded successfully.",
        )

    return _reranker_options


def clear_reranker_options_cache() -> None:
    """Clear the cached options instance. Useful for testing."""
    global _reranker_options
    _reranker_options = None
