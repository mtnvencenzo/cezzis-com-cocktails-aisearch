import logging
import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OAuthOptions(BaseSettings):
    """OAuth OAuth2/OIDC settings loaded from environment variables and .env files."""

    model_config = SettingsConfigDict(
        env_file=(".env", f".env.{os.environ.get('ENV')}"), env_file_encoding="utf-8", extra="allow"
    )

    domain: str = Field(default="", validation_alias="OAUTH_DOMAIN")
    api_audience: str = Field(default="", validation_alias="OAUTH_API_AUDIENCE")
    client_id: str = Field(default="", validation_alias="OAUTH_CLIENT_ID")
    algorithms: list[str] = Field(default_factory=lambda: ["RS256"])
    issuer: str = Field(default="", validation_alias="OAUTH_ISSUER")


_logger: logging.Logger = logging.getLogger("oauth_options")

_oauth_options: OAuthOptions | None = None


def get_oauth_options() -> OAuthOptions:
    """Get the singleton instance of OAuthOptions.
    Returns:
        OAuthOptions: The OAuth options instance.
    """
    global _oauth_options
    if _oauth_options is None:
        _oauth_options = OAuthOptions()
        # Validate required configuration
        if not _oauth_options.domain:
            _logger.warning("OAUTH_DOMAIN environment variable is not configured")
        if not _oauth_options.api_audience:
            _logger.warning("OAUTH_API_AUDIENCE environment variable is not configured")
        if not _oauth_options.issuer:
            _logger.warning("OAUTH_ISSUER environment variable is not configured")
        if not _oauth_options.client_id:
            _logger.warning("OAUTH_CLIENT_ID environment variable is not configured")

        _logger.info("OAuth options loaded successfully.")

    return _oauth_options


def clear_oauth_options_cache() -> None:
    """Clear the cached options instance. Useful for testing."""
    global _oauth_options
    _oauth_options = None
