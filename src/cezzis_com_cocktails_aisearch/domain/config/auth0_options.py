import logging
import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Auth0Options(BaseSettings):
    """Auth0 OAuth2/OIDC settings loaded from environment variables and .env files."""

    model_config = SettingsConfigDict(
        env_file=(".env", f".env.{os.environ.get('ENV')}"), env_file_encoding="utf-8", extra="allow"
    )

    domain: str = Field(default="", validation_alias="AUTH0_DOMAIN")
    api_audience: str = Field(default="", validation_alias="AUTH0_API_AUDIENCE")
    client_id: str = Field(default="", validation_alias="AUTH0_CLIENT_ID")
    algorithms: list[str] = Field(default_factory=lambda: ["RS256"])
    issuer: str = Field(default="", validation_alias="AUTH0_ISSUER")


_logger: logging.Logger = logging.getLogger("auth0_options")

_auth0_options: Auth0Options | None = None


def get_auth0_options() -> Auth0Options:
    """Get the singleton instance of Auth0Options.

    Returns:
        Auth0Options: The Auth0 options instance.
    """
    global _auth0_options
    if _auth0_options is None:
        _auth0_options = Auth0Options()

        # Validate required configuration
        if not _auth0_options.domain:
            _logger.warning("AUTH0_DOMAIN environment variable is not configured")
        if not _auth0_options.api_audience:
            _logger.warning("AUTH0_API_AUDIENCE environment variable is not configured")
        if not _auth0_options.issuer:
            _logger.warning("AUTH0_ISSUER environment variable is not configured")
        if not _auth0_options.client_id:
            _logger.warning("AUTH0_CLIENT_ID environment variable is not configured")

        _logger.info("Auth0 options loaded successfully.")

    return _auth0_options


def clear_auth0_options_cache() -> None:
    """Clear the cached options instance. Useful for testing."""
    global _auth0_options
    _auth0_options = None
