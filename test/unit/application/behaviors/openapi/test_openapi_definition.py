import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI

from cezzis_com_cocktails_aisearch.application.behaviors.openapi.openapi_definition import openapi_definition
from cezzis_com_cocktails_aisearch.domain.config.oauth_options import OAuthOptions


class TestOpenApiDefinition:
    """Test cases for openapi_definition function."""

    @patch("cezzis_com_cocktails_aisearch.application.behaviors.openapi.openapi_definition.get_openapi")
    @patch(
        "cezzis_com_cocktails_aisearch.application.behaviors.openapi.openapi_definition.generate_openapi_oauth2_scheme"
    )
    @patch.dict(
        "os.environ",
        {
            "OAUTH_DOMAIN": "auth.example.com",
            "OAUTH_API_AUDIENCE": "api://cocktails",
            "OAUTH_CLIENT_ID": "test-client-id",
            "OAUTH_ISSUER": "https://auth.example.com/",
        },
    )
    def test_openapi_definition_generates_schema(self, mock_generate_scheme, mock_get_openapi):
        """Test that openapi_definition generates and configures OpenAPI schema."""
        app = FastAPI()
        oauth_options = OAuthOptions(
            domain="auth.example.com",
            api_audience="api://cocktails",
            client_id="test-client-id",
            issuer="https://auth.example.com/",
            algorithms=["RS256"],
        )

        mock_openapi_schema = {"openapi": "3.0.0", "info": {"title": "Test", "version": "1.0.0"}, "components": {}}
        mock_get_openapi.return_value = mock_openapi_schema

        mock_security_scheme = {"type": "oauth2", "flows": {"authorizationCode": {}}}
        mock_generate_scheme.return_value = mock_security_scheme

        result = openapi_definition(app, oauth_options)

        # Verify get_openapi was called
        mock_get_openapi.assert_called_once()
        call_kwargs = mock_get_openapi.call_args[1]
        assert call_kwargs["title"] == "Cezzi's Cocktails AI Search API"
        assert call_kwargs["version"] == "1.0.0"

        # Verify security scheme was generated
        mock_generate_scheme.assert_called_once_with(
            name="auth0",
            client_id="test-client-id",
            domain="auth.example.com",
            audience="api://cocktails",
            scopes={"write:embeddings": "Create and update cocktail embeddings"},
            pkce="SHA-256",
        )

        # Verify the result
        assert "components" in result
        assert "securitySchemes" in result["components"]
        assert result["components"]["securitySchemes"] == mock_security_scheme

    @patch("cezzis_com_cocktails_aisearch.application.behaviors.openapi.openapi_definition.get_openapi")
    @patch(
        "cezzis_com_cocktails_aisearch.application.behaviors.openapi.openapi_definition.generate_openapi_oauth2_scheme"
    )
    @patch.dict(
        "os.environ",
        {
            "OAUTH_DOMAIN": "auth.example.com",
            "OAUTH_API_AUDIENCE": "api://cocktails",
            "OAUTH_CLIENT_ID": "test-client-id",
        },
    )
    def test_openapi_definition_sets_app_schema(self, mock_generate_scheme, mock_get_openapi):
        """Test that openapi_definition sets the app's openapi_schema."""
        app = FastAPI()
        oauth_options = OAuthOptions(
            domain="auth.example.com", api_audience="api://cocktails", client_id="test-client-id"
        )

        mock_openapi_schema = {"openapi": "3.0.0", "components": {}}
        mock_get_openapi.return_value = mock_openapi_schema
        mock_generate_scheme.return_value = {"type": "oauth2"}

        result = openapi_definition(app, oauth_options)

        # Verify app's schema was set
        assert app.openapi_schema is not None
        assert app.openapi_schema == result
