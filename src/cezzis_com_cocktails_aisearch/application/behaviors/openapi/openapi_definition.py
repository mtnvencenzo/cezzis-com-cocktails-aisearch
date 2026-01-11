from cezzis_oauth import generate_openapi_oauth2_scheme
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from cezzis_com_cocktails_aisearch.domain.config.oauth_options import OAuthOptions


def openapi_definition(app: FastAPI, oauth_options: OAuthOptions) -> dict:
    openapi_schema = get_openapi(
        title="Cezzi's Cocktails AI Search API",
        description="An AI-powered cocktail search API using semantic search and embeddings.",
        version="1.0.0",
        routes=app.routes,
    )

    openapi_schema["components"]["securitySchemes"] = generate_openapi_oauth2_scheme(
        name="auth0",
        client_id=oauth_options.client_id or "",
        domain=oauth_options.domain,
        audience=oauth_options.audience,
        scopes={"write:embeddings": "Create and update cocktail embeddings"},
        pkce="SHA-256",
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema
