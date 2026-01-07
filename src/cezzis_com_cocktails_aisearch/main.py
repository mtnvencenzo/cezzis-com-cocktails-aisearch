import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from cezzis_com_cocktails_aisearch.apis.embedding import EmbeddingRouter
from cezzis_com_cocktails_aisearch.apis.scalar_docs import ScalarDocsRouter
from cezzis_com_cocktails_aisearch.apis.semantic_search import SemanticSearchRouter
from cezzis_com_cocktails_aisearch.app_module import create_injector
from cezzis_com_cocktails_aisearch.application.behaviors import global_exception_handler, initialize_opentelemetry
from cezzis_com_cocktails_aisearch.domain.config.app_options import AppOptions
from cezzis_com_cocktails_aisearch.domain.config.auth0_options import Auth0Options

sys.excepthook = global_exception_handler

initialize_opentelemetry()
injector = create_injector()
app_options = injector.get(AppOptions)
auth0_options = injector.get(Auth0Options)

# Configure OAuth2 security scheme for OpenAPI/Scalar docs
oauth2_scheme_config = None
if auth0_options.domain and auth0_options.api_audience:
    oauth2_scheme_config = {
        "auth0": {
            "type": "oauth2",
            "flows": {
                "authorizationCode": {
                    "authorizationUrl": f"https://{auth0_options.domain}/authorize",
                    "tokenUrl": f"https://{auth0_options.domain}/oauth/token",
                    "scopes": {"write:embeddings": "Create and update cocktail embeddings"},
                    "x-usePkce": "SHA-256",
                    "x-defaultClientId": auth0_options.client_id or "",
                }
            },
        }
    }

app = FastAPI()

# Add security schemes to OpenAPI schema
if oauth2_scheme_config:
    app.openapi_schema = None  # Force regeneration

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        from fastapi.openapi.utils import get_openapi

        openapi_schema = get_openapi(
            title="Cezzi's Cocktails AI Search API",
            description="An AI-powered cocktail search API using semantic search and embeddings.",
            version="1.0.0",
            routes=app.routes,
        )
        openapi_schema["components"]["securitySchemes"] = oauth2_scheme_config
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_options.allowed_origins.replace(" ", "").split(",") if app_options.allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(injector.get(SemanticSearchRouter))
app.include_router(injector.get(ScalarDocsRouter))
app.include_router(injector.get(EmbeddingRouter))
