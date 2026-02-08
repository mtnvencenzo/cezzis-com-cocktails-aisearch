from typing import Any

from cezzis_oauth import generate_openapi_oauth2_scheme
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from cezzis_com_cocktails_aisearch.domain.config.oauth_options import OAuthOptions


def _convert_to_openapi_3_0(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively convert OpenAPI 3.1.0 format to 3.0.1 format.

    Handles:
    - anyOf: [{type: x}, {type: null}] -> type: x, nullable: true
    - $ref with nullable: true -> allOf wrapper
    - examples (array) -> example (first item) for schema properties
    """
    if isinstance(schema, dict):
        # Check if this is an anyOf with null type pattern
        if "anyOf" in schema:
            any_of = schema["anyOf"]
            if isinstance(any_of, list) and len(any_of) == 2:
                # Check if one is null type
                non_null_schemas = [s for s in any_of if s.get("type") != "null"]
                null_schemas = [s for s in any_of if s.get("type") == "null"]

                if len(null_schemas) == 1 and len(non_null_schemas) == 1:
                    non_null_schema = non_null_schemas[0]
                    # Remove anyOf from current schema
                    schema = {k: v for k, v in schema.items() if k != "anyOf"}

                    # If the non-null schema is a $ref, wrap in allOf for 3.0 compatibility
                    if "$ref" in non_null_schema:
                        schema["allOf"] = [non_null_schema]
                        schema["nullable"] = True
                    else:
                        # Convert to nullable format
                        base_schema = non_null_schema.copy()
                        base_schema["nullable"] = True
                        schema.update(base_schema)

        # Fix direct $ref with nullable: true (invalid in OpenAPI 3.0)
        if "$ref" in schema and "nullable" in schema and schema.get("nullable") is True:
            ref_value = schema.pop("$ref")
            schema.pop("nullable")
            schema["allOf"] = [{"$ref": ref_value}]
            schema["nullable"] = True

        # Convert 'examples' (3.1) to 'example' (3.0) for schema properties
        # In OpenAPI 3.0, schema objects use 'example' (singular), not 'examples'
        if "examples" in schema and isinstance(schema["examples"], list):
            examples_list = schema.pop("examples")
            if examples_list:
                schema["example"] = examples_list[0]

        # Recursively process all values in the dict
        for key, value in list(schema.items()):
            if isinstance(value, dict):
                schema[key] = _convert_to_openapi_3_0(value)
            elif isinstance(value, list):
                schema[key] = [_convert_to_openapi_3_0(item) if isinstance(item, dict) else item for item in value]

    return schema


def openapi_definition(app: FastAPI, oauth_options: OAuthOptions) -> dict:
    openapi_schema = get_openapi(
        title="Cezzi's Cocktails AI Search API",
        description="An AI-powered cocktail search API using semantic search and embeddings.",
        version="1.0.0",
        openapi_version="3.0.1",
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

    # Convert nullable types to OpenAPI 3.0.1 format
    openapi_schema = _convert_to_openapi_3_0(openapi_schema)

    app.openapi_schema = openapi_schema
    return app.openapi_schema
