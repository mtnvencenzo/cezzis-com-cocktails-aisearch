"""Common OpenAPI definitions and constants."""

# OpenAPI parameter definition for X-Key header
# Used to document the API gateway subscription key as required in OpenAPI
# while keeping it optional at runtime (validated by apim_host_key_authorization decorator)
X_KEY_OPENAPI_PARAMETER: dict = {
    "name": "X-Key",
    "in": "header",
    "required": True,
    "description": "The API gateway subscription key",
    "schema": {"type": "string"},
}


def create_openapi_extra(
    parameters: list[dict] | None = None,
    security: list[dict] | None = None,
) -> dict:
    """
    Create an openapi_extra dict with common X-Key parameter.

    Args:
        parameters: Additional parameters to include (X-Key is always added)
        security: Security requirements to include

    Returns:
        OpenAPI extra configuration dict
    """
    all_parameters = [X_KEY_OPENAPI_PARAMETER]
    if parameters:
        all_parameters.extend(parameters)

    extra: dict = {"parameters": all_parameters}

    if security:
        extra["security"] = security

    return extra
