import logging
import os
from functools import wraps
from typing import Callable, Union, cast

from fastapi import HTTPException, Request

from cezzis_com_cocktails_aisearch.application.behaviors.oauth_authorization.oauth_verification import (
    TokenVerificationError,
    get_token_verifier,
)

_logger = logging.getLogger("oauth_authorization")


def oauth_authorization(scopes: list[str] | None = None):
    """Decorator for OAuth2 authorization with OAuth token verification.

    Can be applied to:
    - Individual endpoint methods (async functions)
    - APIRouter classes (all methods will be protected)

    Args:
        scopes: Optional list of required OAuth scopes. If None, only validates token.

    Example:
        # On a method
        @oauth_authorization(scopes=["write:embeddings"])
        async def get_cocktails(self, _rq: Request):
            ...

        # On a class (protects all methods)
        @oauth_authorization(scopes=["admin:cocktails"])
        class AdminRouter(APIRouter):
            ...

        # Without scope verification
        @oauth_authorization()
        async def public_endpoint(self, _rq: Request):
            ...
    """
    required_scopes = scopes or []

    def decorator(target: Union[Callable, type]) -> Union[Callable, type]:
        # Check if target is a class
        if isinstance(target, type):
            return _wrap_class(target, required_scopes)
        else:
            return _wrap_function(target, required_scopes)

    return decorator


def _wrap_function(func: Callable, required_scopes: list[str]) -> Callable:
    """Wrap an individual async function with OAuth authorization."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract the Request object from kwargs
        request: Request = cast(Request, kwargs.get("_rq"))

        if not request:
            _logger.error("Request object not found in function arguments")
            raise HTTPException(status_code=500, detail="Internal server error")

        # Check if authorization should be bypassed in local environment
        if os.getenv("ENV") == "local":
            _logger.info("OAuth authorization bypassed in local environment")
            return await func(*args, **kwargs)

        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization", "")
        if not auth_header or not auth_header.startswith("Bearer "):
            _logger.warning("Missing or invalid Authorization header")
            raise HTTPException(status_code=401, detail="Missing or invalid authorization token")

        token = auth_header.replace("Bearer ", "").strip()

        try:
            # Verify the token
            verifier = get_token_verifier()
            payload = await verifier.verify_token(token)

            # Verify scopes if required
            if required_scopes:
                verifier.verify_scopes(payload, required_scopes)

            _logger.info(f"OAuth authorization successful for subject: {payload.get('sub', 'unknown')}")

            # Call the original function
            return await func(*args, **kwargs)

        except TokenVerificationError as e:
            _logger.warning(f"OAuth authorization failed: {e}")
            raise HTTPException(status_code=403, detail=str(e))
        except Exception as e:
            _logger.error(f"Unexpected error during OAuth authorization: {e}")
            raise HTTPException(status_code=500, detail="Authorization error")

    return wrapper


def _wrap_class(cls: type, required_scopes: list[str]) -> type:
    """Wrap all async methods in a class with OAuth authorization."""

    # Get all methods from the class
    for attr_name in dir(cls):
        # Skip private/magic methods and non-callables
        if attr_name.startswith("_"):
            continue

        attr = getattr(cls, attr_name)

        # Check if it's a callable method (not a property or static attribute)
        if callable(attr) and not isinstance(attr, (staticmethod, classmethod, property)):
            # Wrap the method
            wrapped = _wrap_function(attr, required_scopes)
            setattr(cls, attr_name, wrapped)

    return cls
