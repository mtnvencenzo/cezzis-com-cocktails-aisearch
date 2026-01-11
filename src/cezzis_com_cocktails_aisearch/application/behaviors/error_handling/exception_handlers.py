"""
FastAPI exception handlers for converting exceptions to RFC 7807 problem details responses.
"""

import logging

from fastapi import Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from cezzis_com_cocktails_aisearch.application.behaviors.error_handling.problem_details import (
    ProblemDetails,
    ProblemDetailsException,
)

_logger = logging.getLogger("exception_handlers")


async def problem_details_exception_handler(request: Request, exc: ProblemDetailsException) -> JSONResponse:
    """
    Handle ProblemDetailsException and return RFC 7807 compliant JSON response.
    """
    problem_details = exc.to_problem_details()
    if not problem_details.instance:
        problem_details.instance = str(request.url.path)

    _logger.warning(
        "Problem details exception occurred",
        extra={
            "status": problem_details.status,
            "title": problem_details.title,
            "detail": problem_details.detail,
            "path": problem_details.instance,
        },
    )

    return JSONResponse(
        status_code=problem_details.status,
        content=problem_details.model_dump(exclude_none=True),
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handle FastAPI HTTPException and convert to RFC 7807 problem details format.
    """
    # Map status codes to RFC 7807 types and titles
    status_info = {
        400: ("https://tools.ietf.org/html/rfc7231#section-6.5.1", "Bad Request"),
        401: ("https://tools.ietf.org/html/rfc7235#section-3.1", "Unauthorized"),
        403: ("https://tools.ietf.org/html/rfc7231#section-6.5.3", "Forbidden"),
        404: ("https://tools.ietf.org/html/rfc7231#section-6.5.4", "Not Found"),
        422: ("https://tools.ietf.org/html/rfc4918#section-11.2", "Unprocessable Entity"),
        500: ("https://tools.ietf.org/html/rfc7231#section-6.6.1", "Internal Server Error"),
    }

    status_type, title = status_info.get(exc.status_code, ("about:blank", "Error"))

    problem_details = ProblemDetails(
        type=status_type,
        title=title,
        status=exc.status_code,
        detail=str(exc.detail) if exc.detail else None,
        instance=str(request.url.path),
    )

    _logger.warning(
        "HTTP exception occurred",
        extra={
            "status": exc.status_code,
            "detail": exc.detail,
            "path": str(request.url.path),
        },
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=problem_details.model_dump(exclude_none=True),
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError | ValidationError) -> JSONResponse:
    """
    Handle Pydantic validation errors and convert to RFC 7807 problem details format.
    """
    # Extract validation errors from Pydantic
    errors: dict[str, list[str]] = {}

    if isinstance(exc, RequestValidationError):
        validation_errors = exc.errors()
    else:
        validation_errors = exc.errors()

    for error in validation_errors:
        # Build field path from location tuple
        field_path = ".".join(str(loc) for loc in error["loc"] if loc not in ("body", "query"))
        if not field_path:
            field_path = "request"

        error_message = error["msg"]

        if field_path not in errors:
            errors[field_path] = []

        errors[field_path].append(error_message)

    problem_details = ProblemDetails(
        type="https://tools.ietf.org/html/rfc4918#section-11.2",
        title="Validation Error",
        status=422,
        detail="One or more validation errors occurred.",
        instance=str(request.url.path),
        errors=errors,
    )

    _logger.warning(
        "Validation error occurred",
        extra={
            "path": str(request.url.path),
            "errors": errors,
        },
    )

    return JSONResponse(
        status_code=422,
        content=problem_details.model_dump(exclude_none=True),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle any unhandled exceptions and convert to RFC 7807 problem details format.
    """
    problem_details = ProblemDetails(
        type="https://tools.ietf.org/html/rfc7231#section-6.6.1",
        title="Internal Server Error",
        status=500,
        detail="An unexpected error occurred while processing the request.",
        instance=str(request.url.path),
    )

    _logger.exception(
        "Unhandled exception occurred",
        extra={
            "path": str(request.url.path),
            "exception_type": type(exc).__name__,
        },
        exc_info=exc,
    )

    return JSONResponse(
        status_code=500,
        content=problem_details.model_dump(exclude_none=True),
    )
