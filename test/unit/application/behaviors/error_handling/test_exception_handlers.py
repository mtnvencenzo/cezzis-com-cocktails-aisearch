"""
Test cases for FastAPI exception handlers.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError

from cezzis_com_cocktails_aisearch.application.behaviors.error_handling.exception_handlers import (
    generic_exception_handler,
    http_exception_handler,
    problem_details_exception_handler,
    validation_exception_handler,
)
from cezzis_com_cocktails_aisearch.application.behaviors.error_handling.problem_details import (
    BadRequestException,
    ProblemDetailsException,
)


class TestProblemDetailsExceptionHandler:
    """Test cases for problem_details_exception_handler."""

    @pytest.mark.anyio
    async def test_handler_converts_exception_to_response(self):
        """Test that handler converts ProblemDetailsException to JSONResponse."""
        mock_request = MagicMock()
        mock_request.url.path = "/api/test"

        exc = BadRequestException(detail="Test error")

        response = await problem_details_exception_handler(mock_request, exc)

        assert response.status_code == 400
        body = bytes(response.body).decode("utf-8")
        assert "Bad Request" in body
        assert "Test error" in body
        assert "/api/test" in body

    @pytest.mark.anyio
    async def test_handler_sets_instance_from_request(self):
        """Test that handler sets instance from request path."""
        mock_request = MagicMock()
        mock_request.url.path = "/v1/cocktails/search"

        exc = ProblemDetailsException(
            status=404,
            title="Not Found",
            detail="Resource not found",
        )

        response = await problem_details_exception_handler(mock_request, exc)

        body = bytes(response.body).decode("utf-8")
        assert "/v1/cocktails/search" in body


class TestHttpExceptionHandler:
    """Test cases for http_exception_handler."""

    @pytest.mark.anyio
    async def test_handler_converts_http_exception(self):
        """Test that handler converts HTTPException to problem details."""
        mock_request = MagicMock()
        mock_request.url.path = "/api/test"

        exc = HTTPException(status_code=403, detail="Access denied")

        response = await http_exception_handler(mock_request, exc)

        assert response.status_code == 403
        body = bytes(response.body).decode("utf-8")
        assert "Forbidden" in body
        assert "Access denied" in body
        assert "/api/test" in body

    @pytest.mark.anyio
    async def test_handler_handles_different_status_codes(self):
        """Test handler with various HTTP status codes."""
        mock_request = MagicMock()
        mock_request.url.path = "/api/test"

        test_cases = [
            (400, "Bad Request"),
            (401, "Unauthorized"),
            (403, "Forbidden"),
            (404, "Not Found"),
            (422, "Unprocessable Entity"),
            (500, "Internal Server Error"),
        ]

        for status_code, expected_title in test_cases:
            exc = HTTPException(status_code=status_code)
            response = await http_exception_handler(mock_request, exc)

            assert response.status_code == status_code
            body = bytes(response.body).decode("utf-8")
            assert expected_title in body


class TestValidationExceptionHandler:
    """Test cases for validation_exception_handler."""

    @pytest.mark.anyio
    async def test_handler_converts_request_validation_error(self):
        """Test that handler converts RequestValidationError to problem details."""
        mock_request = MagicMock()
        mock_request.url.path = "/api/test"

        # Create a mock validation error
        errors = [
            {
                "loc": ("body", "field1"),
                "msg": "Field is required",
                "type": "value_error.missing",
            }
        ]

        exc = RequestValidationError(errors=errors)

        response = await validation_exception_handler(mock_request, exc)

        assert response.status_code == 422
        body = bytes(response.body).decode("utf-8")
        assert "Validation Error" in body
        assert "field1" in body
        assert "Field is required" in body

    @pytest.mark.anyio
    async def test_handler_handles_multiple_field_errors(self):
        """Test handler with multiple field validation errors."""
        mock_request = MagicMock()
        mock_request.url.path = "/api/test"

        errors = [
            {
                "loc": ("body", "email"),
                "msg": "Invalid email format",
                "type": "value_error.email",
            },
            {
                "loc": ("body", "age"),
                "msg": "Must be positive",
                "type": "value_error.number",
            },
        ]

        exc = RequestValidationError(errors=errors)

        response = await validation_exception_handler(mock_request, exc)

        body = bytes(response.body).decode("utf-8")
        assert "email" in body
        assert "age" in body
        assert "Invalid email format" in body
        assert "Must be positive" in body

    @pytest.mark.anyio
    async def test_handler_handles_nested_field_errors(self):
        """Test handler with nested field validation errors."""
        mock_request = MagicMock()
        mock_request.url.path = "/api/test"

        errors = [
            {
                "loc": ("body", "user", "profile", "name"),
                "msg": "Field is required",
                "type": "value_error.missing",
            }
        ]

        exc = RequestValidationError(errors=errors)

        response = await validation_exception_handler(mock_request, exc)

        body = bytes(response.body).decode("utf-8")
        assert "user.profile.name" in body


class TestGenericExceptionHandler:
    """Test cases for generic_exception_handler."""

    @pytest.mark.anyio
    async def test_handler_converts_generic_exception(self):
        """Test that handler converts generic Exception to problem details."""
        mock_request = MagicMock()
        mock_request.url.path = "/api/test"

        exc = Exception("Unexpected error")

        response = await generic_exception_handler(mock_request, exc)

        assert response.status_code == 500
        body = bytes(response.body).decode("utf-8")
        assert "Internal Server Error" in body
        assert "An unexpected error occurred" in body
        assert "/api/test" in body

    @pytest.mark.anyio
    async def test_handler_handles_different_exception_types(self):
        """Test handler with various exception types."""
        mock_request = MagicMock()
        mock_request.url.path = "/api/test"

        exceptions = [
            ValueError("Invalid value"),
            KeyError("Missing key"),
            RuntimeError("Runtime error"),
        ]

        for exc in exceptions:
            response = await generic_exception_handler(mock_request, exc)

            assert response.status_code == 500
            body = bytes(response.body).decode("utf-8")
            assert "Internal Server Error" in body
