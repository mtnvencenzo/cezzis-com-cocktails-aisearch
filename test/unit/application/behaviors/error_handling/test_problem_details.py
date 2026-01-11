"""
Test cases for RFC 7807 Problem Details implementation.
"""

import pytest

from cezzis_com_cocktails_aisearch.application.behaviors.error_handling.problem_details import (
    BadRequestException,
    ForbiddenException,
    InternalServerErrorException,
    NotFoundException,
    ProblemDetails,
    ProblemDetailsException,
    UnauthorizedException,
    UnprocessableEntityException,
)


class TestProblemDetails:
    """Test cases for ProblemDetails model."""

    def test_problem_details_creation_with_defaults(self):
        """Test creating ProblemDetails with default values."""
        problem = ProblemDetails(
            title="Test Error",
            status=400,
        )

        assert problem.title == "Test Error"
        assert problem.status == 400
        assert problem.type == "about:blank"
        assert problem.detail is None
        assert problem.instance is None
        assert problem.errors is None

    def test_problem_details_creation_with_all_fields(self):
        """Test creating ProblemDetails with all fields."""
        problem = ProblemDetails(
            type="https://example.com/errors/test",
            title="Test Error",
            status=400,
            detail="Detailed error message",
            instance="/api/test",
            errors={"field1": ["error1", "error2"]},
            extensions={"extra": "data"},
        )

        assert problem.type == "https://example.com/errors/test"
        assert problem.title == "Test Error"
        assert problem.status == 400
        assert problem.detail == "Detailed error message"
        assert problem.instance == "/api/test"
        assert problem.errors == {"field1": ["error1", "error2"]}
        assert problem.extensions == {"extra": "data"}

    def test_problem_details_exclude_none(self):
        """Test that None values are excluded when serializing."""
        problem = ProblemDetails(
            title="Test Error",
            status=400,
        )

        dumped = problem.model_dump(exclude_none=True)
        assert "detail" not in dumped
        assert "instance" not in dumped
        assert "errors" not in dumped
        assert "extensions" not in dumped


class TestProblemDetailsException:
    """Test cases for ProblemDetailsException."""

    def test_problem_details_exception_creation(self):
        """Test creating ProblemDetailsException."""
        exc = ProblemDetailsException(
            status=400,
            title="Bad Request",
            detail="Invalid input",
            instance="/api/test",
        )

        assert exc.status == 400
        assert exc.title == "Bad Request"
        assert exc.detail == "Invalid input"
        assert exc.instance == "/api/test"
        assert str(exc) == "Invalid input"

    def test_problem_details_exception_to_problem_details(self):
        """Test converting exception to ProblemDetails."""
        exc = ProblemDetailsException(
            status=400,
            title="Bad Request",
            detail="Invalid input",
            errors={"field": ["error"]},
        )

        problem = exc.to_problem_details()

        assert isinstance(problem, ProblemDetails)
        assert problem.status == 400
        assert problem.title == "Bad Request"
        assert problem.detail == "Invalid input"
        assert problem.errors == {"field": ["error"]}


class TestBadRequestException:
    """Test cases for BadRequestException."""

    def test_bad_request_exception_defaults(self):
        """Test BadRequestException with default values."""
        exc = BadRequestException()

        assert exc.status == 400
        assert exc.title == "Bad Request"
        assert exc.type == "https://tools.ietf.org/html/rfc7231#section-6.5.1"

    def test_bad_request_exception_with_detail(self):
        """Test BadRequestException with custom detail."""
        exc = BadRequestException(detail="Custom error message")

        assert exc.detail == "Custom error message"
        assert str(exc) == "Custom error message"


class TestUnauthorizedException:
    """Test cases for UnauthorizedException."""

    def test_unauthorized_exception_defaults(self):
        """Test UnauthorizedException with default values."""
        exc = UnauthorizedException()

        assert exc.status == 401
        assert exc.title == "Unauthorized"
        assert exc.type == "https://tools.ietf.org/html/rfc7235#section-3.1"


class TestForbiddenException:
    """Test cases for ForbiddenException."""

    def test_forbidden_exception_defaults(self):
        """Test ForbiddenException with default values."""
        exc = ForbiddenException()

        assert exc.status == 403
        assert exc.title == "Forbidden"
        assert exc.type == "https://tools.ietf.org/html/rfc7231#section-6.5.3"

    def test_forbidden_exception_with_detail(self):
        """Test ForbiddenException with custom detail."""
        exc = ForbiddenException(detail="Access denied")

        assert exc.detail == "Access denied"


class TestNotFoundException:
    """Test cases for NotFoundException."""

    def test_not_found_exception_defaults(self):
        """Test NotFoundException with default values."""
        exc = NotFoundException()

        assert exc.status == 404
        assert exc.title == "Not Found"
        assert exc.type == "https://tools.ietf.org/html/rfc7231#section-6.5.4"


class TestUnprocessableEntityException:
    """Test cases for UnprocessableEntityException."""

    def test_unprocessable_entity_exception_defaults(self):
        """Test UnprocessableEntityException with default values."""
        exc = UnprocessableEntityException()

        assert exc.status == 422
        assert exc.title == "Unprocessable Entity"
        assert exc.type == "https://tools.ietf.org/html/rfc4918#section-11.2"

    def test_unprocessable_entity_exception_with_errors(self):
        """Test UnprocessableEntityException with validation errors."""
        exc = UnprocessableEntityException(
            detail="Validation failed",
            errors={
                "name": ["Field is required"],
                "email": ["Invalid email format"],
            },
        )

        assert exc.errors == {
            "name": ["Field is required"],
            "email": ["Invalid email format"],
        }


class TestInternalServerErrorException:
    """Test cases for InternalServerErrorException."""

    def test_internal_server_error_exception_defaults(self):
        """Test InternalServerErrorException with default values."""
        exc = InternalServerErrorException()

        assert exc.status == 500
        assert exc.title == "Internal Server Error"
        assert exc.type == "https://tools.ietf.org/html/rfc7231#section-6.6.1"

    def test_internal_server_error_exception_with_detail(self):
        """Test InternalServerErrorException with custom detail."""
        exc = InternalServerErrorException(detail="Database connection failed")

        assert exc.detail == "Database connection failed"
