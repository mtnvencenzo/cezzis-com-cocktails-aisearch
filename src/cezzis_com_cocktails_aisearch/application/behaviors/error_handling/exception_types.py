from typing import Any

from cezzis_com_cocktails_aisearch.application.behaviors.error_handling.problem_details import ProblemDetails


class ProblemDetailsException(Exception):
    """Base exception for problem details responses."""

    def __init__(
        self,
        status: int,
        title: str,
        detail: str | None = None,
        type: str = "about:blank",
        instance: str | None = None,
        errors: dict[str, list[str]] | None = None,
        extensions: dict[str, Any] | None = None,
    ):
        self.status = status
        self.title = title
        self.detail = detail
        self.type = type
        self.instance = instance
        self.errors = errors
        self.extensions = extensions
        super().__init__(detail or title)

    def to_problem_details(self) -> ProblemDetails:
        """Convert exception to ProblemDetails model."""
        return ProblemDetails(
            type=self.type,
            title=self.title,
            status=self.status,
            detail=self.detail,
            instance=self.instance,
            errors=self.errors,
            extensions=self.extensions,
        )


class BadRequestException(ProblemDetailsException):
    """Exception for 400 Bad Request responses."""

    def __init__(
        self,
        detail: str | None = None,
        title: str = "Bad Request",
        type: str = "https://tools.ietf.org/html/rfc7231#section-6.5.1",
        instance: str | None = None,
        errors: dict[str, list[str]] | None = None,
        extensions: dict[str, Any] | None = None,
    ):
        super().__init__(
            status=400,
            title=title,
            detail=detail,
            type=type,
            instance=instance,
            errors=errors,
            extensions=extensions,
        )


class UnauthorizedException(ProblemDetailsException):
    """Exception for 401 Unauthorized responses."""

    def __init__(
        self,
        detail: str | None = None,
        title: str = "Unauthorized",
        type: str = "https://tools.ietf.org/html/rfc7235#section-3.1",
        instance: str | None = None,
        errors: dict[str, list[str]] | None = None,
        extensions: dict[str, Any] | None = None,
    ):
        super().__init__(
            status=401,
            title=title,
            detail=detail,
            type=type,
            instance=instance,
            errors=errors,
            extensions=extensions,
        )


class ForbiddenException(ProblemDetailsException):
    """Exception for 403 Forbidden responses."""

    def __init__(
        self,
        detail: str | None = None,
        title: str = "Forbidden",
        type: str = "https://tools.ietf.org/html/rfc7231#section-6.5.3",
        instance: str | None = None,
        errors: dict[str, list[str]] | None = None,
        extensions: dict[str, Any] | None = None,
    ):
        super().__init__(
            status=403,
            title=title,
            detail=detail,
            type=type,
            instance=instance,
            errors=errors,
            extensions=extensions,
        )


class NotFoundException(ProblemDetailsException):
    """Exception for 404 Not Found responses."""

    def __init__(
        self,
        detail: str | None = None,
        title: str = "Not Found",
        type: str = "https://tools.ietf.org/html/rfc7231#section-6.5.4",
        instance: str | None = None,
        errors: dict[str, list[str]] | None = None,
        extensions: dict[str, Any] | None = None,
    ):
        super().__init__(
            status=404,
            title=title,
            detail=detail,
            type=type,
            instance=instance,
            errors=errors,
            extensions=extensions,
        )


class UnprocessableEntityException(ProblemDetailsException):
    """Exception for 422 Unprocessable Entity responses."""

    def __init__(
        self,
        detail: str | None = None,
        title: str = "Unprocessable Entity",
        type: str = "https://tools.ietf.org/html/rfc4918#section-11.2",
        instance: str | None = None,
        errors: dict[str, list[str]] | None = None,
        extensions: dict[str, Any] | None = None,
    ):
        super().__init__(
            status=422,
            title=title,
            detail=detail,
            type=type,
            instance=instance,
            errors=errors,
            extensions=extensions,
        )


class InternalServerErrorException(ProblemDetailsException):
    """Exception for 500 Internal Server Error responses."""

    def __init__(
        self,
        detail: str | None = None,
        title: str = "Internal Server Error",
        type: str = "https://tools.ietf.org/html/rfc7231#section-6.6.1",
        instance: str | None = None,
        errors: dict[str, list[str]] | None = None,
        extensions: dict[str, Any] | None = None,
    ):
        super().__init__(
            status=500,
            title=title,
            detail=detail,
            type=type,
            instance=instance,
            errors=errors,
            extensions=extensions,
        )
