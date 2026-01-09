import sys
from unittest.mock import MagicMock, patch

import pytest

from cezzis_com_cocktails_aisearch.application.behaviors.exception_handling.global_exception_handler import (
    global_exception_handler,
)


class TestGlobalExceptionHandler:
    """Test cases for global_exception_handler function."""

    def test_global_exception_handler_logs_exception(self):
        """Test that exception is logged."""
        exc_type = ValueError
        exc_value = ValueError("Test error")
        tb = None

        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            with patch("sys.__excepthook__") as mock_excepthook:
                global_exception_handler(exc_type, exc_value, tb)

                mock_logger.exception.assert_called_once()
                mock_excepthook.assert_called_once_with(exc_type, exc_value, tb)

    def test_global_exception_handler_calls_default_hook(self):
        """Test that default exception hook is called."""
        exc_type = RuntimeError
        exc_value = RuntimeError("Runtime error")
        tb = None

        with patch("logging.getLogger"):
            with patch("sys.__excepthook__") as mock_excepthook:
                global_exception_handler(exc_type, exc_value, tb)

                mock_excepthook.assert_called_once_with(exc_type, exc_value, tb)

    def test_global_exception_handler_with_traceback(self):
        """Test exception handler with a real traceback."""
        try:
            raise ValueError("Test exception")
        except ValueError:
            exc_type, exc_value, tb = sys.exc_info()

        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            with patch("sys.__excepthook__") as mock_excepthook:
                global_exception_handler(exc_type, exc_value, tb)

                # Check that logging was called with exception info
                call_args = mock_logger.exception.call_args
                assert call_args[0][0] == "An unhandled exception occurred"
                assert "exc_info" in call_args[1]

                mock_excepthook.assert_called_once_with(exc_type, exc_value, tb)
