import pytest


class TestExceptionHandlingInit:
    """Test cases for exception_handling __init__ module."""

    def test_module_can_be_imported(self):
        """Test that the exception_handling __init__ module can be imported."""
        import importlib

        module = importlib.import_module("cezzis_com_cocktails_aisearch.application.behaviors.exception_handling")
        assert module is not None
