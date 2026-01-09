import pytest


class TestOpenApiInit:
    """Test cases for openapi __init__ module."""

    def test_module_can_be_imported(self):
        """Test that the openapi __init__ module can be imported."""
        import importlib

        module = importlib.import_module("cezzis_com_cocktails_aisearch.application.behaviors.openapi")
        assert module is not None
