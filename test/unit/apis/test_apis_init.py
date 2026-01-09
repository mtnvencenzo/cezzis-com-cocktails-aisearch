import pytest


class TestApisInit:
    """Test cases for apis __init__ module."""

    def test_module_can_be_imported(self):
        """Test that the apis __init__ module can be imported."""
        import importlib

        module = importlib.import_module("cezzis_com_cocktails_aisearch.apis")
        assert module is not None
