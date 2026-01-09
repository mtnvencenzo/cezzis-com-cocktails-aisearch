import pytest


class TestOtelInit:
    """Test cases for otel __init__ module."""

    def test_module_can_be_imported(self):
        """Test that the otel __init__ module can be imported."""
        import importlib

        module = importlib.import_module("cezzis_com_cocktails_aisearch.application.behaviors.otel")
        assert module is not None
