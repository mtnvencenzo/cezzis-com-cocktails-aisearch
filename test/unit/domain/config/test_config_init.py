import pytest


class TestDomainConfigInit:
    """Test cases for domain/config __init__ module."""

    def test_module_can_be_imported(self):
        """Test that the domain/config __init__ module can be imported."""
        import importlib

        module = importlib.import_module("cezzis_com_cocktails_aisearch.domain.config")
        assert module is not None
