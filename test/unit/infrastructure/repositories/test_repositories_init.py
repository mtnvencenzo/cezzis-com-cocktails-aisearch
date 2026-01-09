import pytest


class TestInfrastructureRepositoriesInit:
    """Test cases for infrastructure/repositories __init__ module."""

    def test_module_can_be_imported(self):
        """Test that the infrastructure/repositories __init__ module can be imported."""
        import importlib

        module = importlib.import_module("cezzis_com_cocktails_aisearch.infrastructure.repositories")
        assert module is not None
