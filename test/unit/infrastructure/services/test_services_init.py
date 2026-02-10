import pytest


class TestInfrastructureServicesInit:
    """Test cases for infrastructure/services __init__ module."""

    def test_module_can_be_imported(self):
        """Test that the infrastructure/services __init__ module can be imported."""
        import importlib

        module = importlib.import_module("cezzis_com_cocktails_aisearch.infrastructure.services")
        assert module is not None

    def test_exports_ireranker_service(self):
        """Test that IRerankerService is exported."""
        from cezzis_com_cocktails_aisearch.infrastructure.services import IRerankerService

        assert IRerankerService is not None

    def test_exports_reranker_service(self):
        """Test that RerankerService is exported."""
        from cezzis_com_cocktails_aisearch.infrastructure.services import RerankerService

        assert RerankerService is not None

    def test_exports_isplade_service(self):
        """Test that ISpladeService is exported."""
        from cezzis_com_cocktails_aisearch.infrastructure.services import ISpladeService

        assert ISpladeService is not None

    def test_exports_splade_service(self):
        """Test that SpladeService is exported."""
        from cezzis_com_cocktails_aisearch.infrastructure.services import SpladeService

        assert SpladeService is not None
