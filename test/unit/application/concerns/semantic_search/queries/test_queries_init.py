# Unit tests for application/concerns/semantic_search/queries/__init__.py
import importlib


def test_import_queries_init():
    importlib.import_module("cezzis_com_cocktails_aisearch.application.concerns.semantic_search.queries")
