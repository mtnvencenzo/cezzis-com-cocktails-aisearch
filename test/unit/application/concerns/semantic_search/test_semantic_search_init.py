# Unit tests for application/concerns/semantic_search/__init__.py
import importlib


def test_import_semantic_search_init():
    importlib.import_module("cezzis_com_cocktails_aisearch.application.concerns.semantic_search")
