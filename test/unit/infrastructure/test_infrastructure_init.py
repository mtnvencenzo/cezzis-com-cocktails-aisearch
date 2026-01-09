# Unit tests for infrastructure/__init__.py
import importlib


def test_import_infrastructure_init():
    importlib.import_module("cezzis_com_cocktails_aisearch.infrastructure")
