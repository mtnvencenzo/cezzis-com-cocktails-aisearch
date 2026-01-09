# Unit tests for application/concerns/__init__.py
import importlib


def test_import_concerns_init():
    importlib.import_module("cezzis_com_cocktails_aisearch.application.concerns")
