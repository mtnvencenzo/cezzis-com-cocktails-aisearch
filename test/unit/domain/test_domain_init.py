# Unit tests for domain/__init__.py
import importlib


def test_import_domain_init():
    importlib.import_module("cezzis_com_cocktails_aisearch.domain")
