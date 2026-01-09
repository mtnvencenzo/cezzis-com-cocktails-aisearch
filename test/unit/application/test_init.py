# Unit tests for application/__init__.py
import importlib


def test_import_application_init():
    # Try importing the application module
    importlib.import_module("cezzis_com_cocktails_aisearch.application")
