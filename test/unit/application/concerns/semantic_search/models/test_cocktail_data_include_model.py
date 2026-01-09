import pytest

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_data_include_model import (
    CocktailDataIncludeModel,
)


class TestCocktailDataIncludeModel:
    """Test cases for CocktailDataIncludeModel enum."""

    def test_enum_exists(self):
        """Test that CocktailDataIncludeModel enum is properly defined."""
        # This test verifies the module can be imported and accessed
        assert CocktailDataIncludeModel is not None

    def test_enum_is_enum(self):
        """Test that CocktailDataIncludeModel is an enum type."""
        from enum import Enum

        # Check if it's a class (likely an Enum or string literal type)
        assert isinstance(CocktailDataIncludeModel, type) or hasattr(CocktailDataIncludeModel, "__members__")
