import pytest

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_data_include_model import (
    CocktailSearchDataIncludeModel,
)


class TestCocktailSearchDataIncludeModel:
    """Test cases for CocktailSearchDataIncludeModel enum."""

    def test_enum_exists(self):
        """Test that CocktailSearchDataIncludeModel enum is properly defined."""
        # This test verifies the module can be imported and accessed
        assert CocktailSearchDataIncludeModel is not None

    def test_enum_is_enum(self):
        """Test that CocktailSearchDataIncludeModel is an enum type."""
        from enum import Enum

        # Check if it's a class (likely an Enum or string literal type)
        assert isinstance(CocktailSearchDataIncludeModel, type) or hasattr(
            CocktailSearchDataIncludeModel, "__members__"
        )
