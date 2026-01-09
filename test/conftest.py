"""Shared test fixtures and helpers for all unit tests."""

import pytest


def create_test_cocktail_model(cocktail_id="test-123", title="Test Cocktail"):
    """Helper function to create a minimal CocktailModel for testing."""
    from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailModel

    return CocktailModel(
        id=cocktail_id,
        title=title,
        descriptiveTitle=f"{title} Description",
        rating=4.5,
        ingredients=[],
        isIba=False,
        serves=1,
        prepTimeMinutes=5,
        searchTiles=[],
        glassware=[],
        search_statistics=None,
    )


@pytest.fixture
def test_cocktail_model():
    """Fixture that provides a test CocktailModel instance."""
    return create_test_cocktail_model()
