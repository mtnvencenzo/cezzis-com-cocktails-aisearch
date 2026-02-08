"""Shared test fixtures and helpers for all unit tests."""

import pytest

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_embedding_model import (
    CocktailEmbeddingModel,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailSearchModel
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_search_statistics import (
    CocktailSearchStatistics,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_vector_search_result import (
    CocktailVectorSearchResult,
)


def create_test_cocktail_model(cocktail_id="test-123", title="Test Cocktail"):
    """Helper function to create a minimal CocktailModel for testing."""

    return CocktailSearchModel(
        id=cocktail_id,
        title=title,
        descriptive_title=f"{title} Description",
        rating=4.5,
        ingredients=[],
        is_iba=False,
        serves=1,
        prep_time_minutes=5,
        search_tiles=[],
        glassware=[],
        search_statistics=CocktailSearchStatistics(
            total_score=1.0,
            hit_results=[
                CocktailVectorSearchResult(score=1.0),
                CocktailVectorSearchResult(score=0.8),
            ],
            max_score=1.0,
            avg_score=0.9,
            weighted_score=0.95,
            hit_count=2,
        ),
    )


def create_test_cocktail_embedding_model(cocktail_id="test-123", title="Test Cocktail"):
    """Helper function to create a minimal CocktailModel for testing."""

    return CocktailEmbeddingModel(
        id=cocktail_id,
        title=title,
        descriptive_title=f"{title} Description",
        rating=4.5,
        ingredients=[],
        is_iba=False,
        serves=1,
        prep_time_minutes=5,
        search_tiles=[],
        glassware=[],
    )


@pytest.fixture
def test_cocktail_model():
    """Fixture that provides a test CocktailModel instance."""
    return create_test_cocktail_model()
