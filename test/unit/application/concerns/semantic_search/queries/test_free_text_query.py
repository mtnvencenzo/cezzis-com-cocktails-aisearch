from unittest.mock import AsyncMock, MagicMock

import pytest
from conftest import create_test_cocktail_model

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailModel
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_search_statistics import (
    CocktailSearchStatistics,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.queries.free_text_query import (
    FreeTextQuery,
    FreeTextQueryHandler,
    FreeTextQueryValidator,
)


class TestFreeTextQuery:
    """Test cases for FreeTextQuery."""

    def test_query_init_with_defaults(self):
        """Test query initialization with default values."""
        query = FreeTextQuery()

        assert query.free_text == ""
        assert query.skip == 0
        assert query.take == 10
        assert query.matches == []
        assert query.match_exclusive is False
        assert query.include == []
        assert query.filters == []

    def test_query_init_with_custom_values(self):
        """Test query initialization with custom values."""
        query = FreeTextQuery(
            free_text="tequila cocktails",
            skip=10,
            take=20,
            matches=["margarita"],
            match_exclusive=True,
            include=[],
            filters=["filter1"],
        )

        assert query.free_text == "tequila cocktails"
        assert query.skip == 10
        assert query.take == 20
        assert query.matches == ["margarita"]
        assert query.match_exclusive is True
        assert query.filters == ["filter1"]


class TestFreeTextQueryValidator:
    """Test cases for FreeTextQueryValidator."""

    def test_validator_always_passes(self):
        """Test that validator always calls next (no validation logic)."""
        query = FreeTextQuery(free_text="test")
        validator = FreeTextQueryValidator()
        next_mock = MagicMock()

        validator.handle(query, next_mock)

        next_mock.assert_called_once()


class TestFreeTextQueryHandler:
    """Test cases for FreeTextQueryHandler."""

    @pytest.mark.anyio
    async def test_handler_success(self):
        """Test successful query handling."""
        mock_repository = AsyncMock()
        mock_cocktail1 = create_test_cocktail_model("1", "Margarita")
        mock_cocktail1.search_statistics = CocktailSearchStatistics(total_score=0.9, hit_results=[])
        mock_cocktail2 = create_test_cocktail_model("2", "Mojito")
        mock_cocktail2.search_statistics = CocktailSearchStatistics(total_score=0.8, hit_results=[])

        mock_repository.search_vectors = AsyncMock(return_value=[mock_cocktail1, mock_cocktail2])

        mock_qdrant_options = MagicMock()
        mock_qdrant_options.semantic_search_total_score_threshold = 0.5

        handler = FreeTextQueryHandler(cocktail_vector_repository=mock_repository, qdrant_opotions=mock_qdrant_options)

        query = FreeTextQuery(free_text="tequila")
        result = await handler.handle(query)

        assert len(result) == 2
        assert result[0].title == "Margarita"  # Higher score first
        assert result[1].title == "Mojito"
        mock_repository.search_vectors.assert_called_once_with(free_text="tequila")

    @pytest.mark.anyio
    async def test_handler_sorts_by_score(self):
        """Test that results are sorted by total_score descending."""
        mock_repository = AsyncMock()

        mock_cocktail1 = create_test_cocktail_model("1", "Low Score")
        mock_cocktail1.search_statistics = CocktailSearchStatistics(total_score=0.5, hit_results=[])

        mock_cocktail2 = create_test_cocktail_model("2", "High Score")
        mock_cocktail2.search_statistics = CocktailSearchStatistics(total_score=0.9, hit_results=[])

        mock_cocktail3 = create_test_cocktail_model("3", "Medium Score")
        mock_cocktail3.search_statistics = CocktailSearchStatistics(total_score=0.7, hit_results=[])

        # Return in unsorted order
        mock_repository.search_vectors = AsyncMock(return_value=[mock_cocktail1, mock_cocktail2, mock_cocktail3])

        mock_qdrant_options = MagicMock()
        mock_qdrant_options.semantic_search_total_score_threshold = 0.0

        handler = FreeTextQueryHandler(cocktail_vector_repository=mock_repository, qdrant_opotions=mock_qdrant_options)

        query = FreeTextQuery(free_text="test")
        result = await handler.handle(query)

        assert len(result) == 3
        assert result[0].title == "High Score"
        assert result[1].title == "Medium Score"
        assert result[2].title == "Low Score"

    @pytest.mark.anyio
    async def test_handler_filters_by_threshold(self):
        """Test that results below threshold are filtered out."""
        mock_repository = AsyncMock()

        mock_cocktail1 = create_test_cocktail_model("1", "High Score")
        mock_cocktail1.search_statistics = CocktailSearchStatistics(total_score=0.9, hit_results=[])

        mock_cocktail2 = create_test_cocktail_model("2", "Low Score")
        mock_cocktail2.search_statistics = CocktailSearchStatistics(total_score=0.3, hit_results=[])

        mock_repository.search_vectors = AsyncMock(return_value=[mock_cocktail1, mock_cocktail2])

        mock_qdrant_options = MagicMock()
        mock_qdrant_options.semantic_search_total_score_threshold = 0.5

        handler = FreeTextQueryHandler(cocktail_vector_repository=mock_repository, qdrant_opotions=mock_qdrant_options)

        query = FreeTextQuery(free_text="test")
        result = await handler.handle(query)

        # Only cocktail above threshold
        assert len(result) == 1
        assert result[0].title == "High Score"

    @pytest.mark.anyio
    async def test_handler_handles_empty_results(self):
        """Test handler with empty search results."""
        mock_repository = AsyncMock()
        mock_repository.search_vectors = AsyncMock(return_value=[])

        mock_qdrant_options = MagicMock()
        mock_qdrant_options.semantic_search_total_score_threshold = 0.5

        handler = FreeTextQueryHandler(cocktail_vector_repository=mock_repository, qdrant_opotions=mock_qdrant_options)

        query = FreeTextQuery(free_text="nonexistent")
        result = await handler.handle(query)

        assert len(result) == 0

    @pytest.mark.anyio
    async def test_handler_handles_none_search_statistics(self):
        """Test handler filters out cocktails without search statistics."""
        mock_repository = AsyncMock()

        mock_cocktail1 = create_test_cocktail_model("1", "With Stats")
        mock_cocktail1.search_statistics = CocktailSearchStatistics(total_score=0.9, hit_results=[])

        mock_cocktail2 = create_test_cocktail_model("2", "Without Stats")
        mock_cocktail2.search_statistics = CocktailSearchStatistics(total_score=0.0, hit_results=[])

        mock_repository.search_vectors = AsyncMock(return_value=[mock_cocktail1, mock_cocktail2])

        mock_qdrant_options = MagicMock()
        mock_qdrant_options.semantic_search_total_score_threshold = 0.5

        handler = FreeTextQueryHandler(cocktail_vector_repository=mock_repository, qdrant_opotions=mock_qdrant_options)

        query = FreeTextQuery(free_text="test")
        result = await handler.handle(query)

        # Only cocktail with search statistics
        assert len(result) == 1
        assert result[0].title == "With Stats"

    @pytest.mark.anyio
    async def test_handler_with_empty_free_text(self):
        """Test handler with empty free text calls get_all_cocktails."""
        mock_repository = AsyncMock()
        mock_cocktail1 = create_test_cocktail_model("1", "Cocktail A")
        mock_cocktail2 = create_test_cocktail_model("2", "Cocktail B")
        mock_repository.get_all_cocktails = AsyncMock(return_value=[mock_cocktail2, mock_cocktail1])

        mock_qdrant_options = MagicMock()
        mock_qdrant_options.semantic_search_total_score_threshold = 0.0

        handler = FreeTextQueryHandler(cocktail_vector_repository=mock_repository, qdrant_opotions=mock_qdrant_options)

        query = FreeTextQuery(free_text=None, skip=0, take=10)
        result = await handler.handle(query)

        # Should call get_all_cocktails, not search_vectors
        mock_repository.get_all_cocktails.assert_called_once()
        mock_repository.search_vectors.assert_not_called()

        # Results should be sorted by title and paginated
        assert len(result) == 2
        assert result[0].title == "Cocktail A"  # Alphabetically first
        assert result[1].title == "Cocktail B"
