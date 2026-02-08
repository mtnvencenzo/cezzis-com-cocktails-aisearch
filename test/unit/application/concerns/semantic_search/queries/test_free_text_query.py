from unittest.mock import AsyncMock, MagicMock

import pytest
from conftest import create_test_cocktail_model
from qdrant_client.http.models import FieldCondition, Filter, MatchValue, Range

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
        """Test successful query handling with semantic search."""
        mock_repository = AsyncMock()
        mock_cocktail1 = create_test_cocktail_model("1", "Margarita")
        mock_cocktail1.search_statistics = CocktailSearchStatistics(
            total_score=0.9, max_score=0.9, avg_score=0.9, weighted_score=0.9, hit_count=1, hit_results=[]
        )
        mock_cocktail2 = create_test_cocktail_model("2", "Mojito")
        mock_cocktail2.search_statistics = CocktailSearchStatistics(
            total_score=0.8, max_score=0.8, avg_score=0.8, weighted_score=0.8, hit_count=1, hit_results=[]
        )

        mock_repository.search_vectors = AsyncMock(return_value=[mock_cocktail1, mock_cocktail2])
        mock_repository.get_all_cocktails = AsyncMock(return_value=[mock_cocktail1, mock_cocktail2])

        mock_qdrant_options = MagicMock()

        handler = FreeTextQueryHandler(cocktail_vector_repository=mock_repository, qdrant_opotions=mock_qdrant_options)

        query = FreeTextQuery(free_text="tequila")
        result = await handler.handle(query)

        assert len(result) == 2
        assert result[0].title == "Margarita"  # Higher score first
        assert result[1].title == "Mojito"
        mock_repository.search_vectors.assert_called_once_with(free_text="tequila", query_filter=None)

    @pytest.mark.anyio
    async def test_handler_sorts_by_score(self):
        """Test that results are sorted by weighted_score descending."""
        mock_repository = AsyncMock()

        mock_cocktail1 = create_test_cocktail_model("1", "Low Score")
        mock_cocktail1.search_statistics = CocktailSearchStatistics(
            total_score=0.5, max_score=0.5, avg_score=0.5, weighted_score=0.5, hit_count=1, hit_results=[]
        )

        mock_cocktail2 = create_test_cocktail_model("2", "High Score")
        mock_cocktail2.search_statistics = CocktailSearchStatistics(
            total_score=0.9, max_score=0.9, avg_score=0.9, weighted_score=0.9, hit_count=1, hit_results=[]
        )

        mock_cocktail3 = create_test_cocktail_model("3", "Medium Score")
        mock_cocktail3.search_statistics = CocktailSearchStatistics(
            total_score=0.7, max_score=0.7, avg_score=0.7, weighted_score=0.7, hit_count=1, hit_results=[]
        )

        # Return in unsorted order
        mock_repository.search_vectors = AsyncMock(return_value=[mock_cocktail1, mock_cocktail2, mock_cocktail3])
        mock_repository.get_all_cocktails = AsyncMock(return_value=[mock_cocktail1, mock_cocktail2, mock_cocktail3])

        mock_qdrant_options = MagicMock()

        handler = FreeTextQueryHandler(cocktail_vector_repository=mock_repository, qdrant_opotions=mock_qdrant_options)

        query = FreeTextQuery(free_text="test query")
        result = await handler.handle(query)

        assert len(result) == 3
        assert result[0].title == "High Score"
        assert result[1].title == "Medium Score"
        assert result[2].title == "Low Score"

    @pytest.mark.anyio
    async def test_handler_handles_empty_results(self):
        """Test handler with empty search results."""
        mock_repository = AsyncMock()
        mock_repository.search_vectors = AsyncMock(return_value=[])
        mock_repository.get_all_cocktails = AsyncMock(return_value=[])

        mock_qdrant_options = MagicMock()

        handler = FreeTextQueryHandler(cocktail_vector_repository=mock_repository, qdrant_opotions=mock_qdrant_options)

        query = FreeTextQuery(free_text="nonexistent")
        result = await handler.handle(query)

        assert len(result) == 0

    @pytest.mark.anyio
    async def test_handler_with_empty_free_text(self):
        """Test handler with empty free text calls get_all_cocktails."""
        mock_repository = AsyncMock()
        mock_cocktail1 = create_test_cocktail_model("1", "Cocktail A")
        mock_cocktail2 = create_test_cocktail_model("2", "Cocktail B")
        mock_repository.get_all_cocktails = AsyncMock(return_value=[mock_cocktail2, mock_cocktail1])

        mock_qdrant_options = MagicMock()

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

    @pytest.mark.anyio
    async def test_handler_exact_name_match(self):
        """Test that exact cocktail name match short-circuits semantic search."""
        mock_repository = AsyncMock()
        mock_cocktail = create_test_cocktail_model("1", "Margarita")
        mock_repository.get_all_cocktails = AsyncMock(return_value=[mock_cocktail])

        mock_qdrant_options = MagicMock()

        handler = FreeTextQueryHandler(cocktail_vector_repository=mock_repository, qdrant_opotions=mock_qdrant_options)

        query = FreeTextQuery(free_text="Margarita")
        result = await handler.handle(query)

        assert len(result) == 1
        assert result[0].title == "Margarita"
        mock_repository.search_vectors.assert_not_called()

    @pytest.mark.anyio
    async def test_handler_short_query_fallback(self):
        """Test that short queries use text-based matching instead of semantic search."""
        mock_repository = AsyncMock()
        mock_cocktail1 = create_test_cocktail_model("1", "Rum Runner")
        mock_cocktail2 = create_test_cocktail_model("2", "Mojito")
        mock_repository.get_all_cocktails = AsyncMock(return_value=[mock_cocktail1, mock_cocktail2])

        mock_qdrant_options = MagicMock()

        handler = FreeTextQueryHandler(cocktail_vector_repository=mock_repository, qdrant_opotions=mock_qdrant_options)

        query = FreeTextQuery(free_text="rum")
        result = await handler.handle(query)

        # Should match "Rum Runner" via text search, not semantic search
        assert len(result) == 1
        assert result[0].title == "Rum Runner"
        mock_repository.search_vectors.assert_not_called()

    @pytest.mark.anyio
    async def test_handler_passes_iba_filter_to_repository(self):
        """Test that IBA filter is built and passed to search_vectors."""
        mock_repository = AsyncMock()
        mock_repository.search_vectors = AsyncMock(return_value=[])
        mock_repository.get_all_cocktails = AsyncMock(return_value=[])

        mock_qdrant_options = MagicMock()

        handler = FreeTextQueryHandler(cocktail_vector_repository=mock_repository, qdrant_opotions=mock_qdrant_options)

        query = FreeTextQuery(free_text="iba cocktail recipes")
        await handler.handle(query)

        call_kwargs = mock_repository.search_vectors.call_args[1]
        query_filter = call_kwargs["query_filter"]
        assert query_filter is not None
        assert isinstance(query_filter.must, list)
        assert len(query_filter.must) == 1
        assert isinstance(query_filter.must[0], FieldCondition)
        assert query_filter.must[0].key == "metadata.is_iba"
        assert query_filter.must[0].match is not None
        assert isinstance(query_filter.must[0].match, MatchValue)
        assert query_filter.must[0].match.value is True

    @pytest.mark.anyio
    async def test_handler_passes_exclusion_filter_to_repository(self):
        """Test that ingredient exclusion filter is built and passed to search_vectors."""
        mock_repository = AsyncMock()
        mock_repository.search_vectors = AsyncMock(return_value=[])
        mock_repository.get_all_cocktails = AsyncMock(return_value=[])

        mock_qdrant_options = MagicMock()

        handler = FreeTextQueryHandler(cocktail_vector_repository=mock_repository, qdrant_opotions=mock_qdrant_options)

        query = FreeTextQuery(free_text="cocktails without honey")
        await handler.handle(query)

        call_kwargs = mock_repository.search_vectors.call_args[1]
        query_filter = call_kwargs["query_filter"]
        assert query_filter is not None
        assert query_filter.must_not is not None
        assert isinstance(query_filter.must_not, list)
        assert len(query_filter.must_not) == 1
        assert isinstance(query_filter.must_not[0], FieldCondition)
        assert query_filter.must_not[0].key == "metadata.ingredient_words"
        assert query_filter.must_not[0].match is not None
        assert isinstance(query_filter.must_not[0].match, MatchValue)
        assert query_filter.must_not[0].match.value == "honey"


class TestBuildQueryFilter:
    """Test cases for _build_query_filter."""

    def _make_handler(self):
        mock_repository = AsyncMock()
        mock_qdrant_options = MagicMock()
        return FreeTextQueryHandler(cocktail_vector_repository=mock_repository, qdrant_opotions=mock_qdrant_options)

    def test_returns_none_for_plain_query(self):
        """Test that a plain query without structured elements returns None."""
        handler = self._make_handler()
        result = handler._build_query_filter("tequila cocktails")
        assert result is None

    def test_iba_filter(self):
        """Test IBA filter detection."""
        handler = self._make_handler()
        result = handler._build_query_filter("iba cocktail recipes")
        assert result is not None
        assert isinstance(result.must, list)
        assert len(result.must) == 1
        assert isinstance(result.must[0], FieldCondition)
        assert result.must[0].key == "metadata.is_iba"
        assert result.must[0].match is not None
        assert isinstance(result.must[0].match, MatchValue)
        assert result.must[0].match.value is True

    def test_non_iba_filter(self):
        """Test non-IBA filter detection."""
        handler = self._make_handler()
        result = handler._build_query_filter("non-iba cocktails")
        assert result is not None
        assert isinstance(result.must, list)
        assert len(result.must) == 1
        assert isinstance(result.must[0], FieldCondition)
        assert result.must[0].key == "metadata.is_iba"
        assert result.must[0].match is not None
        assert isinstance(result.must[0].match, MatchValue)
        assert result.must[0].match.value is False

    def test_glassware_filter(self):
        """Test glassware filter detection."""
        handler = self._make_handler()
        result = handler._build_query_filter("cocktails served in a coupe")
        assert result is not None
        assert isinstance(result.must, list)
        assert len(result.must) == 1
        assert isinstance(result.must[0], FieldCondition)
        assert result.must[0].key == "metadata.glassware_values"
        assert result.must[0].match is not None
        assert isinstance(result.must[0].match, MatchValue)
        assert result.must[0].match.value == "coupe"

    def test_simple_ingredient_count_filter(self):
        """Test simple/easy ingredient count filter."""
        handler = self._make_handler()
        result = handler._build_query_filter("simple cocktails")
        assert result is not None
        assert isinstance(result.must, list)
        assert len(result.must) == 1
        assert isinstance(result.must[0], FieldCondition)
        assert result.must[0].key == "metadata.ingredient_count"
        assert result.must[0].range is not None
        assert result.must[0].range.lte == 4

    def test_complex_ingredient_count_filter(self):
        """Test complex ingredient count filter."""
        handler = self._make_handler()
        result = handler._build_query_filter("complex cocktails")
        assert result is not None
        assert isinstance(result.must, list)
        assert len(result.must) == 1
        assert isinstance(result.must[0], FieldCondition)
        assert result.must[0].key == "metadata.ingredient_count"
        assert result.must[0].range is not None
        assert result.must[0].range.gte == 6

    def test_numeric_ingredient_count_filter(self):
        """Test numeric ingredient count filter (e.g., '3 ingredient')."""
        handler = self._make_handler()
        result = handler._build_query_filter("3 ingredient cocktails")
        assert result is not None
        assert isinstance(result.must, list)
        assert len(result.must) == 1
        assert isinstance(result.must[0], FieldCondition)
        assert result.must[0].key == "metadata.ingredient_count"
        assert result.must[0].range is not None
        assert result.must[0].range.gte == 3
        assert result.must[0].range.lte == 3

    def test_prep_time_filter(self):
        """Test quick/fast prep time filter."""
        handler = self._make_handler()
        result = handler._build_query_filter("quick cocktails")
        assert result is not None
        assert isinstance(result.must, list)
        assert len(result.must) == 1
        assert isinstance(result.must[0], FieldCondition)
        assert result.must[0].key == "metadata.prep_time_minutes"
        assert result.must[0].range is not None
        assert result.must[0].range.lte == 5

    def test_serves_filter(self):
        """Test serves filter."""
        handler = self._make_handler()
        result = handler._build_query_filter("cocktail serves 2")
        assert result is not None
        assert isinstance(result.must, list)
        assert len(result.must) == 1
        assert isinstance(result.must[0], FieldCondition)
        assert result.must[0].key == "metadata.serves"
        assert result.must[0].match is not None
        assert isinstance(result.must[0].match, MatchValue)
        assert result.must[0].match.value == 2

    def test_exclusion_filter(self):
        """Test ingredient exclusion via must_not."""
        handler = self._make_handler()
        result = handler._build_query_filter("cocktails without rum")
        assert result is not None
        assert result.must_not is not None
        assert isinstance(result.must_not, list)
        assert len(result.must_not) == 1
        assert isinstance(result.must_not[0], FieldCondition)
        assert result.must_not[0].key == "metadata.ingredient_words"
        assert result.must_not[0].match is not None
        assert isinstance(result.must_not[0].match, MatchValue)
        assert result.must_not[0].match.value == "rum"

    def test_combined_filters(self):
        """Test that multiple filter conditions combine correctly."""
        handler = self._make_handler()
        result = handler._build_query_filter("simple iba cocktails without rum")
        assert result is not None
        # Should have IBA + ingredient_count in must
        assert isinstance(result.must, list)
        assert len(result.must) == 2
        assert all(isinstance(c, FieldCondition) for c in result.must)
        must_keys = {c.key for c in result.must if isinstance(c, FieldCondition)}
        assert "metadata.is_iba" in must_keys
        assert "metadata.ingredient_count" in must_keys
        # Should have rum exclusion in must_not
        assert result.must_not is not None
        assert isinstance(result.must_not, list)
        assert len(result.must_not) == 1
        assert isinstance(result.must_not[0], FieldCondition)
        assert result.must_not[0].match is not None
        assert isinstance(result.must_not[0].match, MatchValue)
        assert result.must_not[0].match.value == "rum"


class TestExtractExclusionTerms:
    """Test cases for _extract_exclusion_terms."""

    def _make_handler(self):
        mock_repository = AsyncMock()
        mock_qdrant_options = MagicMock()
        return FreeTextQueryHandler(cocktail_vector_repository=mock_repository, qdrant_opotions=mock_qdrant_options)

    def test_without_pattern(self):
        handler = self._make_handler()
        terms = handler._extract_exclusion_terms("cocktails without honey")
        assert terms == ["honey"]

    def test_no_pattern(self):
        handler = self._make_handler()
        terms = handler._extract_exclusion_terms("cocktails no rum")
        assert terms == ["rum"]

    def test_multiple_exclusions(self):
        handler = self._make_handler()
        terms = handler._extract_exclusion_terms("cocktails without honey no rum")
        assert "honey" in terms
        assert "rum" in terms

    def test_no_exclusion_in_plain_query(self):
        handler = self._make_handler()
        terms = handler._extract_exclusion_terms("tequila cocktails")
        assert terms == []

    def test_short_terms_ignored(self):
        handler = self._make_handler()
        terms = handler._extract_exclusion_terms("cocktails no a")
        assert terms == []
