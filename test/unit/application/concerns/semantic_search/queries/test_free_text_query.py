from unittest.mock import AsyncMock, MagicMock

import pytest
from conftest import create_test_cocktail_model
from qdrant_client.http.models import FieldCondition, Filter, MatchAny, MatchValue, Range

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailSearchModel
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
            total_score=0.9,
            max_score=0.9,
            avg_score=0.9,
            weighted_score=0.9,
            reranker_score=0.0,
            hit_count=1,
            hit_results=[],
        )
        mock_cocktail2 = create_test_cocktail_model("2", "Mojito")
        mock_cocktail2.search_statistics = CocktailSearchStatistics(
            total_score=0.8,
            max_score=0.8,
            avg_score=0.8,
            weighted_score=0.8,
            reranker_score=0.0,
            hit_count=1,
            hit_results=[],
        )

        mock_repository.search_vectors = AsyncMock(return_value=[mock_cocktail1, mock_cocktail2])
        mock_repository.get_all_cocktails = AsyncMock(return_value=[mock_cocktail1, mock_cocktail2])

        mock_qdrant_options = MagicMock()
        mock_reranker = AsyncMock()
        mock_reranker.rerank = AsyncMock(side_effect=lambda query, cocktails, top_k=10: cocktails)

        handler = FreeTextQueryHandler(
            cocktail_vector_repository=mock_repository,
            qdrant_opotions=mock_qdrant_options,
            reranker_service=mock_reranker,
        )

        query = FreeTextQuery(free_text="tequila")
        result = await handler.handle(query)

        assert len(result) == 2
        assert result[0].title == "Margarita"  # Higher score first
        assert result[1].title == "Mojito"
        # "tequila" now triggers base_spirit keyword filter
        call_kwargs = mock_repository.search_vectors.call_args[1]
        assert call_kwargs["free_text"] == "tequila"
        assert call_kwargs["query_filter"] is not None
        must_keys = {c.key for c in call_kwargs["query_filter"].must if isinstance(c, FieldCondition)}
        assert "metadata.keywords_base_spirit" in must_keys

    @pytest.mark.anyio
    async def test_handler_sorts_by_score(self):
        """Test that results are sorted by weighted_score descending."""
        mock_repository = AsyncMock()

        mock_cocktail1 = create_test_cocktail_model("1", "Low Score")
        mock_cocktail1.search_statistics = CocktailSearchStatistics(
            total_score=0.5,
            max_score=0.5,
            avg_score=0.5,
            weighted_score=0.5,
            reranker_score=0.0,
            hit_count=1,
            hit_results=[],
        )

        mock_cocktail2 = create_test_cocktail_model("2", "High Score")
        mock_cocktail2.search_statistics = CocktailSearchStatistics(
            total_score=0.9,
            max_score=0.9,
            avg_score=0.9,
            weighted_score=0.9,
            reranker_score=0.0,
            hit_count=1,
            hit_results=[],
        )

        mock_cocktail3 = create_test_cocktail_model("3", "Medium Score")
        mock_cocktail3.search_statistics = CocktailSearchStatistics(
            total_score=0.7,
            max_score=0.7,
            avg_score=0.7,
            weighted_score=0.7,
            reranker_score=0.0,
            hit_count=1,
            hit_results=[],
        )

        # Return in unsorted order
        mock_repository.search_vectors = AsyncMock(return_value=[mock_cocktail1, mock_cocktail2, mock_cocktail3])
        mock_repository.get_all_cocktails = AsyncMock(return_value=[mock_cocktail1, mock_cocktail2, mock_cocktail3])

        mock_qdrant_options = MagicMock()
        mock_reranker = AsyncMock()
        mock_reranker.rerank = AsyncMock(side_effect=lambda query, cocktails, top_k=10: cocktails)

        handler = FreeTextQueryHandler(
            cocktail_vector_repository=mock_repository,
            qdrant_opotions=mock_qdrant_options,
            reranker_service=mock_reranker,
        )

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
        mock_reranker = AsyncMock()
        mock_reranker.rerank = AsyncMock(side_effect=lambda query, cocktails, top_k=10: cocktails)

        handler = FreeTextQueryHandler(
            cocktail_vector_repository=mock_repository,
            qdrant_opotions=mock_qdrant_options,
            reranker_service=mock_reranker,
        )

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
        mock_reranker = AsyncMock()
        mock_reranker.rerank = AsyncMock(side_effect=lambda query, cocktails, top_k=10: cocktails)

        handler = FreeTextQueryHandler(
            cocktail_vector_repository=mock_repository,
            qdrant_opotions=mock_qdrant_options,
            reranker_service=mock_reranker,
        )

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
        mock_reranker = AsyncMock()
        mock_reranker.rerank = AsyncMock(side_effect=lambda query, cocktails, top_k=10: cocktails)

        handler = FreeTextQueryHandler(
            cocktail_vector_repository=mock_repository,
            qdrant_opotions=mock_qdrant_options,
            reranker_service=mock_reranker,
        )

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
        mock_reranker = AsyncMock()
        mock_reranker.rerank = AsyncMock(side_effect=lambda query, cocktails, top_k=10: cocktails)

        handler = FreeTextQueryHandler(
            cocktail_vector_repository=mock_repository,
            qdrant_opotions=mock_qdrant_options,
            reranker_service=mock_reranker,
        )

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
        mock_reranker = AsyncMock()
        mock_reranker.rerank = AsyncMock(side_effect=lambda query, cocktails, top_k=10: cocktails)

        handler = FreeTextQueryHandler(
            cocktail_vector_repository=mock_repository,
            qdrant_opotions=mock_qdrant_options,
            reranker_service=mock_reranker,
        )

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
        mock_reranker = AsyncMock()
        mock_reranker.rerank = AsyncMock(side_effect=lambda query, cocktails, top_k=10: cocktails)

        handler = FreeTextQueryHandler(
            cocktail_vector_repository=mock_repository,
            qdrant_opotions=mock_qdrant_options,
            reranker_service=mock_reranker,
        )

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
        mock_reranker = AsyncMock()
        mock_reranker.rerank = AsyncMock(side_effect=lambda query, cocktails, top_k=10: cocktails)
        return FreeTextQueryHandler(
            cocktail_vector_repository=mock_repository,
            qdrant_opotions=mock_qdrant_options,
            reranker_service=mock_reranker,
        )

    def test_returns_none_for_plain_query(self):
        """Test that a plain query without structured elements returns None."""
        handler = self._make_handler()
        result = handler._build_query_filter("delicious recipes")
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
        result = handler._build_query_filter("cocktails without honey")
        assert result is not None
        assert result.must_not is not None
        assert isinstance(result.must_not, list)
        assert len(result.must_not) == 1
        assert isinstance(result.must_not[0], FieldCondition)
        assert result.must_not[0].key == "metadata.ingredient_words"
        assert result.must_not[0].match is not None
        assert isinstance(result.must_not[0].match, MatchValue)
        assert result.must_not[0].match.value == "honey"

    def test_combined_filters(self):
        """Test that multiple filter conditions combine correctly."""
        handler = self._make_handler()
        result = handler._build_query_filter("simple iba cocktails without honey")
        assert result is not None
        # Should have IBA + ingredient_count in must
        assert isinstance(result.must, list)
        assert all(isinstance(c, FieldCondition) for c in result.must)
        must_keys = {c.key for c in result.must if isinstance(c, FieldCondition)}
        assert "metadata.is_iba" in must_keys
        assert "metadata.ingredient_count" in must_keys
        # Should have honey exclusion in must_not
        assert result.must_not is not None
        assert isinstance(result.must_not, list)
        assert len(result.must_not) == 1
        assert isinstance(result.must_not[0], FieldCondition)
        assert result.must_not[0].match is not None
        assert isinstance(result.must_not[0].match, MatchValue)
        assert result.must_not[0].match.value == "honey"

    def test_inclusion_filter_with_honey(self):
        """Test that 'with honey' triggers ingredient inclusion filter."""
        handler = self._make_handler()
        result = handler._build_query_filter("cocktails with honey")
        assert result is not None
        assert isinstance(result.must, list)
        inclusion_conditions = [
            c for c in result.must if isinstance(c, FieldCondition) and c.key == "metadata.ingredient_words"
        ]
        assert len(inclusion_conditions) == 1
        assert isinstance(inclusion_conditions[0].match, MatchValue)
        assert inclusion_conditions[0].match.value == "honey"

    def test_inclusion_filter_using_pattern(self):
        """Test that 'using lime' triggers ingredient inclusion filter."""
        handler = self._make_handler()
        result = handler._build_query_filter("cocktails using lime")
        assert result is not None
        assert isinstance(result.must, list)
        inclusion_conditions = [
            c for c in result.must if isinstance(c, FieldCondition) and c.key == "metadata.ingredient_words"
        ]
        assert len(inclusion_conditions) == 1
        assert isinstance(inclusion_conditions[0].match, MatchValue)
        assert inclusion_conditions[0].match.value == "lime"

    def test_inclusion_filter_made_with_pattern(self):
        """Test that 'made with bourbon' triggers ingredient inclusion filter."""
        handler = self._make_handler()
        result = handler._build_query_filter("cocktails made with bourbon")
        assert result is not None
        assert isinstance(result.must, list)
        inclusion_conditions = [
            c for c in result.must if isinstance(c, FieldCondition) and c.key == "metadata.ingredient_words"
        ]
        assert len(inclusion_conditions) >= 1
        inclusion_values = [c.match.value for c in inclusion_conditions if isinstance(c.match, MatchValue)]
        assert "bourbon" in inclusion_values

    def test_inclusion_does_not_trigger_on_without(self):
        """Test that 'without honey' does NOT trigger ingredient inclusion."""
        handler = self._make_handler()
        result = handler._build_query_filter("cocktails without honey")
        assert result is not None
        # Must should not have ingredient_words inclusion
        if result.must:
            inclusion_conditions = [
                c for c in result.must if isinstance(c, FieldCondition) and c.key == "metadata.ingredient_words"
            ]
            assert len(inclusion_conditions) == 0
        # Must_not should have the exclusion
        assert result.must_not is not None
        assert isinstance(result.must_not, list)
        assert len(result.must_not) == 1

    def test_inclusion_and_exclusion_combined(self):
        """Test that 'with lime without honey' includes lime and excludes honey."""
        handler = self._make_handler()
        result = handler._build_query_filter("cocktails with lime without honey")
        assert result is not None
        assert isinstance(result.must, list)
        inclusion_conditions = [
            c for c in result.must if isinstance(c, FieldCondition) and c.key == "metadata.ingredient_words"
        ]
        assert len(inclusion_conditions) == 1
        assert isinstance(inclusion_conditions[0].match, MatchValue)
        assert inclusion_conditions[0].match.value == "lime"
        # Exclusion
        assert result.must_not is not None
        assert isinstance(result.must_not, list)
        assert len(result.must_not) == 1
        assert isinstance(result.must_not[0], FieldCondition)
        assert isinstance(result.must_not[0].match, MatchValue)
        assert result.must_not[0].match.value == "honey"


class TestExtractExclusionTerms:
    """Test cases for _extract_exclusion_terms."""

    def _make_handler(self):
        mock_repository = AsyncMock()
        mock_qdrant_options = MagicMock()
        mock_reranker = AsyncMock()
        mock_reranker.rerank = AsyncMock(side_effect=lambda query, cocktails, top_k=10: cocktails)
        return FreeTextQueryHandler(
            cocktail_vector_repository=mock_repository,
            qdrant_opotions=mock_qdrant_options,
            reranker_service=mock_reranker,
        )

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

    def test_multi_word_exclusion(self):
        """Test that multi-word ingredients are extracted (e.g., 'blue curacao')."""
        handler = self._make_handler()
        terms = handler._extract_exclusion_terms("cocktails without blue curacao")
        assert "blue" in terms
        assert "curacao" in terms

    def test_multi_word_exclusion_orange_juice(self):
        """Test multi-word exclusion with 'orange juice'."""
        handler = self._make_handler()
        terms = handler._extract_exclusion_terms("cocktails without orange juice")
        assert "orange" in terms
        assert "juice" in terms

    def test_multi_word_exclusion_stops_at_stop_word(self):
        """Test that multi-word extraction stops at conjunctions/stop words."""
        handler = self._make_handler()
        terms = handler._extract_exclusion_terms("cocktails without honey and lime")
        assert "honey" in terms
        # "and" is a stop word, so "lime" is not part of the honey exclusion
        assert "lime" not in terms

    def test_multi_word_exclusion_max_three_words(self):
        """Test that multi-word extraction is capped at 3 words."""
        handler = self._make_handler()
        terms = handler._extract_exclusion_terms("cocktails without dark aged jamaican overproof rum")
        # Should only capture first 3 words of the phrase
        assert len(terms) == 3

    def test_not_containing_pattern(self):
        handler = self._make_handler()
        terms = handler._extract_exclusion_terms("cocktails not containing honey")
        assert terms == ["honey"]

    def test_not_featuring_pattern(self):
        handler = self._make_handler()
        terms = handler._extract_exclusion_terms("cocktails not featuring rum")
        assert terms == ["rum"]

    def test_that_exclude_pattern(self):
        handler = self._make_handler()
        terms = handler._extract_exclusion_terms("cocktails that exclude lime")
        assert terms == ["lime"]


class TestExtractInclusionTerms:
    """Test cases for _extract_inclusion_terms."""

    def _make_handler(self):
        mock_repository = AsyncMock()
        mock_qdrant_options = MagicMock()
        mock_reranker = AsyncMock()
        mock_reranker.rerank = AsyncMock(side_effect=lambda query, cocktails, top_k=10: cocktails)
        return FreeTextQueryHandler(
            cocktail_vector_repository=mock_repository,
            qdrant_opotions=mock_qdrant_options,
            reranker_service=mock_reranker,
        )

    def test_with_pattern(self):
        handler = self._make_handler()
        terms = handler._extract_inclusion_terms("cocktails with honey")
        assert terms == ["honey"]

    def test_using_pattern(self):
        handler = self._make_handler()
        terms = handler._extract_inclusion_terms("cocktails using lime")
        assert terms == ["lime"]

    def test_containing_pattern(self):
        handler = self._make_handler()
        terms = handler._extract_inclusion_terms("cocktails containing ginger")
        assert terms == ["ginger"]

    def test_contains_pattern(self):
        handler = self._make_handler()
        terms = handler._extract_inclusion_terms("cocktails contains honey")
        assert terms == ["honey"]

    def test_made_with_pattern(self):
        handler = self._make_handler()
        terms = handler._extract_inclusion_terms("cocktails made with bourbon")
        assert terms == ["bourbon"]

    def test_no_inclusion_in_plain_query(self):
        handler = self._make_handler()
        terms = handler._extract_inclusion_terms("tequila cocktails")
        assert terms == []

    def test_without_does_not_trigger_inclusion(self):
        """Test that 'without honey' does NOT trigger inclusion for honey."""
        handler = self._make_handler()
        terms = handler._extract_inclusion_terms("cocktails without honey")
        assert "honey" not in terms

    def test_multi_word_inclusion(self):
        """Test that multi-word ingredients are extracted."""
        handler = self._make_handler()
        terms = handler._extract_inclusion_terms("cocktails with lime juice")
        assert "lime" in terms
        assert "juice" in terms

    def test_inclusion_stops_at_stop_word(self):
        """Test that extraction stops at stop words."""
        handler = self._make_handler()
        terms = handler._extract_inclusion_terms("cocktails with honey and lime")
        assert "honey" in terms
        # 'and' is a stop word, so 'lime' is not part of the honey inclusion
        assert "lime" not in terms

    def test_inclusion_excludes_already_excluded_terms(self):
        """Test that terms in exclusion are not also included."""
        handler = self._make_handler()
        terms = handler._extract_inclusion_terms("cocktails with lime without lime")
        assert "lime" not in terms

    def test_multi_word_inclusion_max_three_words(self):
        """Test that multi-word extraction is capped at 3 words."""
        handler = self._make_handler()
        terms = handler._extract_inclusion_terms("cocktails with fresh squeezed blood orange juice")
        assert len(terms) == 3

    def test_featuring_pattern(self):
        handler = self._make_handler()
        terms = handler._extract_inclusion_terms("cocktails featuring honey")
        assert terms == ["honey"]

    def test_have_pattern(self):
        handler = self._make_handler()
        terms = handler._extract_inclusion_terms("cocktails that have ginger")
        assert terms == ["ginger"]

    def test_that_include_pattern(self):
        handler = self._make_handler()
        terms = handler._extract_inclusion_terms("cocktails that include lime")
        assert terms == ["lime"]


class TestFuzzyNameMatch:
    """Test cases for fuzzy name matching."""

    def _make_handler(self):
        mock_repository = AsyncMock()
        mock_qdrant_options = MagicMock()
        mock_reranker = AsyncMock()
        mock_reranker.rerank = AsyncMock(side_effect=lambda query, cocktails, top_k=10: cocktails)
        return FreeTextQueryHandler(
            cocktail_vector_repository=mock_repository,
            qdrant_opotions=mock_qdrant_options,
            reranker_service=mock_reranker,
        )

    def test_fuzzy_match_misspelled_margarita(self):
        """Test that 'Margerita' fuzzy-matches 'Margarita'."""
        handler = self._make_handler()
        cocktails = [create_test_cocktail_model("1", "Margarita")]
        result = handler._find_fuzzy_name_match("margerita", cocktails)
        assert result is not None
        assert len(result) == 1
        assert result[0].title == "Margarita"

    def test_fuzzy_match_misspelled_mojito(self):
        """Test that 'Mohito' fuzzy-matches 'Mojito'."""
        handler = self._make_handler()
        cocktails = [create_test_cocktail_model("1", "Mojito")]
        result = handler._find_fuzzy_name_match("mohito", cocktails)
        assert result is not None
        assert len(result) == 1
        assert result[0].title == "Mojito"

    def test_fuzzy_match_misspelled_daiquiri(self):
        """Test that 'Daiquri' fuzzy-matches 'Daiquiri'."""
        handler = self._make_handler()
        cocktails = [create_test_cocktail_model("1", "Daiquiri")]
        result = handler._find_fuzzy_name_match("daiquri", cocktails)
        assert result is not None
        assert len(result) == 1
        assert result[0].title == "Daiquiri"

    def test_fuzzy_match_returns_none_for_low_similarity(self):
        """Test that very different strings don't match."""
        handler = self._make_handler()
        cocktails = [create_test_cocktail_model("1", "Margarita")]
        result = handler._find_fuzzy_name_match("whiskey sour", cocktails)
        assert result is None

    def test_fuzzy_match_sorts_by_score(self):
        """Test that fuzzy results are sorted by score descending."""
        handler = self._make_handler()
        cocktails = [
            create_test_cocktail_model("1", "Margarita"),
            create_test_cocktail_model("2", "Margherita Pizza"),
        ]
        result = handler._find_fuzzy_name_match("margerita", cocktails)
        assert result is not None
        assert result[0].title == "Margarita"  # Higher fuzzy score

    def test_exact_name_match_tries_fuzzy_as_fallback(self):
        """Test that _find_exact_name_match falls back to fuzzy matching."""
        handler = self._make_handler()
        cocktails = [
            create_test_cocktail_model("1", "Margarita"),
            create_test_cocktail_model("2", "Mojito"),
        ]
        result = handler._find_exact_name_match("margerita", cocktails)
        assert result is not None
        assert len(result) == 1
        assert result[0].title == "Margarita"


class TestKeywordMetadataFilters:
    """Test cases for keyword metadata filters in _build_query_filter."""

    def _make_handler(self):
        mock_repository = AsyncMock()
        mock_qdrant_options = MagicMock()
        mock_reranker = AsyncMock()
        mock_reranker.rerank = AsyncMock(side_effect=lambda query, cocktails, top_k=10: cocktails)
        return FreeTextQueryHandler(
            cocktail_vector_repository=mock_repository,
            qdrant_opotions=mock_qdrant_options,
            reranker_service=mock_reranker,
        )

    def test_base_spirit_gin_filter(self):
        """Test that 'gin' triggers base_spirit filter."""
        handler = self._make_handler()
        result = handler._build_query_filter("gin cocktails")
        assert result is not None
        assert isinstance(result.must, list)
        must_keys = {c.key for c in result.must if isinstance(c, FieldCondition)}
        assert "metadata.keywords_base_spirit" in must_keys
        spirit_condition = next(
            c for c in result.must if isinstance(c, FieldCondition) and c.key == "metadata.keywords_base_spirit"
        )
        assert isinstance(spirit_condition.match, MatchValue)
        assert spirit_condition.match.value == "gin"

    def test_base_spirit_bourbon_filter(self):
        """Test that 'bourbon' triggers base_spirit filter."""
        handler = self._make_handler()
        result = handler._build_query_filter("bourbon cocktails")
        assert result is not None
        assert isinstance(result.must, list)
        spirit_condition = next(
            c for c in result.must if isinstance(c, FieldCondition) and c.key == "metadata.keywords_base_spirit"
        )
        assert isinstance(spirit_condition.match, MatchValue)
        assert spirit_condition.match.value == "bourbon"

    def test_flavor_profile_filter(self):
        """Test that flavor keywords trigger flavor_profile filter."""
        handler = self._make_handler()
        result = handler._build_query_filter("bitter cocktails")
        assert result is not None
        assert isinstance(result.must, list)
        must_keys = {c.key for c in result.must if isinstance(c, FieldCondition)}
        assert "metadata.keywords_flavor_profile" in must_keys

    def test_cocktail_family_filter(self):
        """Test that cocktail family keywords trigger filter."""
        handler = self._make_handler()
        result = handler._build_query_filter("tiki cocktails")
        assert result is not None
        assert isinstance(result.must, list)
        must_keys = {c.key for c in result.must if isinstance(c, FieldCondition)}
        assert "metadata.keywords_cocktail_family" in must_keys

    def test_technique_shaken_filter(self):
        """Test that 'shaken' triggers technique filter."""
        handler = self._make_handler()
        result = handler._build_query_filter("shaken cocktails")
        assert result is not None
        assert isinstance(result.must, list)
        technique_condition = next(
            c for c in result.must if isinstance(c, FieldCondition) and c.key == "metadata.keywords_technique"
        )
        assert isinstance(technique_condition.match, MatchValue)
        assert technique_condition.match.value == "shaken"

    def test_technique_stirred_filter(self):
        """Test that 'stirred' triggers technique filter."""
        handler = self._make_handler()
        result = handler._build_query_filter("stirred drinks")
        assert result is not None
        assert isinstance(result.must, list)
        technique_condition = next(
            c for c in result.must if isinstance(c, FieldCondition) and c.key == "metadata.keywords_technique"
        )
        assert isinstance(technique_condition.match, MatchValue)
        assert technique_condition.match.value == "stirred"

    def test_strength_filter(self):
        """Test that strength keywords trigger filter."""
        handler = self._make_handler()
        result = handler._build_query_filter("strong cocktails")
        assert result is not None
        assert isinstance(result.must, list)
        strength_condition = next(
            c for c in result.must if isinstance(c, FieldCondition) and c.key == "metadata.keywords_strength"
        )
        assert isinstance(strength_condition.match, MatchValue)
        assert strength_condition.match.value == "strong"

    def test_temperature_filter(self):
        """Test that temperature keywords trigger filter."""
        handler = self._make_handler()
        result = handler._build_query_filter("frozen cocktails")
        assert result is not None
        assert isinstance(result.must, list)
        temp_condition = next(
            c for c in result.must if isinstance(c, FieldCondition) and c.key == "metadata.keywords_temperature"
        )
        assert isinstance(temp_condition.match, MatchValue)
        assert temp_condition.match.value == "frozen"

    def test_season_filter(self):
        """Test that season keywords trigger filter."""
        handler = self._make_handler()
        result = handler._build_query_filter("summer cocktails")
        assert result is not None
        assert isinstance(result.must, list)
        season_condition = next(
            c for c in result.must if isinstance(c, FieldCondition) and c.key == "metadata.keywords_season"
        )
        assert isinstance(season_condition.match, MatchAny)
        assert "summer" in season_condition.match.any

    def test_season_autumn_maps_to_fall(self):
        """Test that 'autumn' is normalized to 'fall'."""
        handler = self._make_handler()
        result = handler._build_query_filter("autumn cocktails")
        assert result is not None
        assert isinstance(result.must, list)
        season_condition = next(
            c for c in result.must if isinstance(c, FieldCondition) and c.key == "metadata.keywords_season"
        )
        assert isinstance(season_condition.match, MatchAny)
        assert "fall" in season_condition.match.any

    def test_occasion_filter(self):
        """Test that occasion keywords trigger filter."""
        handler = self._make_handler()
        result = handler._build_query_filter("brunch cocktails")
        assert result is not None
        assert isinstance(result.must, list)
        occasion_condition = next(
            c for c in result.must if isinstance(c, FieldCondition) and c.key == "metadata.keywords_occasion"
        )
        assert isinstance(occasion_condition.match, MatchValue)
        assert occasion_condition.match.value == "brunch"

    def test_mood_filter(self):
        """Test that mood keywords trigger filter."""
        handler = self._make_handler()
        result = handler._build_query_filter("refreshing cocktails")
        assert result is not None
        assert isinstance(result.must, list)
        mood_condition = next(
            c for c in result.must if isinstance(c, FieldCondition) and c.key == "metadata.keywords_mood"
        )
        assert isinstance(mood_condition.match, MatchValue)
        assert mood_condition.match.value == "refreshing"

    def test_combined_keyword_filters(self):
        """Test that multiple keyword dimensions combine correctly."""
        handler = self._make_handler()
        result = handler._build_query_filter("refreshing gin summer cocktails")
        assert result is not None
        assert isinstance(result.must, list)
        must_keys = {c.key for c in result.must if isinstance(c, FieldCondition)}
        assert "metadata.keywords_base_spirit" in must_keys
        assert "metadata.keywords_season" in must_keys
        assert "metadata.keywords_mood" in must_keys
