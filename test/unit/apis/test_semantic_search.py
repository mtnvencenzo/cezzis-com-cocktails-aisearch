from unittest.mock import AsyncMock, MagicMock

import pytest
from conftest import create_test_cocktail_model

from cezzis_com_cocktails_aisearch.apis.semantic_search import SemanticSearchRouter
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailModel
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktails_search_rs import (
    CocktailsSearchRs,
)


class TestSemanticSearchRouter:
    """Test cases for SemanticSearchRouter."""

    def test_init(self):
        """Test router initialization."""
        mediator = MagicMock()
        router = SemanticSearchRouter(mediator=mediator)

        assert router.mediator == mediator
        assert len(router.routes) > 0

    def test_route_configuration(self):
        """Test that the search route is configured correctly."""
        mediator = MagicMock()
        router = SemanticSearchRouter(mediator=mediator)

        # Verify route was added
        assert len(router.routes) > 0

    @pytest.mark.anyio
    async def test_search_success(self):
        """Test successful search operation."""
        mediator = AsyncMock()
        mock_cocktails = [create_test_cocktail_model("1", "Margarita"), create_test_cocktail_model("2", "Mojito")]
        mediator.send_async = AsyncMock(return_value=mock_cocktails)

        router = SemanticSearchRouter(mediator=mediator)

        request_mock = MagicMock()

        result = await router.search(
            _rq=request_mock, freetext="tequila", skip=0, take=10, m=None, m_ex=False, inc=None, fi=None
        )

        assert isinstance(result, CocktailsSearchRs)
        assert len(result.items) == 2
        assert result.items[0].title == "Margarita"
        mediator.send_async.assert_called_once()

    @pytest.mark.anyio
    async def test_search_with_empty_result(self):
        """Test search with no results."""
        mediator = AsyncMock()
        mediator.send_async = AsyncMock(return_value=[])

        router = SemanticSearchRouter(mediator=mediator)

        request_mock = MagicMock()

        result = await router.search(
            _rq=request_mock, freetext="nonexistent", skip=0, take=10, m=None, m_ex=False, inc=None, fi=None
        )

        assert isinstance(result, CocktailsSearchRs)
        assert len(result.items) == 0

    @pytest.mark.anyio
    async def test_search_with_defaults(self):
        """Test search with default parameter values."""
        mediator = AsyncMock()
        mediator.send_async = AsyncMock(return_value=[])

        router = SemanticSearchRouter(mediator=mediator)

        request_mock = MagicMock()

        result = await router.search(
            _rq=request_mock, freetext=None, skip=None, take=None, m=None, m_ex=None, inc=None, fi=None
        )

        assert isinstance(result, CocktailsSearchRs)
        # Verify that defaults are applied in the query
        call_args = mediator.send_async.call_args[0][0]
        assert call_args.free_text == ""
        assert call_args.skip == 0
        assert call_args.take == 10
        assert call_args.match == []
        assert call_args.match_exclusive is False
