import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import create_test_cocktail_embedding_model, create_test_cocktail_model
from fastapi import Response
from fastapi.testclient import TestClient

from cezzis_com_cocktails_aisearch.apis.embedding import EmbeddingRouter
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_description_chunk import (
    CocktailDescriptionChunk,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailModel
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktails_embedding_rq import (
    CocktailEmbeddingRq,
)


class TestEmbeddingRouter:
    """Test cases for EmbeddingRouter."""

    def test_init(self):
        """Test router initialization."""
        mediator = MagicMock()
        router = EmbeddingRouter(mediator=mediator)

        assert router.mediator == mediator
        assert len(router.routes) > 0

    def test_route_configuration(self):
        """Test that the embed route is configured correctly."""
        mediator = MagicMock()
        router = EmbeddingRouter(mediator=mediator)

        # Verify route was added
        assert len(router.routes) > 0

    @pytest.mark.anyio
    async def test_embed_success(self):
        """Test successful embedding operation."""
        mediator = AsyncMock()
        mediator.send_async = AsyncMock(return_value=True)

        router = EmbeddingRouter(mediator=mediator)

        request_mock = MagicMock()
        cocktail_model = create_test_cocktail_embedding_model("test-123", "Test Cocktail")
        body = CocktailEmbeddingRq(
            content_chunks=[CocktailDescriptionChunk(content="Test content", category="description")],
            cocktail_embedding_model=cocktail_model,
        )

        # Bypass OAuth by setting ENV=local
        with patch.dict(os.environ, {"ENV": "local"}):
            response = await router.embed(_rq=request_mock, body=body)

        assert response.status_code == 204
        mediator.send_async.assert_called_once()

    @pytest.mark.anyio
    async def test_embed_failure(self):
        """Test embedding failure raises exception."""
        mediator = AsyncMock()
        mediator.send_async = AsyncMock(return_value=False)

        router = EmbeddingRouter(mediator=mediator)

        request_mock = MagicMock()
        cocktail_model = create_test_cocktail_embedding_model("test-123", "Test Cocktail")
        body = CocktailEmbeddingRq(
            content_chunks=[CocktailDescriptionChunk(content="Test content", category="description")],
            cocktail_embedding_model=cocktail_model,
        )

        # Bypass OAuth by setting ENV=local
        with patch.dict(os.environ, {"ENV": "local"}):
            with pytest.raises(Exception, match="Failed to embed cocktail description chunks"):
                await router.embed(_rq=request_mock, body=body)
