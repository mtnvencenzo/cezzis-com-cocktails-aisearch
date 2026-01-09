from unittest.mock import AsyncMock, MagicMock

import pytest
from conftest import create_test_cocktail_model

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.commands.cocktail_embedding_command import (
    CocktailEmbeddingCommand,
    CocktailEmbeddingCommandHandler,
    CocktailEmbeddingCommandValidator,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_description_chunk import (
    CocktailDescriptionChunk,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailModel


class TestCocktailEmbeddingCommand:
    """Test cases for CocktailEmbeddingCommand."""

    def test_command_init(self):
        """Test command initialization."""
        cocktail_model = create_test_cocktail_model("test-123", "Test Cocktail")
        chunks = [CocktailDescriptionChunk(content="Test content", category="desc")]

        command = CocktailEmbeddingCommand(chunks=chunks, cocktail_model=cocktail_model)

        assert command.chunks == chunks
        assert command.cocktail_model == cocktail_model


class TestCocktailEmbeddingCommandValidator:
    """Test cases for CocktailEmbeddingCommandValidator."""

    def test_validator_success(self):
        """Test successful validation."""
        cocktail_model = create_test_cocktail_model("test-123", "Test Cocktail")
        chunks = [CocktailDescriptionChunk(content="Test content", category="desc")]
        command = CocktailEmbeddingCommand(chunks=chunks, cocktail_model=cocktail_model)

        validator = CocktailEmbeddingCommandValidator()
        next_mock = MagicMock()

        validator.handle(command, next_mock)

        next_mock.assert_called_once()

    def test_validator_raises_on_missing_cocktail_model(self):
        """Test validator raises error when cocktail model is missing."""
        chunks = [CocktailDescriptionChunk(content="Test", category="desc")]
        command = CocktailEmbeddingCommand(chunks=chunks, cocktail_model=None)  # type: ignore

        validator = CocktailEmbeddingCommandValidator()
        next_mock = MagicMock()

        with pytest.raises(ValueError, match="Invalid cocktail model"):
            validator.handle(command, next_mock)

    def test_validator_raises_on_missing_cocktail_id(self):
        """Test validator raises error when cocktail id is missing."""
        cocktail_model = create_test_cocktail_model("", "Test")
        chunks = [CocktailDescriptionChunk(content="Test", category="desc")]
        command = CocktailEmbeddingCommand(chunks=chunks, cocktail_model=cocktail_model)

        validator = CocktailEmbeddingCommandValidator()
        next_mock = MagicMock()

        with pytest.raises(ValueError, match="Invalid cocktail model"):
            validator.handle(command, next_mock)

    def test_validator_raises_on_empty_chunks(self):
        """Test validator raises error when chunks are empty."""
        cocktail_model = create_test_cocktail_model("test-123", "Test")
        command = CocktailEmbeddingCommand(chunks=[], cocktail_model=cocktail_model)

        validator = CocktailEmbeddingCommandValidator()
        next_mock = MagicMock()

        with pytest.raises(ValueError, match="No chunks provided"):
            validator.handle(command, next_mock)

    def test_validator_raises_on_all_empty_content_chunks(self):
        """Test validator raises error when all chunks have empty content."""
        cocktail_model = create_test_cocktail_model("test-123", "Test")
        chunks = [
            CocktailDescriptionChunk(content="  ", category="desc"),
            CocktailDescriptionChunk(content="", category="ingredients"),
        ]
        command = CocktailEmbeddingCommand(chunks=chunks, cocktail_model=cocktail_model)

        validator = CocktailEmbeddingCommandValidator()
        next_mock = MagicMock()

        with pytest.raises(ValueError, match="No valid chunks to embed"):
            validator.handle(command, next_mock)


class TestCocktailEmbeddingCommandHandler:
    """Test cases for CocktailEmbeddingCommandHandler."""

    @pytest.mark.anyio
    async def test_handler_success(self):
        """Test successful command handling."""
        mock_repository = AsyncMock()
        mock_repository.delete_vectors = AsyncMock()
        mock_repository.store_vectors = AsyncMock()

        handler = CocktailEmbeddingCommandHandler(cocktail_vector_repository=mock_repository)

        cocktail_model = create_test_cocktail_model("test-123", "Test Cocktail")
        chunks = [CocktailDescriptionChunk(content="Test content", category="desc")]
        command = CocktailEmbeddingCommand(chunks=chunks, cocktail_model=cocktail_model)

        result = await handler.handle(command)

        assert result is True
        mock_repository.delete_vectors.assert_called_once_with("test-123")
        mock_repository.store_vectors.assert_called_once()

    @pytest.mark.anyio
    async def test_handler_filters_empty_chunks(self):
        """Test that handler filters out empty chunks before storage."""
        mock_repository = AsyncMock()
        mock_repository.delete_vectors = AsyncMock()
        mock_repository.store_vectors = AsyncMock()

        handler = CocktailEmbeddingCommandHandler(cocktail_vector_repository=mock_repository)

        cocktail_model = create_test_cocktail_model("test-123", "Test Cocktail")
        chunks = [
            CocktailDescriptionChunk(content="Valid content", category="desc"),
            CocktailDescriptionChunk(content="  ", category="empty"),
            CocktailDescriptionChunk(content="Another valid", category="ingredients"),
        ]
        command = CocktailEmbeddingCommand(chunks=chunks, cocktail_model=cocktail_model)

        await handler.handle(command)

        # Verify only non-empty chunks are stored
        store_call = mock_repository.store_vectors.call_args
        stored_chunks = store_call[1]["chunks"]
        assert len(stored_chunks) == 2
        assert all(chunk.content.strip() != "" for chunk in stored_chunks)

    @pytest.mark.anyio
    async def test_handler_deletes_before_storing(self):
        """Test that handler deletes existing vectors before storing new ones."""
        mock_repository = AsyncMock()
        call_order = []

        async def track_delete(*args, **kwargs):
            call_order.append("delete")

        async def track_store(*args, **kwargs):
            call_order.append("store")

        mock_repository.delete_vectors = AsyncMock(side_effect=track_delete)
        mock_repository.store_vectors = AsyncMock(side_effect=track_store)

        handler = CocktailEmbeddingCommandHandler(cocktail_vector_repository=mock_repository)

        cocktail_model = create_test_cocktail_model("test-123", "Test")
        chunks = [CocktailDescriptionChunk(content="Test", category="desc")]
        command = CocktailEmbeddingCommand(chunks=chunks, cocktail_model=cocktail_model)

        await handler.handle(command)

        assert call_order == ["delete", "store"]
