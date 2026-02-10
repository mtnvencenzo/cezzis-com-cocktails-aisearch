from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from conftest import create_test_cocktail_model

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailSearchModel
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_search_statistics import (
    CocktailSearchStatistics,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.ingredient_model import (
    CocktailSearchIngredientModel,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.ingredient_requirment_type_model import (
    CocktailSearchIngredientRequirementTypeModel,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.preparation_type_model import (
    CocktailSearchPreparationTypeModel,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.uofm_type_model import (
    CocktailSearchUofMTypeModel,
)
from cezzis_com_cocktails_aisearch.infrastructure.services.reranker_service import RerankerService


def _make_ingredient(name: str) -> CocktailSearchIngredientModel:
    """Create a minimal ingredient model for testing."""
    return CocktailSearchIngredientModel(
        name=name,
        uoM=CocktailSearchUofMTypeModel.Ounces,
        requirement=CocktailSearchIngredientRequirementTypeModel.required,
        display=f"2 oz {name}",
        units=2.0,
        preparation=CocktailSearchPreparationTypeModel.None_,
        suggestions="",
        types=[],
        applications=[],
    )


class TestRerankerService:
    """Test cases for RerankerService."""

    def _make_options(
        self, endpoint="http://localhost:8990", api_key="", score_threshold=0.0, relative_score_cutoff=0.0
    ):
        options = MagicMock()
        options.endpoint = endpoint
        options.api_key = api_key
        options.score_threshold = score_threshold
        options.relative_score_cutoff = relative_score_cutoff
        return options

    @pytest.mark.anyio
    async def test_rerank_empty_list_returns_empty(self):
        """Test that empty cocktail list returns empty."""
        options = self._make_options()
        service = RerankerService(reranker_options=options)

        result = await service.rerank(query="tequila", cocktails=[])

        assert result == []

    @pytest.mark.anyio
    async def test_rerank_success_reorders_by_score(self):
        """Test that reranker reorders cocktails by cross-encoder score."""
        options = self._make_options(endpoint="http://localhost:8990")
        service = RerankerService(reranker_options=options)

        cocktails = [
            create_test_cocktail_model("1", "Low Relevance"),
            create_test_cocktail_model("2", "High Relevance"),
            create_test_cocktail_model("3", "Medium Relevance"),
        ]

        # TEI returns results sorted by score desc, with original indices
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"index": 1, "score": 0.95},
            {"index": 2, "score": 0.70},
            {"index": 0, "score": 0.30},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.services.reranker_service.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await service.rerank(query="tequila cocktails", cocktails=cocktails)

        assert len(result) == 3
        assert result[0].title == "High Relevance"
        assert result[1].title == "Medium Relevance"
        assert result[2].title == "Low Relevance"

    @pytest.mark.anyio
    async def test_rerank_applies_score_threshold(self):
        """Test that reranker filters out cocktails below score threshold."""
        options = self._make_options(endpoint="http://localhost:8990", score_threshold=0.5)
        service = RerankerService(reranker_options=options)

        cocktails = [
            create_test_cocktail_model("1", "Good Match"),
            create_test_cocktail_model("2", "Poor Match"),
        ]

        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"index": 0, "score": 0.85},
            {"index": 1, "score": 0.20},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.services.reranker_service.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await service.rerank(query="refreshing", cocktails=cocktails)

        assert len(result) == 1
        assert result[0].title == "Good Match"

    @pytest.mark.anyio
    async def test_rerank_applies_top_k(self):
        """Test that reranker limits results to top_k."""
        options = self._make_options(endpoint="http://localhost:8990")
        service = RerankerService(reranker_options=options)

        cocktails = [create_test_cocktail_model(str(i), f"Cocktail {i}") for i in range(5)]

        mock_response = MagicMock()
        mock_response.json.return_value = [{"index": i, "score": 0.9 - i * 0.1} for i in range(5)]
        mock_response.raise_for_status = MagicMock()

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.services.reranker_service.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await service.rerank(query="cocktail", cocktails=cocktails, top_k=2)

        assert len(result) == 2

    @pytest.mark.anyio
    async def test_rerank_sets_reranker_score_on_statistics(self):
        """Test that reranker updates search_statistics.reranker_score."""
        options = self._make_options(endpoint="http://localhost:8990")
        service = RerankerService(reranker_options=options)

        cocktail = create_test_cocktail_model("1", "Margarita")

        mock_response = MagicMock()
        mock_response.json.return_value = [{"index": 0, "score": 0.88}]
        mock_response.raise_for_status = MagicMock()

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.services.reranker_service.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await service.rerank(query="tequila", cocktails=[cocktail])

        assert len(result) == 1
        assert result[0].search_statistics.reranker_score == 0.88

    @pytest.mark.anyio
    async def test_rerank_graceful_degradation_on_http_error(self):
        """Test that reranker returns original list on HTTP error."""
        options = self._make_options(endpoint="http://localhost:8990")
        service = RerankerService(reranker_options=options)

        cocktails = [create_test_cocktail_model("1", "Margarita")]

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.services.reranker_service.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                side_effect=httpx.HTTPStatusError("Server Error", request=MagicMock(), response=MagicMock())
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await service.rerank(query="tequila", cocktails=cocktails)

        # Should return original list, not raise
        assert result == cocktails

    @pytest.mark.anyio
    async def test_rerank_graceful_degradation_on_connection_error(self):
        """Test that reranker returns original list when TEI is unreachable."""
        options = self._make_options(endpoint="http://localhost:8990")
        service = RerankerService(reranker_options=options)

        cocktails = [create_test_cocktail_model("1", "Margarita")]

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.services.reranker_service.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await service.rerank(query="tequila", cocktails=cocktails)

        assert result == cocktails

    @pytest.mark.anyio
    async def test_rerank_sends_correct_payload(self):
        """Test that reranker sends correct payload to TEI."""
        options = self._make_options(endpoint="http://localhost:8990", api_key="test-key")
        service = RerankerService(reranker_options=options)

        cocktails = [create_test_cocktail_model("1", "Margarita")]

        mock_response = MagicMock()
        mock_response.json.return_value = [{"index": 0, "score": 0.9}]
        mock_response.raise_for_status = MagicMock()

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.services.reranker_service.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await service.rerank(query="tequila lime", cocktails=cocktails)

            # Verify the POST call
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "http://localhost:8990/rerank"  # URL
            payload = call_args[1]["json"]
            assert payload["query"] == "tequila lime"
            assert payload["truncate"] is True
            assert len(payload["texts"]) == 1
            headers = call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer test-key"

    @pytest.mark.anyio
    async def test_rerank_mismatched_results_returns_original(self):
        """Test that reranker returns original list when result count doesn't match."""
        options = self._make_options(endpoint="http://localhost:8990")
        service = RerankerService(reranker_options=options)

        cocktails = [create_test_cocktail_model("1", "Margarita"), create_test_cocktail_model("2", "Mojito")]

        mock_response = MagicMock()
        # Return wrong number of results
        mock_response.json.return_value = [{"index": 0, "score": 0.9}]
        mock_response.raise_for_status = MagicMock()

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.services.reranker_service.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await service.rerank(query="tequila", cocktails=cocktails)

        assert result == cocktails

    @pytest.mark.anyio
    async def test_rerank_applies_relative_score_cutoff(self):
        """Test that relative score cutoff drops results below a fraction of the top score."""
        options = self._make_options(endpoint="http://localhost:8990", relative_score_cutoff=0.05)
        service = RerankerService(reranker_options=options)

        cocktails = [
            create_test_cocktail_model("1", "Michelada"),
            create_test_cocktail_model("2", "Margarita"),
            create_test_cocktail_model("3", "Bamboo"),
            create_test_cocktail_model("4", "Aviation"),
        ]

        # Top score=0.80, cutoff=0.80*0.05=0.04
        # Michelada 0.80 >= 0.04 -> keep
        # Margarita 0.054 >= 0.04 -> keep
        # Bamboo 0.036 < 0.04 -> drop
        # Aviation 0.014 < 0.04 -> drop
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"index": 0, "score": 0.80},
            {"index": 1, "score": 0.054},
            {"index": 2, "score": 0.036},
            {"index": 3, "score": 0.014},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.services.reranker_service.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await service.rerank(query="cocktails with salt", cocktails=cocktails)

        assert len(result) == 2
        assert result[0].title == "Michelada"
        assert result[1].title == "Margarita"

    @pytest.mark.anyio
    async def test_rerank_relative_cutoff_zero_disables(self):
        """Test that relative_score_cutoff=0.0 does not filter any results."""
        options = self._make_options(endpoint="http://localhost:8990", relative_score_cutoff=0.0)
        service = RerankerService(reranker_options=options)

        cocktails = [
            create_test_cocktail_model("1", "High"),
            create_test_cocktail_model("2", "Low"),
        ]

        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"index": 0, "score": 0.90},
            {"index": 1, "score": 0.001},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.services.reranker_service.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await service.rerank(query="test", cocktails=cocktails)

        assert len(result) == 2

    @pytest.mark.anyio
    async def test_rerank_relative_cutoff_combined_with_absolute_threshold(self):
        """Test that both absolute threshold and relative cutoff are applied."""
        options = self._make_options(
            endpoint="http://localhost:8990",
            score_threshold=0.03,
            relative_score_cutoff=0.1,
        )
        service = RerankerService(reranker_options=options)

        cocktails = [
            create_test_cocktail_model("1", "Top"),
            create_test_cocktail_model("2", "Mid"),
            create_test_cocktail_model("3", "Low"),
        ]

        # Score threshold=0.03 filters out Low (0.02)
        # Relative cutoff=0.1 of top (0.80)=0.08, filters out Mid (0.05)
        # Only Top survives both
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"index": 0, "score": 0.80},
            {"index": 1, "score": 0.05},
            {"index": 2, "score": 0.02},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.services.reranker_service.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await service.rerank(query="test", cocktails=cocktails)

        assert len(result) == 1
        assert result[0].title == "Top"


class TestBuildDocumentText:
    """Test cases for _build_document_text."""

    def test_basic_cocktail(self):
        """Test document text for a basic cocktail."""
        cocktail = create_test_cocktail_model("1", "Margarita")
        text = RerankerService._build_document_text(cocktail)
        assert "Margarita" in text
        assert "Description" in text

    def test_cocktail_with_ingredients(self):
        """Test document text includes ingredients."""
        cocktail = create_test_cocktail_model("1", "Margarita")
        cocktail.ingredients = [_make_ingredient("Tequila"), _make_ingredient("Lime Juice")]
        text = RerankerService._build_document_text(cocktail)
        assert "Tequila" in text
        assert "Lime Juice" in text
        assert "Ingredients:" in text

    def test_cocktail_same_title_and_descriptive_title(self):
        """Test that duplicate title/descriptive_title is not repeated."""
        cocktail = create_test_cocktail_model("1", "Margarita")
        cocktail.descriptive_title = "Margarita"
        text = RerankerService._build_document_text(cocktail)
        # Should not have "Margarita" twice as separate parts
        parts = text.split(". ")
        assert parts[0] == "Margarita"
        assert len([p for p in parts if p == "Margarita"]) == 1
