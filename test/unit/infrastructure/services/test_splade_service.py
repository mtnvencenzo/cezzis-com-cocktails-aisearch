from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from cezzis_com_cocktails_aisearch.infrastructure.services.splade_service import SpladeService


class TestSpladeService:
    """Test cases for SpladeService."""

    def _make_options(self, endpoint="http://localhost:8991", api_key=""):
        options = MagicMock()
        options.endpoint = endpoint
        options.api_key = api_key
        return options

    @pytest.mark.anyio
    async def test_encode_success(self):
        """Test successful SPLADE encoding returns sparse vector."""
        options = self._make_options(endpoint="http://localhost:8991")
        service = SpladeService(splade_options=options)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [
                {"index": 42, "value": 0.8},
                {"index": 100, "value": 0.5},
                {"index": 7, "value": 0.3},
            ]
        ]
        mock_response.raise_for_status = MagicMock()

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.services.splade_service.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            indices, values = await service.encode("cocktails with gin")

        assert indices == [42, 100, 7]
        assert values == [0.8, 0.5, 0.3]

    @pytest.mark.anyio
    async def test_encode_sends_correct_payload(self):
        """Test that encode sends correct payload to TEI."""
        options = self._make_options(endpoint="http://localhost:8991", api_key="test-key")
        service = SpladeService(splade_options=options)

        mock_response = MagicMock()
        mock_response.json.return_value = [[{"index": 1, "value": 0.5}]]
        mock_response.raise_for_status = MagicMock()

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.services.splade_service.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await service.encode("tequila lime")

            call_args = mock_client.post.call_args
            assert call_args[0][0] == "http://localhost:8991/embed_sparse"
            payload = call_args[1]["json"]
            assert payload["inputs"] == ["tequila lime"]
            assert payload["truncate"] is True
            headers = call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer test-key"

    @pytest.mark.anyio
    async def test_encode_no_api_key_omits_auth_header(self):
        """Test that encode omits Authorization header when no API key."""
        options = self._make_options(endpoint="http://localhost:8991", api_key="")
        service = SpladeService(splade_options=options)

        mock_response = MagicMock()
        mock_response.json.return_value = [[{"index": 1, "value": 0.5}]]
        mock_response.raise_for_status = MagicMock()

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.services.splade_service.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await service.encode("test")

            headers = mock_client.post.call_args[1]["headers"]
            assert "Authorization" not in headers

    @pytest.mark.anyio
    async def test_encode_graceful_degradation_on_http_error(self):
        """Test that encode returns empty on HTTP error."""
        options = self._make_options(endpoint="http://localhost:8991")
        service = SpladeService(splade_options=options)

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.services.splade_service.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                side_effect=httpx.HTTPStatusError("Server Error", request=MagicMock(), response=MagicMock())
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            indices, values = await service.encode("test")

        assert indices == []
        assert values == []

    @pytest.mark.anyio
    async def test_encode_graceful_degradation_on_connection_error(self):
        """Test that encode returns empty when TEI is unreachable."""
        options = self._make_options(endpoint="http://localhost:8991")
        service = SpladeService(splade_options=options)

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.services.splade_service.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            indices, values = await service.encode("test")

        assert indices == []
        assert values == []

    @pytest.mark.anyio
    async def test_encode_batch_empty_input(self):
        """Test that empty input returns empty list."""
        options = self._make_options()
        service = SpladeService(splade_options=options)

        result = await service.encode_batch([])
        assert result == []

    @pytest.mark.anyio
    async def test_encode_batch_success(self):
        """Test successful batch SPLADE encoding."""
        options = self._make_options(endpoint="http://localhost:8991")
        service = SpladeService(splade_options=options)

        mock_response = MagicMock()
        mock_response.json.return_value = [
            [{"index": 10, "value": 0.9}, {"index": 20, "value": 0.4}],
            [{"index": 30, "value": 0.7}],
        ]
        mock_response.raise_for_status = MagicMock()

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.services.splade_service.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await service.encode_batch(["gin cocktail", "vodka cocktail"])

        assert len(result) == 2
        assert result[0] == ([10, 20], [0.9, 0.4])
        assert result[1] == ([30], [0.7])

    @pytest.mark.anyio
    async def test_encode_batch_graceful_degradation_on_error(self):
        """Test that batch encode returns empty on error."""
        options = self._make_options(endpoint="http://localhost:8991")
        service = SpladeService(splade_options=options)

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.services.splade_service.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await service.encode_batch(["text 1", "text 2"])

        assert len(result) == 2
        assert result[0] == ([], [])
        assert result[1] == ([], [])

    @pytest.mark.anyio
    async def test_encode_strips_trailing_slash_from_endpoint(self):
        """Test that trailing slash is stripped from endpoint URL."""
        options = self._make_options(endpoint="http://localhost:8991/")
        service = SpladeService(splade_options=options)

        mock_response = MagicMock()
        mock_response.json.return_value = [[{"index": 1, "value": 0.5}]]
        mock_response.raise_for_status = MagicMock()

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.services.splade_service.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await service.encode("test")

            url = mock_client.post.call_args[0][0]
            assert url == "http://localhost:8991/embed_sparse"

    @pytest.mark.anyio
    async def test_encode_empty_tei_response(self):
        """Test that empty TEI response returns empty sparse vector."""
        options = self._make_options(endpoint="http://localhost:8991")
        service = SpladeService(splade_options=options)

        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()

        with patch(
            "cezzis_com_cocktails_aisearch.infrastructure.services.splade_service.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            indices, values = await service.encode("test")

        assert indices == []
        assert values == []
