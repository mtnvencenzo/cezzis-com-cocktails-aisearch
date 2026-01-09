import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from cezzis_com_cocktails_aisearch.application.behaviors.apim_host_key_authorization.apim_host_key_authorization import (
    apim_host_key_authorization,
)


class TestApimHostKeyAuthorization:
    """Test cases for apim_host_key_authorization decorator."""

    @pytest.mark.anyio
    async def test_authorization_success(self):
        """Test successful authorization with valid host key."""

        @apim_host_key_authorization
        async def mock_endpoint(**kwargs):
            return "success"

        mock_request = MagicMock()
        mock_request.headers.get.return_value = "valid-key"

        with patch(
            "cezzis_com_cocktails_aisearch.application.behaviors.apim_host_key_authorization.apim_host_key_authorization._app_options"
        ) as mock_options:
            mock_options.apim_host_key = "valid-key"
            result = await mock_endpoint(_rq=mock_request)

        assert result == "success"

    @pytest.mark.anyio
    async def test_authorization_failure_invalid_key(self):
        """Test authorization failure with invalid host key."""

        @apim_host_key_authorization
        async def mock_endpoint(**kwargs):
            return "success"

        mock_request = MagicMock()
        mock_request.headers.get.return_value = "invalid-key"

        with patch(
            "cezzis_com_cocktails_aisearch.application.behaviors.apim_host_key_authorization.apim_host_key_authorization._app_options"
        ) as mock_options:
            mock_options.apim_host_key = "valid-key"

            with pytest.raises(HTTPException) as exc_info:
                await mock_endpoint(_rq=mock_request)

            assert exc_info.value.status_code == 403
            assert exc_info.value.detail == "Invalid host key"

    @pytest.mark.anyio
    async def test_authorization_bypassed_empty_key_local_env(self):
        """Test authorization bypassed when host key is empty in local environment."""

        @apim_host_key_authorization
        async def mock_endpoint(**kwargs):
            return "success"

        mock_request = MagicMock()
        mock_request.headers.get.return_value = "any-key"

        with patch(
            "cezzis_com_cocktails_aisearch.application.behaviors.apim_host_key_authorization.apim_host_key_authorization._app_options"
        ) as mock_options:
            with patch.dict(os.environ, {"ENV": "local"}):
                mock_options.apim_host_key = ""
                result = await mock_endpoint(_rq=mock_request)

        assert result == "success"

    @pytest.mark.anyio
    async def test_authorization_bypassed_empty_key_non_local_env(self):
        """Test authorization bypassed when host key is empty in non-local environment with warning."""

        @apim_host_key_authorization
        async def mock_endpoint(**kwargs):
            return "success"

        mock_request = MagicMock()
        mock_request.headers.get.return_value = "any-key"

        with patch(
            "cezzis_com_cocktails_aisearch.application.behaviors.apim_host_key_authorization.apim_host_key_authorization._app_options"
        ) as mock_options:
            with patch.dict(os.environ, {"ENV": "production"}):
                mock_options.apim_host_key = "   "  # Empty after strip
                result = await mock_endpoint(_rq=mock_request)

        assert result == "success"

    @pytest.mark.anyio
    async def test_missing_host_key_header(self):
        """Test with missing host key header."""

        @apim_host_key_authorization
        async def mock_endpoint(**kwargs):
            return "success"

        mock_request = MagicMock()
        mock_request.headers.get.return_value = ""

        with patch(
            "cezzis_com_cocktails_aisearch.application.behaviors.apim_host_key_authorization.apim_host_key_authorization._app_options"
        ) as mock_options:
            mock_options.apim_host_key = "valid-key"

            with pytest.raises(HTTPException) as exc_info:
                await mock_endpoint(_rq=mock_request)

            assert exc_info.value.status_code == 403
