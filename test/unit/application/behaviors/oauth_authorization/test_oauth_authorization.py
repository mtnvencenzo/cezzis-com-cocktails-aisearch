import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from cezzis_com_cocktails_aisearch.application.behaviors.oauth_authorization.oauth_authorization import (
    _wrap_class,
    _wrap_function,
    oauth_authorization,
)


class TestOAuthAuthorization:
    """Test cases for oauth_authorization decorator."""

    @pytest.mark.anyio
    async def test_authorization_success(self):
        """Test successful OAuth authorization with valid token."""

        @oauth_authorization(scopes=["write:embeddings"])
        async def mock_endpoint(**kwargs):
            return "success"

        mock_request = MagicMock()
        mock_request.headers.get.return_value = "Bearer valid-token"

        mock_payload = {"sub": "user123", "scope": "write:embeddings"}

        with patch(
            "cezzis_com_cocktails_aisearch.application.behaviors.oauth_authorization.oauth_authorization.get_oauth_options"
        ):
            with patch(
                "cezzis_com_cocktails_aisearch.application.behaviors.oauth_authorization.oauth_authorization.OAuth2TokenVerifier"
            ) as mock_verifier_class:
                mock_verifier = AsyncMock()
                mock_verifier.verify_token = AsyncMock(return_value=mock_payload)
                mock_verifier.verify_scopes = MagicMock()
                mock_verifier_class.return_value = mock_verifier

                result = await mock_endpoint(_rq=mock_request)

        assert result == "success"
        mock_verifier.verify_token.assert_called_once()
        mock_verifier.verify_scopes.assert_called_once()

    @pytest.mark.anyio
    async def test_authorization_bypassed_in_local_env(self):
        """Test authorization bypassed in local environment."""

        @oauth_authorization(scopes=["write:embeddings"])
        async def mock_endpoint(**kwargs):
            return "success"

        mock_request = MagicMock()
        mock_request.headers.get.return_value = "Bearer any-token"

        with patch.dict(os.environ, {"ENV": "local"}):
            result = await mock_endpoint(_rq=mock_request)

        assert result == "success"

    @pytest.mark.anyio
    async def test_missing_authorization_header(self):
        """Test error when Authorization header is missing."""

        @oauth_authorization(scopes=["write:embeddings"])
        async def mock_endpoint(**kwargs):
            return "success"

        mock_request = MagicMock()
        mock_request.headers.get.return_value = ""

        with pytest.raises(HTTPException) as exc_info:
            await mock_endpoint(_rq=mock_request)

        assert exc_info.value.status_code == 401
        assert "Missing or invalid authorization token" in exc_info.value.detail

    @pytest.mark.anyio
    async def test_invalid_bearer_format(self):
        """Test error when Authorization header doesn't start with 'Bearer '."""

        @oauth_authorization(scopes=["write:embeddings"])
        async def mock_endpoint(**kwargs):
            return "success"

        mock_request = MagicMock()
        mock_request.headers.get.return_value = "Basic invalid"

        with pytest.raises(HTTPException) as exc_info:
            await mock_endpoint(_rq=mock_request)

        assert exc_info.value.status_code == 401

    @pytest.mark.anyio
    async def test_token_verification_error(self):
        """Test error when token verification fails."""
        from cezzis_oauth import TokenVerificationError

        @oauth_authorization(scopes=["write:embeddings"])
        async def mock_endpoint(**kwargs):
            return "success"

        mock_request = MagicMock()
        mock_request.headers.get.return_value = "Bearer invalid-token"

        with patch(
            "cezzis_com_cocktails_aisearch.application.behaviors.oauth_authorization.oauth_authorization.get_oauth_options"
        ):
            with patch(
                "cezzis_com_cocktails_aisearch.application.behaviors.oauth_authorization.oauth_authorization.OAuth2TokenVerifier"
            ) as mock_verifier_class:
                mock_verifier = AsyncMock()
                mock_verifier.verify_token = AsyncMock(side_effect=TokenVerificationError("Token invalid"))
                mock_verifier_class.return_value = mock_verifier

                with pytest.raises(HTTPException) as exc_info:
                    await mock_endpoint(_rq=mock_request)

                assert exc_info.value.status_code == 403

    @pytest.mark.anyio
    async def test_no_request_object(self):
        """Test error when Request object is not found in kwargs."""

        @oauth_authorization(scopes=["write:embeddings"])
        async def mock_endpoint(**kwargs):
            return "success"

        with pytest.raises(HTTPException) as exc_info:
            await mock_endpoint()

        assert exc_info.value.status_code == 500
        assert "Internal server error" in exc_info.value.detail

    @pytest.mark.anyio
    async def test_authorization_without_scopes(self):
        """Test authorization without scope verification."""

        @oauth_authorization()
        async def mock_endpoint(**kwargs):
            return "success"

        mock_request = MagicMock()
        mock_request.headers.get.return_value = "Bearer valid-token"

        mock_payload = {"sub": "user123"}

        with patch(
            "cezzis_com_cocktails_aisearch.application.behaviors.oauth_authorization.oauth_authorization.get_oauth_options"
        ):
            with patch(
                "cezzis_com_cocktails_aisearch.application.behaviors.oauth_authorization.oauth_authorization.OAuth2TokenVerifier"
            ) as mock_verifier_class:
                mock_verifier = AsyncMock()
                mock_verifier.verify_token = AsyncMock(return_value=mock_payload)
                mock_verifier.verify_scopes = MagicMock()
                mock_verifier_class.return_value = mock_verifier

                result = await mock_endpoint(_rq=mock_request)

        assert result == "success"
        mock_verifier.verify_token.assert_called_once()
        # verify_scopes should not be called when scopes is None/empty
        mock_verifier.verify_scopes.assert_not_called()

    @pytest.mark.anyio
    async def test_unexpected_error_during_authorization(self):
        """Test unexpected error during authorization."""

        @oauth_authorization(scopes=["write:embeddings"])
        async def mock_endpoint(**kwargs):
            return "success"

        mock_request = MagicMock()
        mock_request.headers.get.return_value = "Bearer valid-token"

        with patch(
            "cezzis_com_cocktails_aisearch.application.behaviors.oauth_authorization.oauth_authorization.get_oauth_options"
        ):
            with patch(
                "cezzis_com_cocktails_aisearch.application.behaviors.oauth_authorization.oauth_authorization.OAuth2TokenVerifier"
            ) as mock_verifier_class:
                mock_verifier = AsyncMock()
                mock_verifier.verify_token = AsyncMock(side_effect=Exception("Unexpected error"))
                mock_verifier_class.return_value = mock_verifier

                with pytest.raises(HTTPException) as exc_info:
                    await mock_endpoint(_rq=mock_request)

                assert exc_info.value.status_code == 500
                assert "Authorization error" in exc_info.value.detail
