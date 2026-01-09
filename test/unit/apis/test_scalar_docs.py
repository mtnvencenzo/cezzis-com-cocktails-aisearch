from unittest.mock import MagicMock, patch

import pytest

from cezzis_com_cocktails_aisearch.apis.scalar_docs import ScalarDocsRouter


class TestScalarDocsRouter:
    """Test cases for ScalarDocsRouter."""

    def test_init(self):
        """Test router initialization."""
        mediator = MagicMock()
        router = ScalarDocsRouter(mediator=mediator)

        assert router.mediator == mediator
        assert len(router.routes) > 0

    def test_route_configuration(self):
        """Test that the scalar docs route is configured correctly."""
        mediator = MagicMock()
        router = ScalarDocsRouter(mediator=mediator)

        # Verify route was added
        assert len(router.routes) > 0

    @pytest.mark.anyio
    @patch("cezzis_com_cocktails_aisearch.apis.scalar_docs.get_scalar_api_reference")
    async def test_scalar_html(self, mock_get_scalar):
        """Test scalar_html returns the correct response."""
        mock_get_scalar.return_value = {"openapi": "3.0.0"}

        mediator = MagicMock()
        router = ScalarDocsRouter(mediator=mediator)

        result = await router.scalar_html()

        assert result == {"openapi": "3.0.0"}
        mock_get_scalar.assert_called_once_with(
            openapi_url="/openapi.json",
            scalar_favicon_url="/static/favicon.svg",
            title="Cezzi's Cocktails AI Search Api",
        )
