import os
from unittest.mock import patch

import pytest

from cezzis_com_cocktails_aisearch.domain.config.hugging_face_options import (
    HuggingFaceOptions,
    clear_huggingface_options_cache,
    get_huggingface_options,
)


class TestHuggingFaceOptions:
    """ "Test cases for HuggingFaceOptions configuration."""

    def test_huggingface_options_init_with_defaults(self):
        """ "Test HuggingFaceOptions initialization with default values."""
        with patch.dict(os.environ, {}, clear=True):
            options = HuggingFaceOptions()

            assert options.inference_model == ""
            assert options.api_token == ""

    def test_huggingface_options_init_with_env_vars(self):
        """ "Test HuggingFaceOptions initialization with environment variables."""
        with patch.dict(
            os.environ,
            {
                "HUGGINGFACE_INFERENCE_MODEL": "sentence-transformers/all-mpnet-base-v2",
                "HUGGINGFACE_API_TOKEN": "hf_test_token_123",
            },
        ):
            options = HuggingFaceOptions()

            assert options.inference_model == "sentence-transformers/all-mpnet-base-v2"
            assert options.api_token == "hf_test_token_123"

    def test_get_huggingface_options_raises_on_missing_model(self):
        """Test that get_huggingface_options raises ValueError when model is missing."""
        clear_huggingface_options_cache()

        with patch.dict(os.environ, {"HUGGINGFACE_INFERENCE_MODEL": "", "HUGGINGFACE_API_TOKEN": "test-token"}):
            with pytest.raises(ValueError, match="HUGGINGFACE_INFERENCE_MODEL"):
                get_huggingface_options()

    def test_get_huggingface_options_raises_on_missing_token(self):
        """Test that get_huggingface_options raises ValueError when token is missing."""
        clear_huggingface_options_cache()

        with patch.dict(os.environ, {"HUGGINGFACE_INFERENCE_MODEL": "test-model", "HUGGINGFACE_API_TOKEN": ""}):
            with pytest.raises(ValueError, match="HUGGINGFACE_API_TOKEN"):
                get_huggingface_options()

    def test_get_huggingface_options_singleton(self):
        """Test that get_huggingface_options returns a singleton instance."""
        clear_huggingface_options_cache()

        with patch.dict(
            os.environ, {"HUGGINGFACE_INFERENCE_MODEL": "test-model", "HUGGINGFACE_API_TOKEN": "test-token"}
        ):
            options1 = get_huggingface_options()
            options2 = get_huggingface_options()

            assert options1 is options2

    def test_clear_huggingface_options_cache(self):
        """Test that clear_huggingface_options_cache resets the singleton."""
        clear_huggingface_options_cache()

        with patch.dict(os.environ, {"HUGGINGFACE_INFERENCE_MODEL": "model1", "HUGGINGFACE_API_TOKEN": "token1"}):
            options1 = get_huggingface_options()
            assert options1.inference_model == "model1"

        clear_huggingface_options_cache()

        with patch.dict(os.environ, {"HUGGINGFACE_INFERENCE_MODEL": "model2", "HUGGINGFACE_API_TOKEN": "token2"}):
            options2 = get_huggingface_options()
            assert options2.inference_model == "model2"
            assert options1 is not options2
