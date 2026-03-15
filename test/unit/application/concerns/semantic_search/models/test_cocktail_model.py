from conftest import create_test_cocktail_model


class TestCocktailSearchModel:
    """Test cases for CocktailSearchModel."""

    def test_keywords_search_terms_excluded_from_serialization(self):
        """Test that keywords_search_terms is excluded from model_dump and JSON output."""
        cocktail = create_test_cocktail_model("1", "Michelada")
        cocktail.keywords_search_terms = ["hangover cure", "brunch", "beer cocktail"]

        dumped = cocktail.model_dump()
        assert "keywords_search_terms" not in dumped

        json_str = cocktail.model_dump_json()
        assert "keywords_search_terms" not in json_str
        assert "hangover cure" not in json_str

    def test_keywords_search_terms_accessible_on_model(self):
        """Test that keywords_search_terms is still accessible as a model attribute."""
        cocktail = create_test_cocktail_model("1", "Michelada")
        cocktail.keywords_search_terms = ["hangover cure", "brunch"]

        assert cocktail.keywords_search_terms == ["hangover cure", "brunch"]

    def test_keywords_search_terms_defaults_to_empty_list(self):
        """Test that keywords_search_terms defaults to empty list."""
        cocktail = create_test_cocktail_model("1", "Michelada")
        assert cocktail.keywords_search_terms == []
