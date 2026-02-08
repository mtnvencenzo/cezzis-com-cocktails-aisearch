import logging
import re
from typing import Optional

from injector import inject
from mediatr import GenericQuery, Mediator
from qdrant_client.http.models import Condition, FieldCondition, Filter, MatchValue, Range

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_data_include_model import (
    CocktailDataIncludeModel,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailModel
from cezzis_com_cocktails_aisearch.domain.config.qdrant_options import QdrantOptions
from cezzis_com_cocktails_aisearch.infrastructure.repositories.icocktail_vector_repository import (
    ICocktailVectorRepository,
)


class FreeTextQuery(GenericQuery[list[tuple[str, float]]]):
    def __init__(
        self,
        free_text: Optional[str] = "",
        skip: Optional[int] = 0,
        take: Optional[int] = 10,
        matches: Optional[list[str]] = [],
        match_exclusive: Optional[bool] = False,
        include: Optional[list[CocktailDataIncludeModel]] = [],
        filters: Optional[list[str]] = [],
    ):
        self.free_text = free_text
        self.skip = skip
        self.take = take
        self.matches = matches
        self.match_exclusive = match_exclusive
        self.include = include
        self.filters = filters


@Mediator.behavior
class FreeTextQueryValidator:
    def handle(self, command: FreeTextQuery, next) -> None:
        return next()


@Mediator.handler
class FreeTextQueryHandler:
    # Minimum query length for semantic search (embeddings struggle with very short text)
    _MIN_SEMANTIC_LENGTH: int = 4

    # Patterns that indicate ingredient exclusion
    _EXCLUSION_PATTERNS: list[str] = [
        "without ",
        "no ",
        "excluding ",
        "exclude ",
    ]

    # Glassware keyword to Qdrant payload value mapping
    _GLASSWARE_MAPPING: dict[str, str] = {
        "coupe": "coupe",
        "rocks glass": "rocks",
        "rocks": "rocks",
        "lowball": "lowball",
        "highball": "highball",
        "collins": "collins",
        "martini glass": "cocktailGlass",
        "cocktail glass": "cocktailGlass",
        "copper mug": "copperMug",
        "moscow mule mug": "copperMug",
        "wine glass": "wineGlass",
        "flute": "flute",
        "champagne flute": "flute",
        "tiki mug": "tikiMug",
        "hurricane": "hurricane",
        "snifter": "snifter",
        "shot glass": "shotGlass",
        "pint glass": "pintGlass",
    }

    @inject
    def __init__(
        self,
        cocktail_vector_repository: ICocktailVectorRepository,
        qdrant_opotions: QdrantOptions,
    ):
        self.cocktail_vector_repository = cocktail_vector_repository
        self.qdrant_options = qdrant_opotions
        self.logger = logging.getLogger("free_text_query_handler")

    async def handle(self, command: FreeTextQuery) -> list[CocktailModel]:
        if not command.free_text:
            return await self._handle_browse(command)

        search_text = command.free_text.strip().lower()

        # Fast path: exact cocktail name match (uses cached data)
        all_cocktails = await self.cocktail_vector_repository.get_all_cocktails()
        exact_match = self._find_exact_name_match(search_text, all_cocktails)
        if exact_match:
            take = command.take or 10
            return exact_match[:take]

        # Short queries: text-based fallback on cached data (embeddings struggle with < 4 chars)
        if len(search_text) < self._MIN_SEMANTIC_LENGTH:
            return self._handle_short_query(search_text, all_cocktails, command)

        # Build Qdrant payload filter from structured query elements
        query_filter = self._build_query_filter(search_text)

        # Semantic search with Qdrant-native filtering
        cocktails = await self.cocktail_vector_repository.search_vectors(
            free_text=command.free_text,
            query_filter=query_filter,
        )

        # Sort by weighted_score (combines avg score with hit count boost)
        sorted_cocktails = sorted(
            cocktails,
            key=lambda p: getattr(p.search_statistics, "weighted_score", 0) if p.search_statistics else 0,
            reverse=True,
        )

        # Apply rating-based sort override for rating queries
        if any(term in search_text for term in ["top rated", "best rated", "highest rated", "popular"]):
            sorted_cocktails = sorted(sorted_cocktails, key=lambda c: c.rating, reverse=True)

        skip = command.skip or 0
        take = command.take or 10
        return sorted_cocktails[skip : skip + take]

    async def _handle_browse(self, command: FreeTextQuery) -> list[CocktailModel]:
        """Handle browsing when no free text is provided."""
        cocktails = await self.cocktail_vector_repository.get_all_cocktails()
        use_matches = command.matches

        if command.match_exclusive and use_matches is None:
            use_matches = []
        elif not command.match_exclusive and command.matches is not None and len(command.matches) == 0:
            use_matches = None

        sorted_cocktails = sorted(cocktails, key=lambda p: p.title or "", reverse=False)
        filtered_cocktails = [c for c in sorted_cocktails if use_matches is None or (c.id in use_matches)]

        skip = command.skip or 0
        take = command.take or 10
        return filtered_cocktails[skip : skip + take]

    def _find_exact_name_match(
        self, search_text: str, all_cocktails: list[CocktailModel]
    ) -> list[CocktailModel] | None:
        """
        Check if the search text matches cocktail names.
        Returns cocktails where:
        1. Title exactly matches the search text (prioritized first)
        2. Title contains the search text as a substring

        This ensures "Mai Tai" returns both "Mai Tai" and "Bitter Mai Tai".
        """
        search_lower = search_text.lower().strip()

        # Remove common suffixes for matching
        search_normalized = search_lower
        for suffix in [" cocktail", " drink", " recipe"]:
            if search_normalized.endswith(suffix):
                search_normalized = search_normalized[: -len(suffix)].strip()

        # Only do name matching if the search looks like a cocktail name
        # (not a descriptive query like "cocktails with honey")
        descriptive_prefixes = ["cocktails ", "drinks ", "recipes ", "show me ", "find "]
        if any(search_lower.startswith(prefix) for prefix in descriptive_prefixes):
            return None

        # Find exact matches (highest priority)
        exact_matches = [c for c in all_cocktails if c.title.lower() == search_normalized]

        # Find partial matches (title contains the search term)
        partial_matches = [c for c in all_cocktails if search_normalized in c.title.lower() and c not in exact_matches]

        # Combine: exact matches first, then partial matches sorted alphabetically
        if exact_matches or partial_matches:
            partial_matches_sorted = sorted(partial_matches, key=lambda c: c.title)
            return exact_matches + partial_matches_sorted

        return None

    def _handle_short_query(
        self, search_text: str, all_cocktails: list[CocktailModel], command: FreeTextQuery
    ) -> list[CocktailModel]:
        """Handle queries too short for meaningful semantic search."""
        filtered = [c for c in all_cocktails if self._matches_text_search(c, search_text)]
        sorted_cocktails = sorted(filtered, key=lambda p: p.title or "")

        skip = command.skip or 0
        take = command.take or 10
        return sorted_cocktails[skip : skip + take]

    def _matches_text_search(self, cocktail: CocktailModel, search_text: str) -> bool:
        """Check if cocktail matches a text-based search (title, descriptive title, ingredients)."""
        if cocktail.title and search_text in cocktail.title.lower():
            return True
        if cocktail.descriptive_title and search_text in cocktail.descriptive_title.lower():
            return True
        if cocktail.ingredients:
            for ingredient in cocktail.ingredients:
                if ingredient.name and search_text in ingredient.name.lower():
                    return True
        return False

    def _build_query_filter(self, search_text: str) -> Filter | None:
        """
        Build Qdrant payload filter from structured query elements.

        Pushes filtering to the vector DB level for better performance and accuracy.
        Handles: IBA status, glassware, ingredient count, prep time, serves, and ingredient exclusion.
        """
        must_conditions: list[Condition] = []
        must_not_conditions: list[Condition] = []

        # IBA filter (check non-IBA first since "non-iba" contains "iba")
        if any(term in search_text for term in ["non-iba", "non iba", "modern cocktail", "contemporary"]):
            must_conditions.append(FieldCondition(key="metadata.is_iba", match=MatchValue(value=False)))
        elif any(term in search_text for term in ["iba ", "iba cocktail", "official cocktail", "classic iba"]):
            must_conditions.append(FieldCondition(key="metadata.is_iba", match=MatchValue(value=True)))

        # Glassware filter
        for term, glassware_value in self._GLASSWARE_MAPPING.items():
            if term in search_text:
                must_conditions.append(
                    FieldCondition(key="metadata.glassware_values", match=MatchValue(value=glassware_value))
                )
                break

        # Ingredient count filters
        ingredient_count_match = re.search(r"(\d+)\s*ingredient", search_text)
        if ingredient_count_match:
            target_count = int(ingredient_count_match.group(1))
            must_conditions.append(
                FieldCondition(key="metadata.ingredient_count", range=Range(gte=target_count, lte=target_count))
            )
        elif any(term in search_text for term in ["simple", "easy", "few ingredients", "basic"]):
            must_conditions.append(FieldCondition(key="metadata.ingredient_count", range=Range(lte=4)))
        elif any(term in search_text for term in ["complex", "many ingredients", "elaborate"]):
            must_conditions.append(FieldCondition(key="metadata.ingredient_count", range=Range(gte=6)))

        # Prep time filters
        if any(term in search_text for term in ["quick", "fast", "5 minute", "5-minute"]):
            must_conditions.append(FieldCondition(key="metadata.prep_time_minutes", range=Range(lte=5)))
        elif any(term in search_text for term in ["10 minute", "10-minute"]):
            must_conditions.append(FieldCondition(key="metadata.prep_time_minutes", range=Range(lte=10)))

        # Serves filter
        serves_match = re.search(r"serves?\s*(\d+)", search_text)
        if serves_match:
            target_serves = int(serves_match.group(1))
            must_conditions.append(FieldCondition(key="metadata.serves", match=MatchValue(value=target_serves)))

        # Ingredient exclusion (e.g., "without honey", "no rum")
        excluded_terms = self._extract_exclusion_terms(search_text)
        for term in excluded_terms:
            must_not_conditions.append(FieldCondition(key="metadata.ingredient_words", match=MatchValue(value=term)))

        if must_conditions or must_not_conditions:
            return Filter(
                must=must_conditions if must_conditions else None,
                must_not=must_not_conditions if must_not_conditions else None,
            )
        return None

    def _extract_exclusion_terms(self, search_text: str) -> list[str]:
        """Extract ingredient terms that should be excluded from results."""
        terms: list[str] = []
        for pattern in self._EXCLUSION_PATTERNS:
            idx = search_text.find(pattern)
            while idx >= 0:
                after = search_text[idx + len(pattern) :].split()
                if after:
                    term = after[0].strip(",.!?").lower()
                    if len(term) >= 2 and term not in terms:
                        terms.append(term)
                idx = search_text.find(pattern, idx + len(pattern))
        return terms
