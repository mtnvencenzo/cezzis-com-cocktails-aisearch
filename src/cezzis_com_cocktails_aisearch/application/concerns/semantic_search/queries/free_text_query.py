import difflib
import logging
from typing import Optional

from injector import inject
from mediatr import GenericQuery, Mediator

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
        if command.free_text:
            search_text = command.free_text.strip().lower()

            # Check for exact cocktail name match first
            all_cocktails = await self.cocktail_vector_repository.get_all_cocktails()
            exact_match = self._find_exact_name_match(search_text, all_cocktails)
            if exact_match:
                return exact_match[: command.take or 10]

            # For short queries (< 4 chars), use text-based matching instead of semantic search
            # Semantic embeddings don't work well with partial words like "hon" vs "honey"
            min_semantic_length = 4

            if len(search_text) < min_semantic_length:
                # Fall back to text-based prefix/contains matching on titles and ingredients
                filtered_cocktails = [c for c in all_cocktails if self._matches_text_search(c, search_text)]

                # Sort alphabetically for text-based results
                sorted_cocktails = sorted(
                    filtered_cocktails,
                    key=lambda p: p.title or "",
                    reverse=False,
                )
            else:
                # Use semantic vector search for longer, more meaningful queries
                cocktails = await self.cocktail_vector_repository.search_vectors(free_text=command.free_text)

                # Sort by weighted_score which combines avg score with hit count boost
                sorted_cocktails = sorted(
                    cocktails,
                    key=lambda p: getattr(p.search_statistics, "weighted_score", 0) if p.search_statistics else 0,
                    reverse=True,
                )

                # Filter by total_score threshold (sum of all hits) for backward compatibility
                sorted_cocktails = [
                    c
                    for c in sorted_cocktails
                    if c.search_statistics
                    and c.search_statistics.total_score > self.qdrant_options.semantic_search_total_score_threshold
                ]

                # Post-filter: verify ingredient mentions actually exist (or don't exist for exclusions)
                # This prevents false positives where semantic similarity doesn't match actual content
                include_terms, exclude_terms = self._extract_ingredient_terms(search_text, all_cocktails)

                self.logger.info(
                    f"Ingredient filtering: include_terms={include_terms}, exclude_terms={exclude_terms}, "
                    f"cocktails_before_filter={len(sorted_cocktails)}"
                )

                if include_terms:
                    before_count = len(sorted_cocktails)
                    sorted_cocktails = [
                        c for c in sorted_cocktails if self._cocktail_matches_ingredient_terms(c, include_terms)
                    ]
                    self.logger.info(
                        f"After include filter: {before_count} -> {len(sorted_cocktails)} "
                        f"(removed {before_count - len(sorted_cocktails)})"
                    )

                if exclude_terms:
                    before_count = len(sorted_cocktails)
                    sorted_cocktails = [
                        c for c in sorted_cocktails if not self._cocktail_matches_ingredient_terms(c, exclude_terms)
                    ]
                    self.logger.info(
                        f"After exclude filter: {before_count} -> {len(sorted_cocktails)} "
                        f"(removed {before_count - len(sorted_cocktails)})"
                    )

            # Apply structured filters (IBA, glassware, ingredient count, prep time, etc.)
            sorted_cocktails = self._apply_structured_filters(search_text, sorted_cocktails)

            skip = command.skip or 0
            take = command.take or 10
            return sorted_cocktails[skip : skip + take]

        else:
            cocktails = await self.cocktail_vector_repository.get_all_cocktails()
            useMatches = command.matches

            if command.match_exclusive and useMatches is None:
                useMatches = []
            elif not command.match_exclusive and command.matches is not None and len(command.matches) == 0:
                useMatches = None

            sorted_cocktails = sorted(
                cocktails,
                key=lambda p: p.title or "",
                reverse=False,
            )

            filtered_cocktails = [c for c in sorted_cocktails if useMatches is None or (c.id in useMatches)]

            filtered_cocktails = [
                c
                for c in filtered_cocktails
                if not command.free_text or c.title.lower().startswith(command.free_text.lower())
            ]

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

    def _apply_structured_filters(self, search_text: str, cocktails: list[CocktailModel]) -> list[CocktailModel]:
        """
        Apply structured filters based on query patterns that vector search can't handle.
        Handles: IBA status, glassware, ingredient count, prep time, etc.
        """
        filtered = cocktails

        # IBA cocktails
        if any(term in search_text for term in ["iba ", "iba cocktail", "official cocktail", "classic iba"]):
            filtered = [c for c in filtered if c.isIba]

        # Non-IBA / modern cocktails
        if any(term in search_text for term in ["non-iba", "non iba", "modern cocktail", "contemporary"]):
            filtered = [c for c in filtered if not c.isIba]

        # Glassware filters
        glassware_mapping = {
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

        for term, glassware_value in glassware_mapping.items():
            if term in search_text:
                filtered = [c for c in filtered if any(g.value == glassware_value for g in c.glassware)]
                break

        # Ingredient count filters
        if any(term in search_text for term in ["simple", "easy", "few ingredients", "basic"]):
            filtered = [c for c in filtered if len(c.ingredients) <= 4]

        if any(term in search_text for term in ["complex", "many ingredients", "elaborate"]):
            filtered = [c for c in filtered if len(c.ingredients) >= 6]

        # Numeric ingredient count (e.g., "3 ingredient cocktails")
        import re

        ingredient_count_match = re.search(r"(\d+)\s*ingredient", search_text)
        if ingredient_count_match:
            target_count = int(ingredient_count_match.group(1))
            filtered = [c for c in filtered if len(c.ingredients) == target_count]

        # Prep time filters
        if any(term in search_text for term in ["quick", "fast", "5 minute", "5-minute"]):
            filtered = [c for c in filtered if c.prepTimeMinutes <= 5]

        if any(term in search_text for term in ["10 minute", "10-minute"]):
            filtered = [c for c in filtered if c.prepTimeMinutes <= 10]

        # Servings
        serves_match = re.search(r"serves?\s*(\d+)", search_text)
        if serves_match:
            target_serves = int(serves_match.group(1))
            filtered = [c for c in filtered if c.serves == target_serves]

        # Highly rated
        if any(term in search_text for term in ["top rated", "best rated", "highest rated", "popular"]):
            filtered = sorted(filtered, key=lambda c: c.rating, reverse=True)

        return filtered

    def _matches_text_search(self, cocktail: CocktailModel, search_text: str) -> bool:
        """
        Check if cocktail matches a text-based search query.
        Searches in title, descriptive title, and ingredient names.
        """
        # Check title
        if cocktail.title and search_text in cocktail.title.lower():
            return True

        # Check descriptive title
        if cocktail.descriptiveTitle and search_text in cocktail.descriptiveTitle.lower():
            return True

        # Check ingredient names
        if cocktail.ingredients:
            for ingredient in cocktail.ingredients:
                if ingredient.name and search_text in ingredient.name.lower():
                    return True

        return False

    # Patterns that indicate ingredient exclusion
    _EXCLUSION_PATTERNS: list[str] = [
        "without ",
        "no ",
        "excluding ",
        "exclude ",
        "not containing ",
        "doesn't have ",
        "doesn't contain ",
        "does not have ",
        "does not contain ",
        "lacking ",
        "missing ",
        "free of ",
        "-free",  # e.g., "alcohol-free"
    ]

    # Words that should not be treated as ingredient terms even if they appear
    # in ingredient names (e.g., "Cocktail Onion" contains "cocktail")
    _IGNORED_INGREDIENT_WORDS: set[str] = {
        "cocktail",
        "cocktails",
        "drink",
        "drinks",
        "recipe",
        "recipes",
        "with",
        "and",
        "the",
        "for",
        "from",
    }

    # Similarity cutoff for fuzzy matching ingredient terms (0.0 to 1.0)
    # 0.8 means words need to be at least 80% similar to match
    # This handles typos like "burbon" -> "bourbon", "vodca" -> "vodka"
    _FUZZY_MATCH_CUTOFF: float = 0.8

    def _extract_ingredient_terms(
        self, search_text: str, all_cocktails: list[CocktailModel]
    ) -> tuple[list[str], list[str]]:
        """
        Extract ingredient terms from the search query by matching against
        actual ingredients in the cocktail database.

        Returns a tuple of (include_terms, exclude_terms):
        - include_terms: ingredients that MUST be in the cocktail
        - exclude_terms: ingredients that must NOT be in the cocktail

        This handles negation patterns like "without honey", "no rum", etc.
        """
        search_lower = search_text.lower()
        include_terms: list[str] = []
        exclude_terms: list[str] = []

        # First, collect all unique ingredient words from the database
        all_ingredient_words: set[str] = set()
        all_ingredient_names: set[str] = set()

        for cocktail in all_cocktails:
            if cocktail.ingredients:
                for ingredient in cocktail.ingredients:
                    if ingredient.name:
                        ingredient_lower = ingredient.name.lower()
                        all_ingredient_names.add(ingredient_lower)
                        for word in ingredient_lower.split():
                            # Skip common words that shouldn't be treated as ingredients
                            if len(word) >= 3 and word not in self._IGNORED_INGREDIENT_WORDS:
                                all_ingredient_words.add(word)

        # Extract query words for fuzzy matching
        query_words = [w.strip(",.()[]?!") for w in search_lower.split()]
        query_words = [w for w in query_words if len(w) >= 3 and w not in self._IGNORED_INGREDIENT_WORDS]

        # Build a map of query words to their fuzzy-matched ingredient words
        # This handles typos like "burbon" -> "bourbon"
        fuzzy_matches: dict[str, str] = {}
        for query_word in query_words:
            if query_word in all_ingredient_words:
                # Exact match
                fuzzy_matches[query_word] = query_word
            else:
                # Try fuzzy matching
                close_matches = difflib.get_close_matches(
                    query_word, list(all_ingredient_words), n=1, cutoff=self._FUZZY_MATCH_CUTOFF
                )
                if close_matches:
                    matched_word = close_matches[0]
                    fuzzy_matches[query_word] = matched_word
                    self.logger.info(f"Fuzzy matched '{query_word}' -> '{matched_word}'")

        # Now find which ingredients are mentioned and whether they're excluded
        for query_word, ingredient_word in fuzzy_matches.items():
            # Check if this ingredient is preceded by an exclusion pattern
            # Use both the original query word and the matched ingredient word for pattern detection
            is_excluded = False
            for pattern in self._EXCLUSION_PATTERNS:
                # Check for patterns like "without honey" or "honey-free"
                if pattern.endswith("-"):
                    # Handle suffix patterns like "-free"
                    if f"{query_word}{pattern}" in search_lower:
                        is_excluded = True
                        break
                else:
                    # Handle prefix patterns like "without ", "no "
                    if f"{pattern}{query_word}" in search_lower:
                        is_excluded = True
                        break

            if is_excluded:
                if ingredient_word not in exclude_terms:
                    exclude_terms.append(ingredient_word)
            else:
                if ingredient_word not in include_terms:
                    include_terms.append(ingredient_word)

        # Also check full ingredient names
        for ingredient_name in all_ingredient_names:
            if ingredient_name in search_lower:
                is_excluded = False
                for pattern in self._EXCLUSION_PATTERNS:
                    if pattern.endswith("-"):
                        if f"{ingredient_name}{pattern}" in search_lower:
                            is_excluded = True
                            break
                    else:
                        if f"{pattern}{ingredient_name}" in search_lower:
                            is_excluded = True
                            break

                if is_excluded:
                    if ingredient_name not in exclude_terms:
                        exclude_terms.append(ingredient_name)
                else:
                    if ingredient_name not in include_terms:
                        include_terms.append(ingredient_name)

        return include_terms, exclude_terms

    def _cocktail_matches_ingredient_terms(self, cocktail: CocktailModel, ingredient_terms: list[str]) -> bool:
        """
        Verify that a cocktail actually contains the specified ingredient terms.
        Returns True if the cocktail has at least one of the ingredient terms
        in its title, descriptive title, or ingredient list.
        """
        # Build a searchable text from the cocktail's content
        searchable_parts = []

        if cocktail.title:
            searchable_parts.append(cocktail.title.lower())

        if cocktail.descriptiveTitle:
            searchable_parts.append(cocktail.descriptiveTitle.lower())

        if cocktail.ingredients:
            for ingredient in cocktail.ingredients:
                if ingredient.name:
                    searchable_parts.append(ingredient.name.lower())

        searchable_text = " ".join(searchable_parts)

        # Check if any of the ingredient terms appear in the cocktail
        for term in ingredient_terms:
            if term in searchable_text:
                return True

        return False
