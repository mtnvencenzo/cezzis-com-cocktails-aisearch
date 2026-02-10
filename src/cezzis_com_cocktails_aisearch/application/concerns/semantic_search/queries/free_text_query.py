import logging
import re
from typing import Optional

from injector import inject
from mediatr import GenericQuery, Mediator
from qdrant_client.http.models import Condition, FieldCondition, Filter, MatchAny, MatchValue, Range
from rapidfuzz import fuzz

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_data_include_model import (
    CocktailSearchDataIncludeModel,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailSearchModel
from cezzis_com_cocktails_aisearch.domain.config.qdrant_options import QdrantOptions
from cezzis_com_cocktails_aisearch.infrastructure.repositories.icocktail_vector_repository import (
    ICocktailVectorRepository,
)
from cezzis_com_cocktails_aisearch.infrastructure.services.ireranker_service import IRerankerService


class FreeTextQuery(GenericQuery[list[tuple[str, float]]]):
    def __init__(
        self,
        free_text: Optional[str] = "",
        skip: Optional[int] = 0,
        take: Optional[int] = 10,
        matches: Optional[list[str]] = [],
        match_exclusive: Optional[bool] = False,
        include: Optional[list[CocktailSearchDataIncludeModel]] = [],
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
        "not containing ",
        "not featuring ",
        "that exclude ",
        "no ",
        "excluding ",
        "exclude ",
    ]

    # Fuzzy matching threshold (0-100) for cocktail name matching
    _FUZZY_MATCH_THRESHOLD: int = 82

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

    # Base spirit keywords to Qdrant metadata values
    _BASE_SPIRIT_MAPPING: dict[str, str] = {
        "gin": "gin",
        "rum": "rum",
        "vodka": "vodka",
        "tequila": "tequila",
        "mezcal": "mezcal",
        "whiskey": "whiskey",
        "whisky": "whiskey",
        "bourbon": "bourbon",
        "rye": "rye",
        "scotch": "scotch",
        "brandy": "brandy",
        "cognac": "cognac",
        "pisco": "pisco",
        "absinthe": "absinthe",
        "cachaça": "cachaça",
        "cachaca": "cachaça",
    }

    # Flavor profile keywords
    _FLAVOR_PROFILE_KEYWORDS: list[str] = [
        "bitter",
        "sweet",
        "sour",
        "citrus",
        "fruity",
        "herbal",
        "spicy",
        "smoky",
        "floral",
        "savory",
        "tropical",
        "dry",
        "creamy",
        "nutty",
        "tart",
        "minty",
        "boozy",
    ]

    # Cocktail family keywords
    _COCKTAIL_FAMILY_KEYWORDS: list[str] = [
        "sour",
        "fizz",
        "tiki",
        "negroni",
        "martini",
        "highball",
        "julep",
        "smash",
        "flip",
        "punch",
        "spritz",
        "cobbler",
        "daisy",
        "mule",
        "toddy",
        "collins",
        "swizzle",
    ]

    # Technique keywords
    _TECHNIQUE_MAPPING: dict[str, str] = {
        "shaken": "shaken",
        "shake": "shaken",
        "stirred": "stirred",
        "stir": "stirred",
        "built": "built",
        "build": "built",
        "muddled": "muddled",
        "muddle": "muddled",
        "blended": "blended",
        "blend": "blended",
        "layered": "layered",
        "layer": "layered",
    }

    # Strength keywords
    _STRENGTH_KEYWORDS: dict[str, str] = {
        "light": "light",
        "mild": "light",
        "low abv": "light",
        "low-abv": "light",
        "sessionable": "light",
        "medium": "medium",
        "strong": "strong",
        "boozy": "strong",
        "stiff": "strong",
    }

    # Temperature keywords
    _TEMPERATURE_KEYWORDS: dict[str, str] = {
        "cold": "cold",
        "chilled": "cold",
        "frozen": "frozen",
        "blended": "frozen",
        "warm": "warm",
        "hot": "warm",
    }

    # Season keywords
    _SEASON_KEYWORDS: list[str] = [
        "summer",
        "winter",
        "spring",
        "fall",
        "autumn",
        "all-season",
    ]

    # Occasion keywords
    _OCCASION_KEYWORDS: list[str] = [
        "aperitif",
        "digestif",
        "party",
        "brunch",
        "dinner",
        "date night",
        "celebration",
        "nightcap",
        "after dinner",
    ]

    # Mood keywords
    _MOOD_KEYWORDS: list[str] = [
        "refreshing",
        "sophisticated",
        "fun",
        "relaxing",
        "cozy",
        "elegant",
        "festive",
        "adventurous",
        "romantic",
    ]

    @inject
    def __init__(
        self,
        cocktail_vector_repository: ICocktailVectorRepository,
        qdrant_opotions: QdrantOptions,
        reranker_service: IRerankerService,
    ):
        self.cocktail_vector_repository = cocktail_vector_repository
        self.qdrant_options = qdrant_opotions
        self.reranker_service = reranker_service
        self.logger = logging.getLogger("free_text_query_handler")

    async def handle(self, command: FreeTextQuery) -> list[CocktailSearchModel]:
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

        skip = command.skip or 0
        take = command.take or 10

        # Cross-encoder reranking: refine relevance ordering using TEI /rerank
        sorted_cocktails = await self.reranker_service.rerank(
            query=command.free_text,
            cocktails=sorted_cocktails,
            top_k=skip + take,
        )

        # Apply rating-based sort override for rating queries
        if any(
            self._fuzzy_keyword_in_text(search_text, term)
            for term in ["top rated", "best rated", "highest rated", "popular"]
        ):
            sorted_cocktails = sorted(sorted_cocktails, key=lambda c: c.rating, reverse=True)

        return sorted_cocktails[skip : skip + take]

    async def _handle_browse(self, command: FreeTextQuery) -> list[CocktailSearchModel]:
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
        self, search_text: str, all_cocktails: list[CocktailSearchModel]
    ) -> list[CocktailSearchModel] | None:
        """
        Check if the search text matches cocktail names.
        Returns cocktails where:
        1. Title exactly matches the search text (prioritized first)
        2. Title contains the search text as a substring
        3. Title fuzzy-matches the search text (handles typos like "Margerita", "Mohito")

        This ensures "Mai Tai" returns both "Mai Tai" and "Bitter Mai Tai".
        """
        search_lower = search_text.lower().strip()

        # Remove common suffixes for matching (singular and plural) with fuzzy tolerance
        search_normalized = search_lower
        _SUFFIX_WORDS = ["cocktails", "cocktail", "drinks", "drink", "recipes", "recipe"]
        words = search_normalized.split()
        if len(words) > 1 and any(self._fuzzy_word_match(words[-1], sw) for sw in _SUFFIX_WORDS):
            search_normalized = " ".join(words[:-1])

        # Only do name matching if the search looks like a cocktail name
        # (not a descriptive query like "cocktails with honey" or "gin cocktails")
        descriptive_prefixes = ["cocktails", "drinks", "recipes", "show me", "find"]
        text_words = search_lower.split()
        for prefix in descriptive_prefixes:
            prefix_word_count = len(prefix.split())
            if len(text_words) > prefix_word_count and self._fuzzy_startswith(search_lower, prefix):
                return None

        # Queries like "gin cocktails" or "vodka drinks" are descriptive, not name lookups
        # Uses exact matching to preserve the intentional singular/plural distinction:
        # "gin cocktails" (plural) → skip name matching; "champagne cocktail" (singular) → proceed
        if len(text_words) > 1 and text_words[-1] in {"cocktails", "drinks", "recipes"}:
            return None

        # Find exact matches (highest priority)
        exact_matches = [c for c in all_cocktails if c.title.lower() == search_normalized]

        # Find partial matches (title contains the search term)
        partial_matches = [c for c in all_cocktails if search_normalized in c.title.lower() and c not in exact_matches]

        # Combine: exact matches first, then partial matches sorted alphabetically
        if exact_matches or partial_matches:
            partial_matches_sorted = sorted(partial_matches, key=lambda c: c.title)
            return exact_matches + partial_matches_sorted

        # Fuzzy matching fallback for typo tolerance (e.g., "Margerita" → "Margarita")
        fuzzy_matches = self._find_fuzzy_name_match(search_normalized, all_cocktails)
        if fuzzy_matches:
            return fuzzy_matches

        return None

    def _find_fuzzy_name_match(
        self, search_text: str, all_cocktails: list[CocktailSearchModel]
    ) -> list[CocktailSearchModel] | None:
        """
        Find cocktails whose names fuzzy-match the search text using rapidfuzz.
        Handles common misspellings like "Margerita", "Mohito", "Daiquri".
        Returns matches sorted by fuzzy score descending, or None if no good matches.
        """
        scored_matches: list[tuple[CocktailSearchModel, float]] = []

        for cocktail in all_cocktails:
            title_lower = cocktail.title.lower()
            # Use ratio for overall similarity and partial_ratio for substring similarity
            ratio_score = fuzz.ratio(search_text, title_lower)
            partial_score = fuzz.partial_ratio(search_text, title_lower)
            best_score = max(ratio_score, partial_score)

            if best_score >= self._FUZZY_MATCH_THRESHOLD:
                scored_matches.append((cocktail, best_score))

        if not scored_matches:
            return None

        # Sort by score descending, then alphabetically for ties
        scored_matches.sort(key=lambda x: (-x[1], x[0].title))
        return [match[0] for match in scored_matches]

    def _handle_short_query(
        self, search_text: str, all_cocktails: list[CocktailSearchModel], command: FreeTextQuery
    ) -> list[CocktailSearchModel]:
        """Handle queries too short for meaningful semantic search."""
        filtered = [c for c in all_cocktails if self._matches_text_search(c, search_text)]
        sorted_cocktails = sorted(filtered, key=lambda p: p.title or "")

        skip = command.skip or 0
        take = command.take or 10
        return sorted_cocktails[skip : skip + take]

    def _matches_text_search(self, cocktail: CocktailSearchModel, search_text: str) -> bool:
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

    # --- Fuzzy matching helpers for misspelling tolerance ---

    def _fuzzy_word_match(self, word: str, target: str, threshold: int = 80) -> bool:
        """Check if two words fuzzy-match. Uses exact match for short targets (<5 chars)
        to avoid false positives with words like 'gin', 'rum', 'rye', 'no', 'dry'."""
        if len(target) < 5:
            return word == target
        return fuzz.ratio(word, target) >= threshold

    def _fuzzy_keyword_in_text(self, search_text: str, keyword: str, threshold: int = 80) -> bool:
        """Check if a keyword (single or multi-word) appears in search_text with fuzzy tolerance.
        Words shorter than 5 characters require exact matching to avoid false positives."""
        text_words = [w.strip(",.!?;:") for w in search_text.split()]
        keyword_words = keyword.strip().split()

        if len(keyword_words) == 1:
            return any(self._fuzzy_word_match(w, keyword_words[0], threshold) for w in text_words)

        # Multi-word: check consecutive word windows
        kw_len = len(keyword_words)
        for i in range(len(text_words) - kw_len + 1):
            if all(self._fuzzy_word_match(text_words[i + j], keyword_words[j], threshold) for j in range(kw_len)):
                return True
        return False

    def _fuzzy_endswith(self, search_text: str, suffix_words: list[str], threshold: int = 80) -> bool:
        """Check if the last word of search_text fuzzy-matches any of the given suffix words."""
        words = search_text.split()
        if not words:
            return False
        last_word = words[-1].strip(",.!?;:")
        return any(self._fuzzy_word_match(last_word, sw, threshold) for sw in suffix_words)

    def _fuzzy_startswith(self, search_text: str, prefix: str, threshold: int = 80) -> bool:
        """Check if search_text starts with a prefix phrase (fuzzy per-word matching)."""
        text_words = [w.strip(",.!?;:") for w in search_text.split()]
        prefix_words = prefix.strip().split()
        if len(text_words) < len(prefix_words):
            return False
        return all(self._fuzzy_word_match(text_words[j], prefix_words[j], threshold) for j in range(len(prefix_words)))

    def _build_query_filter(self, search_text: str) -> Filter | None:
        """
        Build Qdrant payload filter from structured query elements.

        Pushes filtering to the vector DB level for better performance and accuracy.
        Handles: IBA status, glassware, ingredient count, prep time, serves, and ingredient exclusion.
        """
        must_conditions: list[Condition] = []
        must_not_conditions: list[Condition] = []

        # IBA filter (check non-IBA first since "non-iba" contains "iba")
        if any(
            self._fuzzy_keyword_in_text(search_text, term)
            for term in ["non-iba", "non iba", "modern cocktail", "contemporary"]
        ):
            must_conditions.append(FieldCondition(key="metadata.is_iba", match=MatchValue(value=False)))
        elif any(
            self._fuzzy_keyword_in_text(search_text, term)
            for term in ["iba", "iba cocktail", "official cocktail", "classic iba"]
        ):
            must_conditions.append(FieldCondition(key="metadata.is_iba", match=MatchValue(value=True)))

        # Glassware filter
        for term, glassware_value in self._GLASSWARE_MAPPING.items():
            if self._fuzzy_keyword_in_text(search_text, term):
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
        elif any(
            self._fuzzy_keyword_in_text(search_text, term) for term in ["simple", "easy", "few ingredients", "basic"]
        ):
            must_conditions.append(FieldCondition(key="metadata.ingredient_count", range=Range(lte=4)))
        elif any(
            self._fuzzy_keyword_in_text(search_text, term) for term in ["complex", "many ingredients", "elaborate"]
        ):
            must_conditions.append(FieldCondition(key="metadata.ingredient_count", range=Range(gte=6)))

        # Prep time filters
        if any(self._fuzzy_keyword_in_text(search_text, term) for term in ["quick", "fast", "5 minute", "5-minute"]):
            must_conditions.append(FieldCondition(key="metadata.prep_time_minutes", range=Range(lte=5)))
        elif any(self._fuzzy_keyword_in_text(search_text, term) for term in ["10 minute", "10-minute"]):
            must_conditions.append(FieldCondition(key="metadata.prep_time_minutes", range=Range(lte=10)))

        # Serves filter
        serves_match = re.search(r"serves?\s*(\d+)", search_text)
        if serves_match:
            target_serves = int(serves_match.group(1))
            must_conditions.append(FieldCondition(key="metadata.serves", match=MatchValue(value=target_serves)))

        # Base spirit filter
        for term, spirit_value in self._BASE_SPIRIT_MAPPING.items():
            if self._fuzzy_keyword_in_text(search_text, term):
                must_conditions.append(
                    FieldCondition(key="metadata.keywords_base_spirit", match=MatchValue(value=spirit_value))
                )
                break  # Only match the first spirit to avoid over-filtering

        # Flavor profile filter
        matched_flavors = [
            flavor for flavor in self._FLAVOR_PROFILE_KEYWORDS if self._fuzzy_keyword_in_text(search_text, flavor)
        ]
        for flavor in matched_flavors[:2]:  # Limit to 2 flavor filters to avoid over-constraining
            must_conditions.append(
                FieldCondition(key="metadata.keywords_flavor_profile", match=MatchValue(value=flavor))
            )

        # Cocktail family filter
        for family in self._COCKTAIL_FAMILY_KEYWORDS:
            if self._fuzzy_keyword_in_text(search_text, family):
                must_conditions.append(
                    FieldCondition(key="metadata.keywords_cocktail_family", match=MatchValue(value=family))
                )
                break

        # Technique filter
        for term, technique_value in self._TECHNIQUE_MAPPING.items():
            if self._fuzzy_keyword_in_text(search_text, term):
                must_conditions.append(
                    FieldCondition(key="metadata.keywords_technique", match=MatchValue(value=technique_value))
                )
                break

        # Strength filter
        for term, strength_value in self._STRENGTH_KEYWORDS.items():
            if self._fuzzy_keyword_in_text(search_text, term):
                must_conditions.append(
                    FieldCondition(key="metadata.keywords_strength", match=MatchValue(value=strength_value))
                )
                break

        # Temperature filter
        for term, temp_value in self._TEMPERATURE_KEYWORDS.items():
            if self._fuzzy_keyword_in_text(search_text, term):
                must_conditions.append(
                    FieldCondition(key="metadata.keywords_temperature", match=MatchValue(value=temp_value))
                )
                break

        # Season filter
        matched_seasons = [
            season for season in self._SEASON_KEYWORDS if self._fuzzy_keyword_in_text(search_text, season)
        ]
        if matched_seasons:
            # Map "autumn" to "fall" for consistency
            normalized_seasons = ["fall" if s == "autumn" else s for s in matched_seasons]
            must_conditions.append(
                FieldCondition(
                    key="metadata.keywords_season",
                    match=MatchAny(any=normalized_seasons),
                )
            )

        # Occasion filter
        for occasion in self._OCCASION_KEYWORDS:
            if self._fuzzy_keyword_in_text(search_text, occasion):
                must_conditions.append(
                    FieldCondition(key="metadata.keywords_occasion", match=MatchValue(value=occasion))
                )
                break

        # Mood filter
        matched_moods = [mood for mood in self._MOOD_KEYWORDS if self._fuzzy_keyword_in_text(search_text, mood)]
        for mood in matched_moods[:2]:  # Limit to 2 mood filters
            must_conditions.append(FieldCondition(key="metadata.keywords_mood", match=MatchValue(value=mood)))

        # Ingredient exclusion (e.g., "without honey", "no rum", "without blue curacao")
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
        """
        Extract ingredient terms that should be excluded from results.
        Handles multi-word ingredients like "blue curacao", "orange juice", "lime juice".
        Each word of the multi-word phrase is added individually to match against
        the ingredient_words field which stores split words.
        Uses fuzzy matching for pattern detection to handle misspellings.
        """
        # Stop words that signal the end of an exclusion phrase
        _STOP_WORDS = {
            "and",
            "or",
            "but",
            "with",
            "in",
            "on",
            "the",
            "a",
            "an",
            "served",
            "cocktail",
            "cocktails",
            "drink",
            "drinks",
            "that",
            "which",
            "for",
            "from",
        }

        # Pre-compute first words of exclusion patterns for stop detection
        _EXCLUSION_FIRST_WORDS = [p.strip().split()[0] for p in self._EXCLUSION_PATTERNS]

        terms: list[str] = []
        text_words = search_text.split()

        for pattern in self._EXCLUSION_PATTERNS:
            pattern_words = pattern.strip().split()
            pat_len = len(pattern_words)

            i = 0
            while i <= len(text_words) - pat_len:
                if all(self._fuzzy_word_match(text_words[i + j], pattern_words[j]) for j in range(pat_len)):
                    # Pattern matched at word index i; extract words after it
                    after_start = i + pat_len
                    phrase_words: list[str] = []
                    for k in range(after_start, len(text_words)):
                        cleaned = text_words[k].strip(",.!?").lower()
                        if cleaned in _STOP_WORDS or len(cleaned) < 2:
                            break
                        # Check if this word starts another exclusion pattern
                        if any(self._fuzzy_word_match(cleaned, fw) for fw in _EXCLUSION_FIRST_WORDS):
                            break
                        phrase_words.append(cleaned)
                        # Limit to 3-word phrases to avoid runaway matching
                        if len(phrase_words) >= 3:
                            break

                    # Add each word individually for matching against ingredient_words
                    for word in phrase_words:
                        if word not in terms:
                            terms.append(word)

                    i = after_start + max(len(phrase_words), 1)
                else:
                    i += 1

        return terms
