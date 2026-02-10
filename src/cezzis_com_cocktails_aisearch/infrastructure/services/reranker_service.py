import logging

import httpx
from injector import inject

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailSearchModel
from cezzis_com_cocktails_aisearch.domain.config.reranker_options import RerankerOptions
from cezzis_com_cocktails_aisearch.infrastructure.services.ireranker_service import IRerankerService


class RerankerService(IRerankerService):
    """Cross-encoder reranker using TEI (Text Embeddings Inference) /rerank endpoint.

    Sends candidate cocktails and the original query to a TEI instance running
    a cross-encoder model. The cross-encoder jointly encodes query-document pairs
    to produce more accurate relevance scores than bi-encoder cosine similarity alone.
    """

    @inject
    def __init__(self, reranker_options: RerankerOptions):
        self.options = reranker_options
        self.logger = logging.getLogger("reranker_service")

    async def rerank(
        self,
        query: str,
        cocktails: list[CocktailSearchModel],
        top_k: int = 10,
    ) -> list[CocktailSearchModel]:
        """Rerank cocktails using the TEI cross-encoder /rerank endpoint.

        If the reranker is disabled, unavailable, or fails, returns the original
        list unchanged (graceful degradation).
        """
        if not cocktails:
            return cocktails

        # Build text representations for each cocktail
        texts = [self._build_document_text(c) for c in cocktails]

        try:
            scores = await self._call_tei_rerank(query, texts)
        except Exception:
            self.logger.warning("Reranker call failed, returning original order", exc_info=True)
            return cocktails

        if not scores or len(scores) != len(cocktails):
            self.logger.warning(
                "Reranker returned unexpected results count",
                extra={"expected": len(cocktails), "got": len(scores) if scores else 0},
            )
            return cocktails

        # Apply reranker scores and filter by threshold
        scored_cocktails: list[tuple[CocktailSearchModel, float]] = []
        for idx, cocktail in enumerate(cocktails):
            reranker_score = scores[idx]
            if cocktail.search_statistics:
                cocktail.search_statistics.reranker_score = reranker_score

            if reranker_score >= self.options.score_threshold:
                scored_cocktails.append((cocktail, reranker_score))

        # Sort by reranker score descending
        scored_cocktails.sort(key=lambda x: x[1], reverse=True)

        # Apply top_k limit
        result = [c for c, _ in scored_cocktails[:top_k]]

        self.logger.info(
            "Reranking complete",
            extra={
                "candidates": len(cocktails),
                "after_threshold": len(scored_cocktails),
                "returned": len(result),
            },
        )

        return result

    async def _call_tei_rerank(self, query: str, texts: list[str]) -> list[float]:
        """Call the TEI /rerank endpoint and return scores in original index order."""
        endpoint = self.options.endpoint.rstrip("/")
        url = f"{endpoint}/rerank"

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.options.api_key:
            headers["Authorization"] = f"Bearer {self.options.api_key}"

        payload = {
            "query": query,
            "texts": texts,
            "truncate": True,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()

        results = response.json()

        # TEI returns [{"index": 0, "score": 0.95}, ...] sorted by score desc
        # We need to map scores back to original index order
        scores = [0.0] * len(texts)
        for item in results:
            idx = item["index"]
            score = item["score"]
            scores[idx] = score

        return scores

    @staticmethod
    def _build_document_text(cocktail: CocktailSearchModel) -> str:
        """Build a text representation of a cocktail for cross-encoder scoring.

        Combines title, descriptive title, and ingredients into a single string
        that gives the cross-encoder enough context to assess relevance.
        """
        parts = [cocktail.title]

        if cocktail.descriptive_title and cocktail.descriptive_title != cocktail.title:
            parts.append(cocktail.descriptive_title)

        if cocktail.ingredients:
            ingredient_names = [i.name for i in cocktail.ingredients if i.name]
            if ingredient_names:
                parts.append("Ingredients: " + ", ".join(ingredient_names))

        return ". ".join(parts)
