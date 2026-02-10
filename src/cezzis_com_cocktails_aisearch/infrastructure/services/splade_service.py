import logging

import httpx
from injector import inject

from cezzis_com_cocktails_aisearch.domain.config.splade_options import SpladeOptions
from cezzis_com_cocktails_aisearch.infrastructure.services.isplade_service import ISpladeService


class SpladeService(ISpladeService):
    """SPLADE sparse encoder using TEI (Text Embeddings Inference) /embed_sparse endpoint.

    Sends text to a TEI instance running a SPLADE model to produce sparse vector
    representations for hybrid search. The sparse vectors capture lexical (keyword)
    signals that complement dense semantic embeddings.
    """

    @inject
    def __init__(self, splade_options: SpladeOptions):
        self.options = splade_options
        self.logger = logging.getLogger("splade_service")

    async def encode(self, text: str) -> tuple[list[int], list[float]]:
        """Encode text into a sparse vector using the TEI /embed_sparse endpoint.

        If the call fails, returns empty lists (graceful degradation).
        """
        try:
            result = await self._call_tei_embed_sparse([text])
            if result:
                return result[0]
            return ([], [])
        except Exception:
            self.logger.warning("SPLADE encode failed, returning empty sparse vector", exc_info=True)
            return ([], [])

    async def encode_batch(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        """Encode multiple texts into sparse vectors using the TEI /embed_sparse endpoint.

        If the call fails, returns empty tuples (graceful degradation).
        """
        if not texts:
            return []

        try:
            return await self._call_tei_embed_sparse(texts)
        except Exception:
            self.logger.warning("SPLADE encode_batch failed, returning empty sparse vectors", exc_info=True)
            return [([], [])] * len(texts)

    async def _call_tei_embed_sparse(self, inputs: list[str]) -> list[tuple[list[int], list[float]]]:
        """Call the TEI /embed_sparse endpoint and return sparse vectors.

        TEI returns a list of sparse embeddings, each being a list of
        {index: int, value: float} objects.
        """
        endpoint = self.options.endpoint.rstrip("/")
        url = f"{endpoint}/embed_sparse"

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.options.api_key:
            headers["Authorization"] = f"Bearer {self.options.api_key}"

        payload = {
            "inputs": inputs,
            "truncate": True,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()

        results = response.json()

        # TEI returns [[{"index": 42, "value": 0.8}, ...], ...]
        sparse_vectors: list[tuple[list[int], list[float]]] = []
        for embedding in results:
            indices: list[int] = []
            values: list[float] = []
            for token in embedding:
                indices.append(token["index"])
                values.append(token["value"])
            sparse_vectors.append((indices, values))

        return sparse_vectors
