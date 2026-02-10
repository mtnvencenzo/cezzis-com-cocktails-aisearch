from abc import ABC, abstractmethod


class ISpladeService(ABC):
    @abstractmethod
    async def encode(self, text: str) -> tuple[list[int], list[float]]:
        """Encode text into a sparse vector using the SPLADE model via TEI.

        Args:
            text: The text to encode into a sparse vector.

        Returns:
            A tuple of (indices, values) representing the sparse vector.
            Returns empty lists if SPLADE is disabled or encoding fails.
        """
        pass

    @abstractmethod
    async def encode_batch(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        """Encode multiple texts into sparse vectors using the SPLADE model via TEI.

        Args:
            texts: The texts to encode into sparse vectors.

        Returns:
            A list of (indices, values) tuples, one per input text.
            Returns empty tuples if encoding fails.
        """
        pass
