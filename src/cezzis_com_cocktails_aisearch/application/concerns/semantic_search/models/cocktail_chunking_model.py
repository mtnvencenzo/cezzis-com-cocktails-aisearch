import dataclasses
from dataclasses import dataclass
from typing import List
from uuid import NAMESPACE_DNS, uuid5

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_description_chunk import CocktailDescriptionChunk
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailModel


@dataclass
class CocktailChunkingModel:
    cocktail_model: CocktailModel
    chunks: List[CocktailDescriptionChunk]

    def as_serializable_json(self) -> bytes:
        from pydantic import TypeAdapter

        serializable_dict = {
            "cocktail_model": self.cocktail_model.model_dump(),
            "chunks": [dataclasses.asdict(chunk) for chunk in self.chunks],
        }
        return TypeAdapter(dict).dump_json(serializable_dict)

    def get_chunk_uuids(self) -> List[str]:
        return [chunk.to_uuid() for chunk in self.chunks]
