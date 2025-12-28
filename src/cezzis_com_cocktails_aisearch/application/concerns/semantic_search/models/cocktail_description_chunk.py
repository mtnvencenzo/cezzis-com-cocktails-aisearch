from dataclasses import dataclass
from uuid import NAMESPACE_DNS, uuid5


@dataclass
class CocktailDescriptionChunk:
    category: str
    content: str

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def to_uuid(self) -> str:
        return str(uuid5(NAMESPACE_DNS, f"{self.category}-{self.content}"))
