from uuid import NAMESPACE_DNS, uuid5

from pydantic import BaseModel, Field


class CocktailDescriptionChunk(BaseModel):
    category: str = Field(..., description="Category of the description chunk")
    content: str = Field(..., description="Textual content of the description chunk")

    def to_uuid(self) -> str:
        return str(uuid5(NAMESPACE_DNS, f"{self.category}-{self.content}"))
