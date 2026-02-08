import logging
from typing import Optional

from injector import inject
from mediatr import GenericQuery, Mediator

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailModel
from cezzis_com_cocktails_aisearch.domain.config.qdrant_options import QdrantOptions
from cezzis_com_cocktails_aisearch.infrastructure.repositories.icocktail_vector_repository import (
    ICocktailVectorRepository,
)


class TypeAheadQuery(GenericQuery[list[tuple[str, float]]]):
    def __init__(
        self,
        free_text: Optional[str] = "",
        skip: Optional[int] = 0,
        take: Optional[int] = 10,
        filters: Optional[list[str]] = [],
    ):
        self.free_text = free_text
        self.skip = skip
        self.take = take
        self.filters = filters


@Mediator.behavior
class TypeAheadQueryValidator:
    def handle(self, command: TypeAheadQuery, next) -> None:
        return next()


@Mediator.handler
class TypeAheadQueryHandler:
    @inject
    def __init__(
        self,
        cocktail_vector_repository: ICocktailVectorRepository,
        qdrant_opotions: QdrantOptions,
    ):
        self.cocktail_vector_repository = cocktail_vector_repository
        self.qdrant_options = qdrant_opotions
        self.logger = logging.getLogger("type_ahead_query_handler")

    async def handle(self, command: TypeAheadQuery) -> list[CocktailModel]:
        cocktails = await self.cocktail_vector_repository.get_all_cocktails()

        sorted_cocktails = sorted(
            cocktails,
            key=lambda p: p.title or "",
            reverse=False,
        )

        filtered_start_cocktails = [
            c
            for c in sorted_cocktails
            if not command.free_text
            or c.title.lower().startswith(command.free_text.lower())
            or c.descriptive_title.lower().startswith(command.free_text.lower())
        ]

        filtered_cocktails = filtered_start_cocktails

        if len(filtered_start_cocktails) < (command.take or 10):
            filtered_contains = [
                c
                for c in sorted_cocktails
                if command.free_text
                and not filtered_start_cocktails.__contains__(c)
                and command.free_text.lower()
                in (c.title.lower() if c.title else "" or c.descriptive_title.lower() if c.descriptive_title else "")
                and c not in filtered_start_cocktails
            ]

            filtered_cocktails = filtered_start_cocktails + filtered_contains

        skip = command.skip or 0
        take = command.take or 10
        return filtered_cocktails[skip : skip + take]
