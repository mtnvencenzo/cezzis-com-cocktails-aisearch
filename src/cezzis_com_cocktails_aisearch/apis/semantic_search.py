from typing import cast

from fastapi import APIRouter, Header, Query, Request
from injector import inject
from mediatr import Mediator

from cezzis_com_cocktails_aisearch.application.behaviors.apim_host_key_authorization.apim_host_key_authorization import (
    apim_host_key_authorization,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_data_include_model import (
    CocktailDataIncludeModel,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailModel
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktails_search_rs import (
    CocktailsSearchRs,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.queries.free_text_query import FreeTextQuery
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.queries.type_ahead_query import TypeAheadQuery


class SemanticSearchRouter(APIRouter):
    @inject
    def __init__(self, mediator: Mediator):
        super().__init__(prefix="/v1/cocktails", tags=["Semantic Search"])
        self.mediator = mediator
        self.add_api_route(
            path="/search",
            endpoint=self.search,
            methods=["GET"],
            operation_id="getV1CocktailsSearch",
            responses={
                200: {"model": CocktailsSearchRs, "description": "Successful search results"},
            },
        )

        self.add_api_route(
            path="/typeahead",
            endpoint=self.typeahead,
            methods=["GET"],
            operation_id="getV1CocktailsTypeahead",
            responses={
                200: {"model": CocktailsSearchRs, "description": "Successful search results"},
            },
        )

    @apim_host_key_authorization
    async def search(
        self,
        _rq: Request,
        x_key: str = Header(..., alias="X-Key", description="The API gateway subscription key"),
        freetext: str | None = Query(None, description="The free text search term to match against"),
        skip: int | None = Query(0, description="The number of cocktail recipes to skip from the paged response"),
        take: int | None = Query(10, description="The number of cocktail recipes to take for pagination"),
        m: list[str] | None = Query(None, description="A list of cocktails that can be included in the list"),
        m_ex: bool | None = Query(
            False, description="Whether or not the supplied matches must be exclusively returned"
        ),
        inc: list[CocktailDataIncludeModel] | None = Query(
            None, description="The list of extension objects to include for each cocktail recipe"
        ),
        fi: list[str] | None = Query(
            None, description="An optional list of filters to use when quering the cocktail recipes"
        ),
    ) -> CocktailsSearchRs:
        """
        Performs a semantic search for cocktails based on a free text query.
        """
        query = FreeTextQuery(
            free_text=freetext or "",
            skip=skip or 0,
            take=take or 10,
            matches=m or [],
            match_exclusive=m_ex or False,
            include=inc or [],
            filters=fi or [],
        )

        items = cast(list[CocktailModel], await self.mediator.send_async(query))  # casting due to type hinting issues

        return CocktailsSearchRs(items=items)

    @apim_host_key_authorization
    async def typeahead(
        self,
        _rq: Request,
        x_key: str = Header(..., alias="X-Key", description="The API gateway subscription key"),
        freetext: str | None = Query(None, description="The free text search term to match against"),
        skip: int | None = Query(0, description="The number of cocktail recipes to skip from the paged response"),
        take: int | None = Query(10, description="The number of cocktail recipes to take for pagination"),
        fi: list[str] | None = Query(
            None, description="An optional list of filters to use when quering the cocktail recipes"
        ),
    ) -> CocktailsSearchRs:
        """
        Performs a typeahead search for cocktails based on a free text query.
        """
        query = TypeAheadQuery(
            free_text=freetext or "",
            skip=skip or 0,
            take=take or 10,
            filters=fi or [],
        )

        items = cast(list[CocktailModel], await self.mediator.send_async(query))  # casting due to type hinting issues

        return CocktailsSearchRs(items=items)
