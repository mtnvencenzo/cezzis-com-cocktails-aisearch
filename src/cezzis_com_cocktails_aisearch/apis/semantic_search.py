from fastapi import APIRouter
from injector import inject
from mediatr import Mediator

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktail_model import CocktailModel
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktails_rs import CocktailsRs
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.queries.free_text_query import FreeTextQuery


class SemanticSearchRouter(APIRouter):
    @inject
    def __init__(self, mediator: Mediator):
        super().__init__()
        self.mediator = mediator
        self.add_api_route("/v1/cocktails/search", self.search, methods=["GET"])

    async def search(self) -> CocktailsRs:
        """
        Performs a semantic search for cocktails based on a free text query.
        """
        query = FreeTextQuery("Show me cocktails with honey and lemon")
        items = await self.mediator.send_async(query)

        cocktails = [CocktailModel(
            id=item[0],
            title="test",
            descriptiveTitle="test",
            rating=0,
            ingredients=[],
            isIba=False,
            serves=0,
            prepTimeMinutes=0,
            searchTiles=[],
            glassware=[]
        ) for item in items]

        return CocktailsRs(items=cocktails)