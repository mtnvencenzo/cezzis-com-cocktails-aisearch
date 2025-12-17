from fastapi import APIRouter
from injector import inject
from mediatr import Mediator

from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.queries.free_text_query import FreeTextQuery


class ConverstionalSearchRouter(APIRouter):
    @inject
    def __init__(self, mediator: Mediator):
        super().__init__()
        self.mediator = mediator
        self.add_api_route("/v1/cocktails/conversation", self.search, methods=["GET"])

    async def search(self):
        query = FreeTextQuery("Show me cocktails with honey and lemon")
        result = await self.mediator.send_async(query)

        return {"message": result}
