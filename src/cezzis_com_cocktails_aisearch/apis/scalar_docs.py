from fastapi import APIRouter
from injector import inject
from mediatr import Mediator
from scalar_fastapi import get_scalar_api_reference


class ScalarDocsRouter(APIRouter):
    @inject
    def __init__(self, mediator: Mediator):
        super().__init__()
        self.mediator = mediator
        self.add_api_route(path="/scalar/v1", endpoint=self.scalar_html, methods=["GET"], include_in_schema=False)

    async def scalar_html(self):
        return get_scalar_api_reference(
            # Your OpenAPI document
            openapi_url="/openapi.json"
        )
