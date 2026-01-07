from typing import cast

from fastapi import APIRouter, Body, Request, Response
from injector import inject
from mediatr import Mediator

from cezzis_com_cocktails_aisearch.application.behaviors.apim_host_key_authorization.apim_host_key_authorization import (
    apim_host_key_authorization,
)
from cezzis_com_cocktails_aisearch.application.behaviors.oauth_authorization.oauth_authorization import (
    oauth_authorization,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.commands.cocktail_embedding_command import (
    CocktailEmbeddingCommand,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.models.cocktails_embedding_rq import (
    CocktailEmbeddingRq,
)


class EmbeddingRouter(APIRouter):
    @inject
    def __init__(self, mediator: Mediator):
        super().__init__()
        self.mediator = mediator
        self.add_api_route(
            path="/v1/cocktails/embeddings",
            endpoint=self.embed,
            methods=["PUT"],
            status_code=204,
            responses={
                204: {"description": "Embedding successful. No content returned."},
                422: {"description": "Invalid request or embedding failed."},
            },
        )

    @apim_host_key_authorization
    @oauth_authorization(scopes=["write:embeddings"])
    async def embed(
        self, _rq: Request, body: CocktailEmbeddingRq = Body(..., description="The cocktail embedding request")
    ) -> Response:
        """
        Performs a semantic search for cocktails based on a free text query.
        """
        command = CocktailEmbeddingCommand(chunks=body.content_chunks, cocktail_model=body.cocktail_model)

        embedding_result = cast(bool, await self.mediator.send_async(command))  # casting due to type hinting issues

        if embedding_result:
            return Response(status_code=204)

        raise Exception("Failed to embed cocktail description chunks")
