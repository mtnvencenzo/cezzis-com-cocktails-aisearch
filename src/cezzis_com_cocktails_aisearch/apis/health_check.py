from typing import cast

from fastapi import APIRouter
from injector import inject
from mediatr import Mediator

from cezzis_com_cocktails_aisearch.application.concerns.health.models.health_check_rs import HealthCheckRs
from cezzis_com_cocktails_aisearch.application.concerns.health.queries.health_check_query import HealthCheckQuery


class HealthCheckRouter(APIRouter):
    @inject
    def __init__(self, mediator: Mediator):
        super().__init__()
        self.mediator = mediator
        self.add_api_route(
            path="/v1/health",
            endpoint=self.health_check,
            methods=["GET"],
            responses={
                200: {"model": HealthCheckRs, "description": "Successful health check"},
            },
        )

    async def health_check(self) -> HealthCheckRs:
        """
        Performs a health check of the API.
        """

        return cast(HealthCheckRs, await self.mediator.send_async(HealthCheckQuery()))
