from typing import cast

from fastapi import APIRouter, Response, status
from injector import inject
from mediatr import Mediator

from cezzis_com_cocktails_aisearch.application.concerns.health.models.health_check_rs import HealthCheckRs
from cezzis_com_cocktails_aisearch.application.concerns.health.queries.health_check_query import HealthCheckQuery
from cezzis_com_cocktails_aisearch.application.concerns.health.queries.readiness_check_query import ReadinessCheckQuery


class HealthCheckRouter(APIRouter):
    @inject
    def __init__(self, mediator: Mediator):
        super().__init__()
        self.mediator = mediator
        self.add_api_route(
            path="/v1/liveness",
            endpoint=self.liveness_check,
            methods=["GET"],
            responses={
                200: {"model": HealthCheckRs, "description": "Successful liveness check"},
            },
        )
        self.add_api_route(
            path="/v1/readiness",
            endpoint=self.readiness_check,
            methods=["GET"],
            responses={
                200: {"model": HealthCheckRs, "description": "Service is ready"},
                503: {"model": HealthCheckRs, "description": "Service is not ready"},
            },
        )

    async def liveness_check(self) -> HealthCheckRs:
        """
        Performs a liveness check of the API.
        """

        return cast(HealthCheckRs, await self.mediator.send_async(HealthCheckQuery()))

    async def readiness_check(self, response: Response) -> HealthCheckRs:
        """
        Performs a readiness check verifying connectivity to Qdrant.
        """

        result = cast(HealthCheckRs, await self.mediator.send_async(ReadinessCheckQuery()))
        if result.status != "healthy":
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return result
