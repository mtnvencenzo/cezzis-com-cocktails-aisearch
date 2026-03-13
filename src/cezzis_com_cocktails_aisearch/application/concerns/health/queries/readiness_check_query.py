import logging
from importlib.metadata import version

from injector import inject
from mediatr import GenericQuery, Mediator
from qdrant_client import QdrantClient

from cezzis_com_cocktails_aisearch.application.concerns.health.models.health_check_rs import HealthCheckRs


class ReadinessCheckQuery(GenericQuery[HealthCheckRs]):
    def __init__(self):
        pass


@Mediator.handler
class ReadinessCheckQueryHandler:
    @inject
    def __init__(self, qdrant_client: QdrantClient):
        self.logger = logging.getLogger("readiness_check_query_handler")
        self._qdrant_client = qdrant_client

    async def handle(self, command: ReadinessCheckQuery) -> HealthCheckRs:
        details = {}
        overall_healthy = True

        # Check Qdrant connectivity
        try:
            self._qdrant_client.get_collections()
            details["qdrant"] = "healthy"
        except Exception as exc:
            self.logger.warning("Qdrant health check failed: %s", exc)
            details["qdrant"] = "unhealthy"
            overall_healthy = False

        return HealthCheckRs(
            status="healthy" if overall_healthy else "unhealthy",
            version=version("cezzis_com_cocktails_aisearch"),
            output="All dependencies are reachable" if overall_healthy else "One or more dependencies are unreachable",
            details=details,
        )
