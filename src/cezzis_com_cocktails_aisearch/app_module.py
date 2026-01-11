from injector import Binder, Injector, Module, singleton
from mediatr import Mediator
from qdrant_client import QdrantClient

from cezzis_com_cocktails_aisearch.application.concerns.health.queries.health_check_query import HealthCheckQueryHandler
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.commands.cocktail_embedding_command import (
    CocktailEmbeddingCommandHandler,
)
from cezzis_com_cocktails_aisearch.application.concerns.semantic_search.queries import FreeTextQueryHandler
from cezzis_com_cocktails_aisearch.domain.config import QdrantOptions, get_qdrant_options
from cezzis_com_cocktails_aisearch.domain.config.app_options import AppOptions, get_app_options
from cezzis_com_cocktails_aisearch.domain.config.hugging_face_options import HuggingFaceOptions, get_huggingface_options
from cezzis_com_cocktails_aisearch.infrastructure.repositories import (
    CocktailVectorRepository,
    ICocktailVectorRepository,
)


def create_injector() -> Injector:
    return Injector([AppModule()])


def mediator_manager(handler_class, is_behavior=False):
    return injector.get(handler_class)


class AppModule(Module):
    def configure(self, binder: Binder):
        qdrant_options = get_qdrant_options()
        qdrant_client = QdrantClient(
            url=qdrant_options.host,  # http://localhost:6333 | https://aca-vec-eus-glo-qdrant-001.proudfield-08e1f932.eastus.azurecontainerapps.io
            api_key=qdrant_options.api_key if qdrant_options.api_key else None,
            port=qdrant_options.port,
            https=qdrant_options.use_https,
            prefer_grpc=False,
            timeout=60,
        )

        binder.bind(Mediator, Mediator(handler_class_manager=mediator_manager), scope=singleton)
        binder.bind(ICocktailVectorRepository, CocktailVectorRepository, scope=singleton)
        binder.bind(AppOptions, get_app_options(), scope=singleton)
        binder.bind(HuggingFaceOptions, get_huggingface_options(), scope=singleton)
        binder.bind(QdrantOptions, get_qdrant_options(), scope=singleton)
        binder.bind(QdrantClient, qdrant_client, scope=singleton)
        binder.bind(FreeTextQueryHandler, FreeTextQueryHandler, scope=singleton)
        binder.bind(CocktailEmbeddingCommandHandler, CocktailEmbeddingCommandHandler, scope=singleton)
        binder.bind(HealthCheckQueryHandler, HealthCheckQueryHandler, scope=singleton)


injector = create_injector()
