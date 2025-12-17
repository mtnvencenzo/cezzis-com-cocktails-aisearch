from fastapi import FastAPI
from injector import Injector

from cezzis_com_cocktails_aisearch.app_module import AppModule
from cezzis_com_cocktails_aisearch.apis.semantic_search import SemanticSearchRouter
from cezzis_com_cocktails_aisearch.apis.conversational_search import ConverstionalSearchRouter
from cezzis_com_cocktails_aisearch.application.behaviors import initialize_opentelemetry


def create_injector() -> Injector:
    return Injector([AppModule()])

initialize_opentelemetry()
injector = create_injector()

app = FastAPI()
app.include_router(injector.get(SemanticSearchRouter))
app.include_router(injector.get(ConverstionalSearchRouter))
