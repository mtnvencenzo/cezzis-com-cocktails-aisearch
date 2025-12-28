import sys

from fastapi import FastAPI

from cezzis_com_cocktails_aisearch.apis.conversational_search import ConverstionalSearchRouter
from cezzis_com_cocktails_aisearch.apis.scalar_docs import ScalarDocsRouter
from cezzis_com_cocktails_aisearch.apis.semantic_search import SemanticSearchRouter
from cezzis_com_cocktails_aisearch.app_module import create_injector
from cezzis_com_cocktails_aisearch.application.behaviors import global_exception_handler, initialize_opentelemetry

sys.excepthook = global_exception_handler

initialize_opentelemetry()
injector = create_injector()

app = FastAPI()
app.include_router(injector.get(SemanticSearchRouter))
app.include_router(injector.get(ConverstionalSearchRouter))
app.include_router(injector.get(ScalarDocsRouter))
