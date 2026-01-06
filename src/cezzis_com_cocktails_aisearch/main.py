import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from cezzis_com_cocktails_aisearch.apis.embedding import EmbeddingRouter
from cezzis_com_cocktails_aisearch.apis.scalar_docs import ScalarDocsRouter
from cezzis_com_cocktails_aisearch.apis.semantic_search import SemanticSearchRouter
from cezzis_com_cocktails_aisearch.app_module import create_injector
from cezzis_com_cocktails_aisearch.application.behaviors import global_exception_handler, initialize_opentelemetry
from cezzis_com_cocktails_aisearch.domain.config.app_options import AppOptions

sys.excepthook = global_exception_handler

initialize_opentelemetry()
injector = create_injector()
app_options = injector.get(AppOptions)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_options.allowed_origins.replace(" ", "").split(",") if app_options.allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(injector.get(SemanticSearchRouter))
app.include_router(injector.get(ScalarDocsRouter))
app.include_router(injector.get(EmbeddingRouter))
