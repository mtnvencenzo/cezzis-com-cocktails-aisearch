from fastapi import FastAPI

from .routers.semantic_search import router as semantic_search_router

app = FastAPI()

app.include_router(semantic_search_router)
