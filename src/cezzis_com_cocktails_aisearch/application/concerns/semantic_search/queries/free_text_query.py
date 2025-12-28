from injector import inject
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from mediatr import GenericQuery, Mediator
from qdrant_client import QdrantClient

from cezzis_com_cocktails_aisearch.domain.config.hugging_face_options import HuggingFaceOptions
from cezzis_com_cocktails_aisearch.domain.config.qdrant_options import QdrantOptions


class FreeTextQuery(GenericQuery[list[tuple[str, float]]]):
    def __init__(self, text: str):
        self.text = text


@Mediator.behavior
class FreeTextQueryValidator:
    def handle(self, command: FreeTextQuery, next) -> None:
        if not command.text:
            raise ValueError("Invalid text provided for free text query")

        return next()


@Mediator.handler
class FreeTextQueryHandler:
    @inject
    def __init__(
        self, hugging_face_options: HuggingFaceOptions, qdrant_client: QdrantClient, qdrant_options: QdrantOptions
    ):
        self.hugging_face_options = hugging_face_options
        self.qdrant_client = qdrant_client
        self.qdrant_options = qdrant_options

    async def handle(self, command: FreeTextQuery) -> list[tuple[str, float]]:
        hf_endpoint = HuggingFaceEndpointEmbeddings(
            model=self.hugging_face_options.inference_model,  # http://localhost:8989 | sentence-transformers/all-mpnet-base-v2
            huggingfacehub_api_token=self.hugging_face_options.api_token,
            task="feature-extraction",
        )

        query_vector = await hf_endpoint.aembed_query(command.text)

        if len(query_vector) == 0:
            raise ValueError("Failed to generate embeddings for the provided text")

        search_results = self.qdrant_client.query_points(
            collection_name=self.qdrant_options.collection_name,  # Replace with your collection name
            query=query_vector,
            limit=30,
            with_payload=True,
        )

        # Sort points by score descending
        sorted_points = sorted(search_results.points, key=lambda p: getattr(p, "score", 0), reverse=True)

        cocktailIds: list[tuple[str, float]] = []
        seen_ids = set()
        for point in sorted_points:
            payload = point.payload if hasattr(point, "payload") else None
            if payload:
                metadata = payload.get("metadata")
                if metadata:
                    id = metadata.get("cocktail_id")
                    if id and id not in seen_ids:
                        cocktailIds.append((id, point.score))
                        seen_ids.add(id)

        return cocktailIds
