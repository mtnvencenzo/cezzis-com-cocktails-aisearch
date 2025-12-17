from mediatr import Mediator, GenericQuery

class FreeTextQuery(GenericQuery[str]):
    def __init__(self, text: str):
        self.text = text

@Mediator.handler
class FreeTextQueryHandler:
    def handle(self, request: FreeTextQuery) -> str:
        return f"Processed query: {request.text}"