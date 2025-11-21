
# Cezzis.com Cocktails AI Search API

This repository provides the backend API for cocktail search and conversational capabilities as part of the broader Cezzis.com Retrieval-Augmented Generation (RAG) and agentic workflow ecosystem.

This API is designed to work in conjunction with the [Cezzis.com Ingestion Agentic Workflow](https://github.com/mtnvencenzo/cezzis-com-ingestion-agentic-wf) repository, which manages the ingestion and agentic orchestration for cocktail data and related conversational flows.

## Overview

**Status:** Work In Progress (WIP) — This project is in its early stages and actively under development.

The API enables users to search for cocktails using semantic queries against a vector database, leveraging modern information retrieval techniques. In future releases, it will support contextual chat and conversation about cocktails powered by Large Language Models (LLMs), enabling advanced agentic flows and interactive experiences.

### Key Features
- FastAPI-based RESTful API
- Semantic cocktail search using vector embeddings
- Modular architecture for future conversational AI capabilities
- Designed for integration with Cezzis.com RAG and agentic systems

### Roadmap
- [x] Initial API and vector search endpoints
- [ ] Contextual chat and conversation via LLMs
- [ ] Enhanced cocktail metadata and user interaction

## Getting Started

This project uses [FastAPI](https://fastapi.tiangolo.com/) and is managed with Poetry. To install dependencies:

```bash
poetry install
```

To run the development server:

```bash
poetry run uvicorn src.cezzis_com_cocktails_aisearch.main:app --reload
```

## Contributing

Contributions are welcome! Please open issues or pull requests to help improve the API and its capabilities.

## License

This project is licensed under the MIT License.