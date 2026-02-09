
# Cezzis.com Cocktails AI Search API

The backend API powering semantic cocktail search for [Cezzis.com](https://www.cezzis.com). Built with FastAPI, Qdrant vector database, and HuggingFace embeddings, this service enables natural-language cocktail discovery through vector similarity search, typeahead suggestions, and filtered browsing.

This API works in conjunction with the [Cezzis.com Ingestion Agentic Workflow](https://github.com/mtnvencenzo/cezzis-com-ingestion-agentic-wf), which handles cocktail data ingestion, content chunking, and embedding orchestration.

---

## Architecture

```
┌─────────────────┐      ┌──────────────────────┐      ┌─────────────────────┐
│   Client / MCP  │─────▶│  FastAPI Application  │─────▶│  Qdrant Vector DB   │
│                 │◀─────│  (Uvicorn @ :8010)    │◀─────│  (Cloud or Local)   │
└─────────────────┘      └──────────┬───────────┘      └─────────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  HuggingFace         │
                         │  Inference API       │
                         │  (Embeddings)        │
                         └──────────────────────┘
```

The application follows a **CQRS pattern** using [MediatR](https://pypi.org/project/mediatr/) for command/query separation, with **dependency injection** via [Injector](https://pypi.org/project/injector/) and **Pydantic** models for request/response validation.

### Project Structure

```
src/cezzis_com_cocktails_aisearch/
├── main.py                     # FastAPI app entrypoint
├── app_module.py               # DI container configuration
├── apis/                       # Route handlers
│   ├── semantic_search.py      # Search & typeahead endpoints
│   ├── embedding.py            # Embedding ingestion endpoint
│   ├── health_check.py         # Health check endpoint
│   └── scalar_docs.py          # Scalar API docs UI
├── application/
│   ├── behaviors/              # Cross-cutting concerns
│   │   ├── apim_host_key_authorization/  # API key auth
│   │   ├── error_handling/     # RFC 7807 Problem Details
│   │   ├── openapi/            # OpenAPI schema customization
│   │   └── otel/               # OpenTelemetry instrumentation
│   └── concerns/
│       ├── semantic_search/    # Search queries, commands & models
│       └── health/             # Health check query
├── domain/config/              # Configuration (env-based settings)
└── infrastructure/repositories/  # Qdrant vector store access
```

---

## Search Algorithm

The search pipeline uses a multi-strategy approach to return the most relevant cocktail results:

### 1. Exact Name Matching

When the query matches a cocktail name exactly (or closely via normalized comparison), that cocktail is returned immediately — bypassing the vector search entirely. This ensures precise lookups like `"Old Fashioned"` return instant, deterministic results.

### 2. Short Query Fallback

For queries shorter than 4 characters, a text-based fallback scans cocktail titles, descriptive titles, and ingredient names directly. This avoids sending low-signal strings through the embedding model.

### 3. Semantic Vector Search

For standard queries, the search flow is:

1. **Embed the query** — The free-text input is converted into a dense vector using HuggingFace Inference API (`feature-extraction` task via `langchain-huggingface`).
2. **Query Qdrant** — The embedding is compared against stored cocktail description chunk vectors using cosine similarity, filtered by a configurable score threshold.
3. **Build filters** — Natural-language cues in the query are parsed to construct Qdrant metadata filters:
   - **IBA classification** — e.g., `"iba cocktails"`
   - **Glassware** — e.g., `"served in a coupe"`
   - **Ingredient exclusions** — e.g., `"without vodka"`, `"no rum"`
   - **Ingredient count / prep time / serves** — numeric range filters
4. **Aggregate multi-chunk results** — A single cocktail may have multiple embedded description chunks. Results are deduplicated by cocktail ID and aggregated with search statistics:
   - `total_score` — sum of all matching chunk scores
   - `max_score` — highest individual chunk score
   - `avg_score` — average across matching chunks
   - `weighted_score` — average score boosted by hit count (up to 40% boost for 5+ chunk hits), rewarding cocktails with broad topical matches
5. **Sort and paginate** — Results are sorted by `weighted_score` descending, with optional re-ranking for rating-oriented queries (e.g., `"top rated"`).

### 4. Typeahead Search

The typeahead endpoint provides fast prefix-based suggestions without vector search. It matches the query against cocktail titles using `startsWith` first, then fills remaining slots with `contains` matches.

### 5. Browse Mode

When no free-text query is provided, the API returns an alphabetically sorted, paginated list of all cocktails. An optional `matches` parameter allows filtering to a specific set of cocktail IDs.

---

## Vector Database

The API uses [Qdrant](https://qdrant.tech/) as its vector database for storing and querying cocktail embeddings.

### Embedding Storage

Each cocktail is broken into multiple **description chunks** (e.g., overview, ingredients, history, preparation). Each chunk is embedded independently and stored as a separate vector point in Qdrant with rich metadata:

| Metadata Field | Description |
|---|---|
| `cocktail_id` | Unique cocktail identifier |
| `category` | Chunk category (e.g., description, ingredients) |
| `title` | Lowercase cocktail name |
| `is_iba` | IBA classification flag |
| `serves`, `prep_time_minutes` | Numeric attributes for filtering |
| `ingredient_names` | Full ingredient name list |
| `ingredient_words` | Individual words from ingredient names (≥3 chars) |
| `glassware_values` | Glassware type enums |
| `rating` | Cocktail rating |
| `keywords_*` | AI-generated keyword facets (base spirit, flavor profile, technique, season, occasion, mood, etc.) |
| `model` | Serialized full cocktail model JSON |

### Configuration

| Environment Variable | Description | Default |
|---|---|---|
| `QDRANT_HOST` | Qdrant server URL | _(required)_ |
| `QDRANT_PORT` | Qdrant server port | `6333` |
| `QDRANT_API_KEY` | API key for authentication | _(optional)_ |
| `QDRANT_COLLECTION_NAME` | Vector collection name | _(required)_ |
| `QDRANT_VECTOR_SIZE` | Embedding dimensionality | _(required)_ |
| `QDRANT_USE_HTTPS` | Enable HTTPS | `true` |
| `QDRANT_SEMANTIC_SEARCH_LIMIT` | Max vectors returned per search | `30` |
| `QDRANT_SEMANTIC_SEARCH_SCORE_THRESHOLD` | Minimum similarity score | `0.0` |
| `QDRANT_SEMANTIC_SEARCH_TOTAL_SCORE_THRESHOLD` | Minimum total score across chunks | `0.0` |

---

## API Endpoints

### Semantic Search

#### `GET /v1/cocktails/search`

Performs a semantic search for cocktails based on a free-text query.

| Parameter | Type | Description |
|---|---|---|
| `freetext` | `string` | Natural-language search query |
| `skip` | `int` | Number of results to skip (default: `0`) |
| `take` | `int` | Number of results to return (default: `10`) |
| `m` | `list[string]` | Cocktail IDs to include in results |
| `m_ex` | `bool` | If `true`, return only matched IDs |
| `inc` | `list[enum]` | Data includes: `mainImages`, `searchTiles`, `descriptiveTitle` |
| `fi` | `list[string]` | Metadata filters |

#### `GET /v1/cocktails/typeahead`

Provides typeahead/autocomplete suggestions for cocktail names.

| Parameter | Type | Description |
|---|---|---|
| `freetext` | `string` | Partial cocktail name |
| `skip` | `int` | Number of results to skip (default: `0`) |
| `take` | `int` | Number of results to return (default: `10`) |
| `fi` | `list[string]` | Metadata filters |

### Embeddings

#### `PUT /v1/cocktails/embeddings`

Ingests cocktail data by generating and storing vector embeddings. Requires OAuth2 authentication with `write:embeddings` scope.

Accepts a request body containing content chunks, a cocktail embedding model, and optional keyword facets.

### Health

#### `GET /v1/health`

Returns the health status of the API.

### Documentation

#### `GET /scalar/v1`

Interactive API documentation powered by [Scalar](https://github.com/scalar/scalar).

---

## Authentication & Authorization

- **APIM Host Key** — All endpoints are protected by an API Management host key (`APIM_HOST_KEY`), validated via a decorator on each route handler.
- **OAuth2 (Auth0)** — The embedding endpoint additionally requires a valid JWT with `write:embeddings` scope, validated against the configured OAuth2 provider.

---

## Observability

The API is instrumented with [OpenTelemetry](https://opentelemetry.io/) for distributed tracing, logging, and metrics export via OTLP. Configuration is managed through environment variables:

| Variable | Description |
|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint |
| `OTEL_SERVICE_NAME` | Service name for traces |
| `OTEL_SERVICE_NAMESPACE` | Service namespace |
| `OTEL_ENABLE_TRACING` | Enable/disable tracing |
| `OTEL_ENABLE_LOGGING` | Enable/disable log export |
| `OTEL_ENABLE_CONSOLE_LOGGING` | Enable/disable console output |

---

## Error Handling

All error responses follow [RFC 7807 Problem Details](https://www.rfc-editor.org/rfc/rfc7807) format with `application/problem+json` content type. Custom exception handlers map FastAPI `HTTPException`, `RequestValidationError`, `ValidationError`, and generic exceptions to standardized problem detail responses.

---

## Getting Started

### Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/) 2.x
- Access to a Qdrant instance
- HuggingFace API token

### Install Dependencies

```bash
poetry install
```

### Configure Environment

Create a `.env` file with the required variables:

```env
QDRANT_HOST=http://localhost:6333
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=cocktails
QDRANT_VECTOR_SIZE=768
HUGGINGFACE_INFERENCE_MODEL=your-model-name
HUGGINGFACE_API_TOKEN=your-token
APIM_HOST_KEY=your-api-key
OAUTH_DOMAIN=your-auth0-domain
OAUTH_AUDIENCE=your-audience
OAUTH_CLIENT_ID=your-client-id
OAUTH_ISSUER=https://your-auth0-domain/
```

### Run the Development Server

```bash
poetry run uvicorn src.cezzis_com_cocktails_aisearch.main:app --reload --port 8010
```

### Run Tests

```bash
poetry run pytest
```

### Docker

```bash
docker build -t cezzis-cocktails-aisearch .
docker run -p 8010:8010 --env-file .env cezzis-cocktails-aisearch
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Web Framework | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| Vector Database | [Qdrant](https://qdrant.tech/) |
| Embeddings | [HuggingFace Inference API](https://huggingface.co/inference-api) via LangChain |
| CQRS / Mediator | [MediatR](https://pypi.org/project/mediatr/) |
| Dependency Injection | [Injector](https://pypi.org/project/injector/) |
| Configuration | [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) |
| Observability | [OpenTelemetry](https://opentelemetry.io/) |
| Auth | OAuth2 / Auth0, APIM Host Key |
| API Docs | [Scalar](https://github.com/scalar/scalar) |
| Containerization | Docker (multi-stage build) |
| Package Management | [Poetry](https://python-poetry.org/) |

## License

This project is licensed under the MIT License.