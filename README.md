
# Cezzis.com Cocktails AI Search API

The backend API powering semantic cocktail search for [Cezzis.com](https://www.cezzis.com). Built with FastAPI, Qdrant vector database, and HuggingFace embeddings, this service enables natural-language cocktail discovery through vector similarity search, typeahead suggestions, and filtered browsing.

This API works in conjunction with the [Cezzis.com Ingestion Agentic Workflow](https://github.com/mtnvencenzo/cezzis-com-ingestion-agentic-wf), which handles cocktail data ingestion, content chunking, and embedding orchestration.

---

## Architecture

```
┌─────────────────┐      ┌──────────────────────┐      ┌─────────────────────┐
│   Client / MCP  │─────▶│  FastAPI Application  │─────▶│  Qdrant Vector DB   │
│                 │◀─────│  (Uvicorn @ :8010)    │◀─────│  (Named Vectors)    │
└─────────────────┘      └──────────┬───────────┘      └─────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
          ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
          │  TEI          │ │  TEI          │ │  TEI          │
          │  Bi-Encoder   │ │  SPLADE       │ │  Cross-Encoder│
          │  (Dense)      │ │  (Sparse)     │ │  (Reranker)   │
          │  :8989        │ │  :8991        │ │  :8990        │
          └──────────────┘ └──────────────┘ └──────────────┘
```

The application follows a **CQRS pattern** using [MediatR](https://pypi.org/project/mediatr/) for command/query separation, with **dependency injection** via [Injector](https://pypi.org/project/injector/) and **Pydantic** models for request/response validation.

All three TEI (Text Embeddings Inference) services run locally or as containers, providing dense embeddings, SPLADE sparse embeddings, and cross-encoder reranking respectively.

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

The search pipeline uses a multi-stage retrieval approach combining hybrid vector search, cross-encoder reranking, and structured metadata filtering to return the most relevant cocktail results.

### Pipeline Overview

```
  Query
    │
    ▼
  ┌─────────────────────────┐
  │ 1. Exact Name Match?    │──── Yes ──▶ Return immediately
  │    (fuzzy threshold 82) │
  └────────┬────────────────┘
           │ No
           ▼
  ┌─────────────────────────┐
  │ 2. Short Query (<4 ch)? │──── Yes ──▶ Text-based fallback on cached data
  └────────┬────────────────┘
           │ No
           ▼
  ┌─────────────────────────┐
  │ 3. Parse Query Filters  │  IBA, glassware, ingredients, spirit, flavor,
  │    (structured cues)    │  technique, strength, temperature, season, etc.
  └────────┬────────────────┘
           │
           ▼
  ┌─────────────────────────┐
  │ 4. Hybrid Vector Search │  Dense (bi-encoder) + Sparse (SPLADE)
  │    with RRF Fusion      │  via Qdrant prefetch + Fusion.RRF
  └────────┬────────────────┘
           │
           ▼
  ┌─────────────────────────┐
  │ 5. Multi-Chunk          │  Deduplicate by cocktail ID,
  │    Aggregation          │  compute weighted scores
  └────────┬────────────────┘
           │
           ▼
  ┌─────────────────────────┐
  │ 6. Cross-Encoder        │  TEI /rerank endpoint re-scores
  │    Reranking            │  query-document pairs jointly
  └────────┬────────────────┘
           │
           ▼
  ┌─────────────────────────┐
  │ 7. Final Sort &         │  Rating overrides, pagination
  │    Pagination           │
  └─────────────────────────┘
```

### 1. Exact Name Matching

When the query matches a cocktail name (or closely via fuzzy matching with a threshold of 82 using [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz)), that cocktail is returned immediately — bypassing the vector search entirely. This ensures precise lookups like `"Old Fashioned"` return instant, deterministic results.

### 2. Short Query Fallback

For queries shorter than 4 characters, a text-based fallback scans cocktail titles, descriptive titles, and ingredient names directly against cached data. This avoids sending low-signal strings through the embedding model.

### 3. Query Filter Parsing

Natural-language cues in the query are parsed to construct Qdrant metadata filters before vector search:

| Filter Type | Example Query | Metadata Field |
|---|---|---|
| IBA classification | `"iba cocktails"` | `is_iba` |
| Glassware | `"served in a coupe"` | `glassware_values` |
| Ingredient exclusion | `"without vodka"`, `"no rum"` | `ingredient_words` (must_not) |
| Ingredient inclusion | `"made with gin"`, `"containing lime"` | `ingredient_words` (must) |
| Base spirit | `"bourbon cocktails"` | `keywords_base_spirit` |
| Flavor profile | `"refreshing citrus"` | `keywords_flavor_profile` |
| Cocktail family | `"sour"`, `"tiki"`, `"fizz"` | `keywords_cocktail_family` |
| Technique | `"shaken"`, `"stirred"` | `keywords_technique` |
| Strength | `"strong"`, `"light"` | `keywords_strength` |
| Temperature | `"frozen"`, `"hot"` | `keywords_temperature` |
| Season | `"summer"`, `"winter"` | `keywords_season` |
| Occasion | `"brunch"`, `"nightcap"` | `keywords_occasion` |
| Mood | `"sophisticated"`, `"fun"` | `keywords_mood` |
| Numeric ranges | ingredient count, prep time, serves | Range filters |

### 4. Hybrid Vector Search (Dense + Sparse via RRF)

The core retrieval stage combines two complementary search strategies using Qdrant's native [prefetch + fusion](https://qdrant.tech/documentation/concepts/hybrid-queries/) mechanism:

#### Dense Search (Bi-Encoder)

- The query is embedded into a 768-dimensional dense vector using a bi-encoder model (`sentence-transformers/all-mpnet-base-v2`) served via [TEI](https://github.com/huggingface/text-embeddings-inference).
- This captures **semantic similarity** — understanding that "refreshing citrus drink" is related to a Margarita even without shared keywords.
- Dense embeddings are cached per query using an LRU cache (1024 entries) to avoid redundant inference calls.

#### Sparse Search (SPLADE)

- The same query is also encoded into a **sparse vector** using [SPLADE](https://github.com/naver/splade) (`naver/splade-cocondenser-ensembledistil`) served via TEI.
- SPLADE produces learned term-weight sparse representations that excel at **exact keyword matching** and **term expansion** — it can surface documents containing specific ingredient names or cocktail terms that the dense model might underweight.
- Sparse vectors have dynamic dimensionality with explicit `(indices, values)` representation using Qdrant's `SparseVector` type.

#### Reciprocal Rank Fusion (RRF)

Both search results are fused using **Reciprocal Rank Fusion** (`Fusion.RRF`), which combines rankings from both retrieval methods without requiring score normalization:

$$RRF(d) = \sum_{r \in R} \frac{1}{k + rank_r(d)}$$

where $k$ is a constant (typically 60) and $rank_r(d)$ is the rank of document $d$ in result set $r$.

This is implemented via Qdrant's `prefetch` mechanism, which runs both dense and sparse searches in parallel on the server side before fusing:

```python
qdrant_client.query_points(
    prefetch=[
        Prefetch(query=dense_vector, using="dense", limit=N, filter=query_filter),
        Prefetch(query=SparseVector(indices=..., values=...), using="sparse", limit=N, filter=query_filter),
    ],
    query=Fusion.RRF,
    limit=N,
)
```

#### Graceful Degradation

If SPLADE encoding fails or returns empty results, the search automatically falls back to **dense-only** search, ensuring availability even during sparse encoder outages.

### 5. Multi-Chunk Aggregation

Each cocktail has multiple embedded description chunks (overview, ingredients, history, preparation, etc.). After retrieval, results are deduplicated by cocktail ID and aggregated with search statistics:

| Statistic | Description |
|---|---|
| `total_score` | Sum of all matching chunk scores |
| `max_score` | Highest individual chunk score |
| `avg_score` | Average across matching chunks |
| `hit_count` | Number of chunks that matched |
| `weighted_score` | Combined relevance score (see formula below) |

**Weighted scoring formula:**

$$weighted\_score = max\_score \times 0.6 + avg\_score \times 0.3 + \log(hit\_count + 1) \times 0.1$$

- **max_score (60%)** — strongest single-chunk match is the primary signal
- **avg_score (30%)** — rewards consistent relevance across chunks  
- **log(hit_count) (10%)** — diminishing returns for breadth of matching

This prevents a cocktail with 5 low-scoring chunk hits (e.g., 0.3 each) from outranking one with 2 high-scoring hits (e.g., 0.8 each).

### 6. Cross-Encoder Reranking

After initial retrieval and aggregation, the top candidates are re-scored using a **cross-encoder** model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) served via TEI's `/rerank` endpoint.

Unlike bi-encoders (which encode query and document independently), the cross-encoder **jointly encodes** the query-document pair, enabling it to capture fine-grained interactions between the query and each cocktail's content. This produces more accurate relevance scores at the cost of higher latency (hence applied only to top candidates, not the full corpus).

The reranking step:

1. Builds a text representation for each candidate (title + descriptive title + ingredients)
2. Sends all `(query, document)` pairs to TEI `/rerank` in a single batch request
3. Filters candidates below the configured `score_threshold`
4. Re-sorts by cross-encoder score descending
5. Applies `top_k` limit

If the reranker is unavailable or fails, the original ordering is preserved (graceful degradation).

### 7. Typeahead Search

The typeahead endpoint provides fast prefix-based suggestions without vector search. It matches the query against cocktail titles using `startsWith` first, then fills remaining slots with `contains` matches.

### 8. Browse Mode

When no free-text query is provided, the API returns an alphabetically sorted, paginated list of all cocktails. An optional `matches` parameter allows filtering to a specific set of cocktail IDs.

---

## Vector Database

The API uses [Qdrant](https://qdrant.tech/) as its vector database for storing and querying cocktail embeddings.

### Collection Schema

The Qdrant collection uses **named vectors** to support hybrid search with both dense and sparse representations:

```json
{
  "vectors": {
    "dense": { "size": 768, "distance": "Cosine" }
  },
  "sparse_vectors": {
    "sparse": {}
  }
}
```

- **`dense`** — 768-dimensional dense vectors from the bi-encoder (`sentence-transformers/all-mpnet-base-v2`), used for semantic similarity search.
- **`sparse`** — Variable-dimensionality sparse vectors from SPLADE (`naver/splade-cocondenser-ensembledistil`), used for learned term-weight matching.

### Embedding Storage

Each cocktail is broken into multiple **description chunks** (e.g., overview, ingredients, history, preparation). Each chunk is embedded independently and stored as a separate vector point in Qdrant with both dense and sparse vectors plus rich metadata:

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
| `QDRANT_SEMANTIC_SEARCH_LIMIT` | Max vectors returned from RRF fusion | `30` |
| `QDRANT_SEMANTIC_SEARCH_PREFETCH_LIMIT` | Max vectors per prefetch branch (dense/sparse) | `100` |
| `QDRANT_SEMANTIC_SEARCH_SCORE_THRESHOLD` | Minimum similarity score | `0.0` |
| `QDRANT_SEMANTIC_SEARCH_TOTAL_SCORE_THRESHOLD` | Minimum total score across chunks | `0.0` |

### TEI Services Configuration

| Environment Variable | Description |
|---|---|
| `HUGGINGFACE_INFERENCE_MODEL` | Dense bi-encoder TEI endpoint (e.g., `http://localhost:8989`) |
| `HUGGINGFACE_API_TOKEN` | API token for TEI authentication |
| `RERANKER_ENDPOINT` | Cross-encoder reranker TEI endpoint (e.g., `http://localhost:8990`) |
| `RERANKER_API_KEY` | API key for reranker TEI |
| `RERANKER_SCORE_THRESHOLD` | Minimum cross-encoder score to retain a result |
| `SPLADE_ENDPOINT` | SPLADE sparse encoder TEI endpoint (e.g., `http://localhost:8991`) |
| `SPLADE_API_KEY` | API key for SPLADE TEI |

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
| Vector Database | [Qdrant](https://qdrant.tech/) (named vectors: dense + sparse) |
| Dense Embeddings | [TEI](https://github.com/huggingface/text-embeddings-inference) bi-encoder (`sentence-transformers/all-mpnet-base-v2`, 768 dims) |
| Sparse Embeddings | [TEI](https://github.com/huggingface/text-embeddings-inference) SPLADE (`naver/splade-cocondenser-ensembledistil`) |
| Cross-Encoder Reranking | [TEI](https://github.com/huggingface/text-embeddings-inference) reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| Hybrid Fusion | Reciprocal Rank Fusion (RRF) via Qdrant prefetch |
| Fuzzy Matching | [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) |
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