# Würth Product Search Pipeline

A production-grade hybrid product search prototype built for Würth IT GmbH. Combines BM25 lexical search, semantic kNN vector search, and a natural language query chain over an Amazon product catalogue indexed in OpenSearch 3.5.0.

---

## Architecture

```
[HuggingFace Dataset]
        ↓
[Ingestion Service]  →  batch embed (BAAI/bge-small-en-v1.5)  →  helpers.bulk()
        ↓
[OpenSearch 3.5.0]  ←→  [knn_vector (faiss/hnsw) + BM25 index]
        ↓
[Hybrid Search Core]  (BM25 weight=0.4 + kNN weight=0.6, min-max normalisation)
        ↓
[NL Query Chain]  (LangChain 0.3 LCEL + Gemini Flash / stub parser)
        ↓
[FastAPI 0.115 Gateway]  ←→  [Redis 7.4 Cache, TTL=300s]
        ↓
[Client]
```

---

## Project Structure

```
wurth-search/
├── api/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, lifespan, all endpoints
│   ├── search.py        # lexical_search(), semantic_search(), hybrid_search()
│   ├── nl_query.py      # LangChain LCEL NL query chain + stub parser
│   ├── cache.py         # aioredis 2.x async caching helpers
│   ├── models.py        # Pydantic v2 request/response models
│   ├── requirements.txt
│   └── Dockerfile
├── ingestion/
│   ├── __init__.py
│   ├── pipeline.py      # ETL: HuggingFace → preprocess → embed → bulk index
│   ├── schema.py        # OpenSearch index mapping + hybrid pipeline definition
│   ├── requirements.txt
│   └── Dockerfile
├── config/
│   ├── __init__.py
│   └── settings.py      # pydantic-settings env config (shared by all services)
├── infra/
│   └── docker-compose.yml
├── tests/
│   ├── conftest.py
│   ├── test_search.py   # pytest tests for all three search functions
│   └── test_nl_cache.py # pytest tests for NL parsing and cache utilities
├── .env.example
├── .gitignore
├── Makefile
├── pyproject.toml       # pytest + coverage config
└── README.md
```

---

## Tech Stack

| Layer | Tool | Version |
|---|---|---|
| Search Engine | OpenSearch | 3.5.0 |
| Python Client | opensearch-py | 2.8.x |
| API Framework | FastAPI + Pydantic v2 | 0.115.x |
| Embedding Model | sentence-transformers | 3.x |
| NL Query | LangChain LCEL | 0.3.x |
| Caching | aioredis | 2.x |
| Orchestration | Docker Compose | v2 |

---

## Quick Start

### 1. Prerequisites

- Docker + Docker Compose v2
- Python 3.11+
- (Optional) Google API key for Gemini-powered NL queries

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set GOOGLE_API_KEY if you want LLM-powered NL search
```

### 3. Start the stack

```bash
make up
# or manually:
docker compose -f infra/docker-compose.yml up -d --build
```

Services started:
- **OpenSearch**: http://localhost:9200
- **OpenSearch Dashboards**: http://localhost:5601
- **Redis**: localhost:6379
- **API**: http://localhost:8000 — docs at http://localhost:8000/docs

### 4. Run ingestion (loads Amazon Beauty dataset)

```bash
make ingest
# or:
docker compose -f infra/docker-compose.yml --profile ingest up ingestion
```

This will:
1. Download the `McAuley-Lab/Amazon-Reviews-2023 / raw_meta_All_Beauty` dataset (~few thousand products)
2. Preprocess and build `embedding_text` per product
3. Batch encode with `BAAI/bge-small-en-v1.5` (batch_size=128, normalize_embeddings=True)
4. Bulk index into OpenSearch via `helpers.bulk()` (chunk_size=500)

### 5. Search

```bash
# Hybrid search (default)
curl "http://localhost:8000/search?q=moisturizer+for+dry+skin&mode=hybrid&size=5"

# Lexical only
curl "http://localhost:8000/search?q=shampoo&mode=lexical&category=All+Beauty&max_price=20"

# Semantic only
curl "http://localhost:8000/search?q=anti-aging+face+serum&mode=semantic"

# Natural language search
curl -X POST http://localhost:8000/nl-search \
  -H "Content-Type: application/json" \
  -d '{"query": "cheap organic shampoo for curly hair under $15", "size": 10}'

# Autocomplete
curl "http://localhost:8000/suggest?prefix=moist"

# Health check
curl "http://localhost:8000/health"
```

---

## API Reference

### `GET /search`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `q` | string | required | Search query |
| `category` | string | null | Filter by `main_category` |
| `min_price` | float | null | Minimum price |
| `max_price` | float | null | Maximum price |
| `min_rating` | float | null | Minimum average rating (0–5) |
| `mode` | enum | `hybrid` | `lexical` / `semantic` / `hybrid` |
| `size` | int | 10 | Results per page (max 100) |
| `from` | int | 0 | Pagination offset |

### `POST /nl-search`

```json
{
  "query": "natural sulfate-free shampoo for dry hair under $20",
  "size": 10
}
```

Returns structured search results plus `parsed_query` showing what the LLM (or stub parser) extracted.

### `GET /suggest`

| Parameter | Type | Description |
|---|---|---|
| `prefix` | string | Autocomplete prefix |
| `size` | int | Max suggestions (default 5) |

### `GET /health`

Returns OpenSearch cluster status and Redis ping result.

---

## Search Modes

### Lexical (`mode=lexical`)
- `multi_match` across `title^3`, `features^2`, `description^1`
- `function_score` boosting by `average_rating` and `log(rating_number + 1)`
- Best for exact keyword matches

### Semantic (`mode=semantic`)
- kNN query on `embedding_vector` (384-dim, faiss HNSW, innerproduct space)
- Query encoded at runtime with same `BAAI/bge-small-en-v1.5` model
- Best for concept/intent matching

### Hybrid (`mode=hybrid`) — recommended
- Runs BM25 + kNN as sub-queries concurrently
- `normalization_processor`: min-max score normalisation
- `arithmetic_mean` combiner with weights `[0.4 BM25, 0.6 kNN]`
- Falls back to lexical if hybrid pipeline is not registered

---

## NL Query Chain

When `GOOGLE_API_KEY` is set, the `/nl-search` endpoint uses:

```
User NL query
    → ChatGoogleGenerativeAI (gemini-2.0-flash)
    → structured JSON: { semantic_query, filters, boost_terms }
    → hybrid_search() with extracted params
```

Without a key, a regex-based **stub parser** is used that handles:
- Price ranges: "under $25", "between $10 and $50"
- Rating hints: "top rated", "best" → `rating_min=4.0`
- Boost terms: "organic", "sulfate-free", "vegan", etc.

---

## Index Schema

Key field mappings:

| Field | Type | Purpose |
|---|---|---|
| `title` | `text` + `keyword` | BM25 + exact filter |
| `embedding_vector` | `knn_vector` dim=384 | Semantic search |
| `main_category` | `keyword` | Facet filter |
| `price` | `float` | Range filter |
| `average_rating` | `float` | Score boost + filter |
| `rating_number` | `integer` | Popularity signal |

Vector method: `faiss` engine, `hnsw`, `innerproduct` space, `ef_construction=128`, `m=16`.

---

## Running Tests

```bash
# Install test deps and run
make test

# With coverage report
make test-cov
```

Tests use a **fully mocked OpenSearch client** — no live cluster required. Coverage includes:
- All three search functions (`lexical_search`, `semantic_search`, `hybrid_search`)
- All query builders (`_build_lexical_query`, `_build_semantic_query`, `_build_hybrid_query`, `_build_filter_clauses`)
- NL query parsing (`_stub_parse`, `_dict_to_parsed_query`, `_extract_json`)
- Cache key generation (`make_cache_key`)
- `ProductHit.from_hit()` field mapping
- Hybrid fallback to lexical on pipeline exception

---

## Configuration Reference

All settings are loaded via `pydantic-settings` from environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENSEARCH_HOST` | `localhost` | OpenSearch hostname |
| `OPENSEARCH_PORT` | `9200` | OpenSearch port |
| `INDEX_NAME` | `products` | Target index name |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence-transformer model |
| `EMBEDDING_DIM` | `384` | Vector dimension |
| `BATCH_SIZE` | `128` | Embedding batch size |
| `CHUNK_SIZE` | `500` | OpenSearch bulk chunk size |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `CACHE_TTL` | `300` | Cache TTL in seconds |
| `GOOGLE_API_KEY` | _(unset)_ | Gemini API key (optional) |
| `LLM_MODEL` | `gemini-2.0-flash` | LLM model name |
| `HYBRID_BM25_WEIGHT` | `0.4` | BM25 weight in hybrid fusion |
| `HYBRID_KNN_WEIGHT` | `0.6` | kNN weight in hybrid fusion |
| `KNN_K` | `100` | kNN candidates per query |

---

## Stopping the Stack

```bash
make down        # stop containers (keep volumes)
make clean       # stop containers + remove volumes + clean caches
```
