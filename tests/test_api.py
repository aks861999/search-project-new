"""
FastAPI endpoint tests using httpx.AsyncClient + ASGITransport.

All external dependencies (OpenSearch client, embedding model, Redis)
are mocked at the dependency-injection level so the tests are fully
self-contained — no live services required.

Patching strategy
-----------------
api/main.py imports the cache module as ``import api.cache as _cache`` and
calls ``_cache.cache_get()``, ``_cache.ping_redis()`` etc.  This means we
must patch ``api.cache.cache_get`` (the attribute on the SOURCE module),
NOT ``api.main.cache_get`` (which would be a stale binding).
Similarly, ``_os_client`` and ``_embedding_model`` are module-level variables
in api.main, so those ARE patched as ``api.main._os_client``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import status
from httpx import ASGITransport, AsyncClient

from api.main import app
from api.models import SearchMode


# ── Shared mock factories ─────────────────────────────────────────────────────

def _make_os_response(total: int = 1, took: int = 3, hits=None) -> dict:
    if hits is None:
        hits = [
            {
                "_id": "B001",
                "_score": 1.0,
                "_source": {
                    "parent_asin": "B001",
                    "title": "Organic Shampoo",
                    "description": "Great for dry hair.",
                    "features": "Sulfate-free",
                    "main_category": "All Beauty",
                    "sub_category": "Hair Care",
                    "store": "GreenBrand",
                    "price": 14.99,
                    "average_rating": 4.5,
                    "rating_number": 300,
                    "primary_image_url": "https://cdn.example.com/img.jpg",
                },
            }
        ]
    return {
        "took": took,
        "timed_out": False,
        "hits": {"total": {"value": total, "relation": "eq"}, "hits": hits},
    }


def _make_mock_os_client() -> MagicMock:
    client = MagicMock()
    client.search = MagicMock(return_value=_make_os_response())
    client.cluster.health = MagicMock(return_value={"status": "green"})
    client.index = MagicMock(return_value={"result": "created", "_id": "B001"})
    return client


def _make_mock_model() -> MagicMock:
    model = MagicMock()
    arr = MagicMock()
    arr.tolist = MagicMock(return_value=[0.01] * 384)
    model.encode = MagicMock(return_value=arr)
    return model


# ── Standard client fixture ───────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client():
    """
    Async HTTP test client with all dependencies mocked.
    Cache functions are accessed via module reference in main.py,
    so we patch api.cache.* (the source) not api.main.* (stale bindings).
    """
    mock_os = _make_mock_os_client()
    mock_model = _make_mock_model()
    with (
        patch("api.dependencies._os_client", mock_os),
        patch("api.dependencies._embedding_model", mock_model),
        patch("api.cache.cache_get", new=AsyncMock(return_value=None)),
        patch("api.cache.cache_set", new=AsyncMock()),
        patch("api.cache.ping_redis", new=AsyncMock(return_value=True)),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_health_response_schema(self, client):
        body = (await client.get("/health")).json()
        assert "status" in body and "opensearch" in body
        assert "redis" in body and "version" in body

    @pytest.mark.asyncio
    async def test_health_ok_when_services_up(self, client):
        body = (await client.get("/health")).json()
        assert body["status"] == "ok"
        assert body["redis"] == "ok"
        assert body["opensearch"] in ("green", "yellow")

    @pytest.mark.asyncio
    async def test_health_degraded_when_redis_down(self):
        with (
            patch("api.dependencies._os_client", _make_mock_os_client()),
            patch("api.dependencies._embedding_model", _make_mock_model()),
            patch("api.cache.cache_get", new=AsyncMock(return_value=None)),
            patch("api.cache.cache_set", new=AsyncMock()),
            patch("api.cache.ping_redis", new=AsyncMock(return_value=False)),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                body = (await ac.get("/health")).json()
                assert body["status"] == "degraded"
                assert body["redis"] == "unreachable"


# ── /search ───────────────────────────────────────────────────────────────────

class TestSearchEndpoint:
    @pytest.mark.asyncio
    async def test_basic_search_returns_200(self, client):
        assert (await client.get("/search?q=shampoo")).status_code == 200

    @pytest.mark.asyncio
    async def test_search_response_schema(self, client):
        body = (await client.get("/search?q=moisturizer")).json()
        for key in ("total", "hits", "took_ms", "mode", "cached"):
            assert key in body

    @pytest.mark.asyncio
    async def test_search_hit_schema(self, client):
        hit = (await client.get("/search?q=shampoo")).json()["hits"][0]
        assert hit["id"] == "B001"
        assert hit["title"] == "Organic Shampoo"
        assert hit["price"] == 14.99
        assert hit["average_rating"] == 4.5

    @pytest.mark.asyncio
    async def test_default_mode_is_hybrid(self, client):
        assert (await client.get("/search?q=shampoo")).json()["mode"] == "hybrid"

    @pytest.mark.asyncio
    async def test_lexical_mode(self, client):
        resp = await client.get("/search?q=shampoo&mode=lexical")
        assert resp.status_code == 200
        assert resp.json()["mode"] == "lexical"

    @pytest.mark.asyncio
    async def test_semantic_mode(self, client):
        resp = await client.get("/search?q=shampoo&mode=semantic")
        assert resp.status_code == 200
        assert resp.json()["mode"] == "semantic"

    @pytest.mark.asyncio
    async def test_with_category_filter(self, client):
        assert (await client.get("/search?q=serum&category=All+Beauty")).status_code == 200

    @pytest.mark.asyncio
    async def test_with_price_range(self, client):
        assert (await client.get("/search?q=cream&min_price=5&max_price=30")).status_code == 200

    @pytest.mark.asyncio
    async def test_with_min_rating(self, client):
        assert (await client.get("/search?q=lotion&min_rating=4.0")).status_code == 200

    @pytest.mark.asyncio
    async def test_pagination_params(self, client):
        assert (await client.get("/search?q=shampoo&size=5&from=10")).status_code == 200

    @pytest.mark.asyncio
    async def test_empty_query_returns_422(self, client):
        assert (await client.get("/search?q=")).status_code == 422

    @pytest.mark.asyncio
    async def test_missing_query_returns_422(self, client):
        assert (await client.get("/search")).status_code == 422

    @pytest.mark.asyncio
    async def test_size_over_100_returns_422(self, client):
        assert (await client.get("/search?q=shampoo&size=200")).status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_mode_returns_422(self, client):
        assert (await client.get("/search?q=shampoo&mode=notamode")).status_code == 422

    @pytest.mark.asyncio
    async def test_cached_flag_false_on_fresh_result(self, client):
        assert (await client.get("/search?q=shampoo")).json()["cached"] is False

    @pytest.mark.asyncio
    async def test_cached_flag_true_when_cache_hit(self):
        """When cache_get returns a payload the endpoint must return cached=True."""
        cached_payload = {
            "total": 1,
            "hits": [{
                "id": "B001", "score": 1.0, "title": "Cached Shampoo",
                "description": "", "features": "", "main_category": "",
                "sub_category": "", "store": "", "price": None,
                "average_rating": None, "rating_number": None,
                "primary_image_url": "", "parent_asin": "B001",
            }],
            "took_ms": 2, "mode": "hybrid", "cached": False,
        }
        with (
            patch("api.dependencies._os_client", _make_mock_os_client()),
            patch("api.dependencies._embedding_model", _make_mock_model()),
            patch("api.cache.cache_get", new=AsyncMock(return_value=cached_payload)),
            patch("api.cache.cache_set", new=AsyncMock()),
            patch("api.cache.ping_redis", new=AsyncMock(return_value=True)),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/search?q=shampoo")
                assert resp.status_code == 200
                body = resp.json()
                assert body["cached"] is True
                assert body["hits"][0]["title"] == "Cached Shampoo"


# ── /nl-search ────────────────────────────────────────────────────────────────

class TestNLSearchEndpoint:
    @pytest.mark.asyncio
    async def test_nl_search_returns_200(self, client):
        resp = await client.post("/nl-search",
                                 json={"query": "cheap moisturizer for dry skin under $20"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_nl_search_response_schema(self, client):
        body = (await client.post("/nl-search",
                                  json={"query": "organic shampoo for curly hair"})).json()
        for key in ("total", "hits", "took_ms", "parsed_query", "cached"):
            assert key in body

    @pytest.mark.asyncio
    async def test_parsed_query_fields(self, client):
        pq = (await client.post("/nl-search",
                                json={"query": "top rated serum under $30"})).json()["parsed_query"]
        for key in ("semantic_query", "filters", "boost_terms", "search_mode"):
            assert key in pq

    @pytest.mark.asyncio
    async def test_price_extracted_from_nl(self, client):
        filters = (await client.post(
            "/nl-search", json={"query": "moisturizer under $25"}
        )).json()["parsed_query"]["filters"]
        assert filters["price_max"] == 25.0

    @pytest.mark.asyncio
    async def test_rating_extracted_from_nl(self, client):
        filters = (await client.post(
            "/nl-search", json={"query": "top rated anti-aging cream"}
        )).json()["parsed_query"]["filters"]
        assert filters["rating_min"] == 4.0

    @pytest.mark.asyncio
    async def test_short_query_returns_422(self, client):
        assert (await client.post("/nl-search", json={"query": "ab"})).status_code == 422

    @pytest.mark.asyncio
    async def test_missing_query_field_returns_422(self, client):
        assert (await client.post("/nl-search", json={})).status_code == 422

    @pytest.mark.asyncio
    async def test_custom_size_respected(self, client):
        resp = await client.post("/nl-search",
                                 json={"query": "hair care products", "size": 3})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_nl_cached_flag_true_when_cache_hit(self):
        cached_nl_payload = {
            "total": 1,
            "hits": [{
                "id": "B002", "score": 0.9, "title": "Cached NL Hit",
                "description": "", "features": "", "main_category": "",
                "sub_category": "", "store": "", "price": 9.99,
                "average_rating": 4.0, "rating_number": 50,
                "primary_image_url": "", "parent_asin": "B002",
            }],
            "took_ms": 1,
            "parsed_query": {
                "semantic_query": "shampoo",
                "filters": {"main_category": None, "price_min": None,
                            "price_max": None, "rating_min": None},
                "boost_terms": [],
                "search_mode": "hybrid",
            },
            "cached": False,
        }
        with (
            patch("api.dependencies._os_client", _make_mock_os_client()),
            patch("api.dependencies._embedding_model", _make_mock_model()),
            patch("api.cache.cache_get", new=AsyncMock(return_value=cached_nl_payload)),
            patch("api.cache.cache_set", new=AsyncMock()),
            patch("api.cache.ping_redis", new=AsyncMock(return_value=True)),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/nl-search", json={"query": "shampoo query"})
                assert resp.status_code == 200
                assert resp.json()["cached"] is True


# ── /suggest ──────────────────────────────────────────────────────────────────

class TestSuggestEndpoint:
    @pytest.mark.asyncio
    async def test_suggest_returns_200(self, client):
        assert (await client.get("/suggest?prefix=moist")).status_code == 200

    @pytest.mark.asyncio
    async def test_suggest_response_schema(self, client):
        body = (await client.get("/suggest?prefix=sham")).json()
        assert "suggestions" in body
        assert body["prefix"] == "sham"

    @pytest.mark.asyncio
    async def test_suggest_returns_list(self, client):
        assert isinstance(
            (await client.get("/suggest?prefix=org")).json()["suggestions"], list
        )

    @pytest.mark.asyncio
    async def test_suggest_title_in_results(self, client):
        suggestions = (await client.get("/suggest?prefix=Organic")).json()["suggestions"]
        assert any("Organic" in s for s in suggestions)

    @pytest.mark.asyncio
    async def test_missing_prefix_returns_422(self, client):
        assert (await client.get("/suggest")).status_code == 422

    @pytest.mark.asyncio
    async def test_empty_prefix_returns_422(self, client):
        assert (await client.get("/suggest?prefix=")).status_code == 422

    @pytest.mark.asyncio
    async def test_size_param_accepted(self, client):
        assert (await client.get("/suggest?prefix=sha&size=3")).status_code == 200


# ── /index/product ────────────────────────────────────────────────────────────

class TestIndexProductEndpoint:
    def _product_payload(self, **overrides) -> dict:
        base = {
            "id": "B099TEST", "score": 0.0,
            "title": "Test Conditioner",
            "description": "Deep conditioning treatment.",
            "features": "Moisturizing, Paraben-free",
            "main_category": "All Beauty", "sub_category": "Hair Care",
            "store": "TestBrand", "price": 18.50, "average_rating": 4.2,
            "rating_number": 85,
            "primary_image_url": "https://cdn.example.com/cond.jpg",
            "parent_asin": "B099TEST",
        }
        base.update(overrides)
        return base

    @pytest.mark.asyncio
    async def test_index_product_returns_201(self, client):
        resp = await client.post("/index/product", json=self._product_payload())
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_index_product_response_contains_result(self, client):
        body = (await client.post("/index/product", json=self._product_payload())).json()
        assert "result" in body and "id" in body

    @pytest.mark.asyncio
    async def test_index_product_title_optional(self, client):
        """title has default='' in ProductHit so omitting it should still return 201."""
        payload = self._product_payload()
        payload.pop("title")
        resp = await client.post("/index/product", json=payload)
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_index_product_calls_os_index(self, client):
        """The OpenSearch index() method must be called exactly once per request."""
        mock_os = _make_mock_os_client()
        mock_model = _make_mock_model()
        with (
            patch("api.dependencies._os_client", mock_os),
            patch("api.dependencies._embedding_model", mock_model),
            patch("api.cache.cache_get", new=AsyncMock(return_value=None)),
            patch("api.cache.cache_set", new=AsyncMock()),
            patch("api.cache.ping_redis", new=AsyncMock(return_value=True)),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/index/product", json=self._product_payload())
                assert resp.status_code == 201
                mock_os.index.assert_called_once()
