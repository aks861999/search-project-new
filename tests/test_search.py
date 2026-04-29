"""
Tests for api/search.py — lexical_search, semantic_search, hybrid_search.

The OpenSearch client is fully mocked using unittest.mock so no live
cluster is required.  All three async search functions are covered.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.models import ProductHit, SearchMode
from api.search import (
    _build_filter_clauses,
    _build_hybrid_query,
    _build_lexical_query,
    _build_semantic_query,
    hybrid_search,
    lexical_search,
    semantic_search,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_os_response(
    hits: list[dict] | None = None,
    total: int = 0,
    took: int = 5,
) -> dict:
    """Build a minimal OpenSearch response dict."""
    return {
        "took": took,
        "timed_out": False,
        "hits": {
            "total": {"value": total, "relation": "eq"},
            "hits": hits or [],
        },
    }


def _make_hit(
    doc_id: str = "B001",
    score: float = 1.0,
    title: str = "Test Product",
    price: float = 9.99,
    rating: float = 4.5,
) -> dict:
    return {
        "_index": "products",
        "_id": doc_id,
        "_score": score,
        "_source": {
            "parent_asin": doc_id,
            "title": title,
            "description": "A test product.",
            "features": "feature1 feature2",
            "main_category": "All Beauty",
            "sub_category": "",
            "store": "TestStore",
            "price": price,
            "average_rating": rating,
            "rating_number": 100,
            "primary_image_url": "https://example.com/img.jpg",
        },
    }


@pytest.fixture
def mock_client():
    """Return a MagicMock OpenSearch client with a pre-configured search method."""
    client = MagicMock()
    client.search = MagicMock(
        return_value=_make_os_response(
            hits=[_make_hit()], total=1, took=3
        )
    )
    return client


@pytest.fixture
def empty_client():
    """Return a MagicMock OpenSearch client that returns no hits."""
    client = MagicMock()
    client.search = MagicMock(return_value=_make_os_response(hits=[], total=0))
    return client


SAMPLE_VECTOR = [0.01] * 384
INDEX = "products"


# ── Filter clause builder tests ───────────────────────────────────────────────

class TestBuildFilterClauses:
    def test_no_filters_returns_empty_list(self):
        assert _build_filter_clauses() == []

    def test_category_filter(self):
        filters = _build_filter_clauses(category="All Beauty")
        assert any("term" in f for f in filters)
        term_filter = next(f for f in filters if "term" in f)
        assert term_filter["term"]["main_category"] == "All Beauty"

    def test_price_range_filter(self):
        filters = _build_filter_clauses(min_price=5.0, max_price=30.0)
        range_filter = next(f for f in filters if "range" in f)
        assert range_filter["range"]["price"]["gte"] == 5.0
        assert range_filter["range"]["price"]["lte"] == 30.0

    def test_min_price_only(self):
        filters = _build_filter_clauses(min_price=10.0)
        range_filter = next(f for f in filters if "range" in f)
        assert "gte" in range_filter["range"]["price"]
        assert "lte" not in range_filter["range"]["price"]

    def test_rating_filter(self):
        filters = _build_filter_clauses(min_rating=4.0)
        rating_filter = next(f for f in filters if "range" in f)
        assert rating_filter["range"]["average_rating"]["gte"] == 4.0

    def test_all_filters_combined(self):
        filters = _build_filter_clauses(
            category="Beauty", min_price=5.0, max_price=50.0, min_rating=3.5
        )
        assert len(filters) == 3  # term + price range + rating range


# ── Query builder tests ───────────────────────────────────────────────────────

class TestBuildLexicalQuery:
    def test_basic_structure(self):
        body = _build_lexical_query("shampoo", [], size=10)
        assert "query" in body
        assert body["size"] == 10
        assert body["from"] == 0

    def test_excludes_embedding_vector(self):
        body = _build_lexical_query("shampoo", [])
        excludes = body.get("_source", {}).get("excludes", [])
        assert "embedding_vector" in excludes

    def test_boost_terms_added(self):
        body = _build_lexical_query("shampoo", [], boost_terms=["organic", "sulfate-free"])
        should = body["query"]["function_score"]["query"]["bool"]["should"]
        assert len(should) == 2

    def test_with_filters(self):
        filters = _build_filter_clauses(category="All Beauty")
        body = _build_lexical_query("moisturizer", filters)
        filter_clauses = body["query"]["function_score"]["query"]["bool"]["filter"]
        assert len(filter_clauses) == 1


class TestBuildSemanticQuery:
    def test_basic_structure(self):
        body = _build_semantic_query(SAMPLE_VECTOR, [], k=100, size=10)
        assert "query" in body
        assert "knn" in body["query"]
        assert body["query"]["knn"]["embedding_vector"]["k"] == 100

    def test_vector_passed_correctly(self):
        body = _build_semantic_query(SAMPLE_VECTOR, [], k=50, size=5)
        assert body["query"]["knn"]["embedding_vector"]["vector"] == SAMPLE_VECTOR
        assert body["size"] == 5

    def test_filters_injected_into_knn(self):
        filters = _build_filter_clauses(category="All Beauty")
        body = _build_semantic_query(SAMPLE_VECTOR, filters)
        knn_params = body["query"]["knn"]["embedding_vector"]
        assert "filter" in knn_params
        assert knn_params["filter"]["bool"]["filter"] == filters

    def test_no_filters_no_filter_key(self):
        body = _build_semantic_query(SAMPLE_VECTOR, [])
        knn_params = body["query"]["knn"]["embedding_vector"]
        assert "filter" not in knn_params


class TestBuildHybridQuery:
    def test_hybrid_query_structure(self):
        body = _build_hybrid_query("shampoo", SAMPLE_VECTOR, [])
        assert "hybrid" in body["query"]
        assert "queries" in body["query"]["hybrid"]
        assert len(body["query"]["hybrid"]["queries"]) == 2

    def test_both_subqueries_present(self):
        body = _build_hybrid_query("shampoo", SAMPLE_VECTOR, [])
        queries = body["query"]["hybrid"]["queries"]
        has_bool = any("bool" in q for q in queries)
        has_knn = any("knn" in q for q in queries)
        assert has_bool
        assert has_knn

    def test_boost_terms_in_bm25_should(self):
        body = _build_hybrid_query("shampoo", SAMPLE_VECTOR, [], boost_terms=["organic"])
        bm25 = next(q for q in body["query"]["hybrid"]["queries"] if "bool" in q)
        assert len(bm25["bool"]["should"]) == 1

    def test_pagination_params(self):
        body = _build_hybrid_query("shampoo", SAMPLE_VECTOR, [], size=20, from_=10)
        assert body["size"] == 20
        assert body["from"] == 10


# ── Async search function tests ───────────────────────────────────────────────

class TestLexicalSearch:
    @pytest.mark.asyncio
    async def test_returns_search_response(self, mock_client):
        result = await lexical_search(mock_client, "shampoo", INDEX)
        assert result.mode == SearchMode.lexical
        assert result.total == 1
        assert len(result.hits) == 1
        assert result.hits[0].title == "Test Product"

    @pytest.mark.asyncio
    async def test_empty_results(self, empty_client):
        result = await lexical_search(empty_client, "xyznotexist", INDEX)
        assert result.total == 0
        assert result.hits == []

    @pytest.mark.asyncio
    async def test_calls_client_search(self, mock_client):
        await lexical_search(mock_client, "moisturizer", INDEX)
        mock_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_with_all_filters(self, mock_client):
        result = await lexical_search(
            mock_client,
            query="serum",
            index_name=INDEX,
            category="All Beauty",
            min_price=5.0,
            max_price=50.0,
            min_rating=4.0,
            boost_terms=["organic"],
            size=5,
            from_=0,
        )
        assert result.mode == SearchMode.lexical
        mock_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_took_ms_populated(self, mock_client):
        result = await lexical_search(mock_client, "conditioner", INDEX)
        assert result.took_ms == 3  # matches _make_os_response took=3

    @pytest.mark.asyncio
    async def test_hit_fields_mapped_correctly(self, mock_client):
        result = await lexical_search(mock_client, "shampoo", INDEX)
        hit = result.hits[0]
        assert hit.id == "B001"
        assert hit.price == 9.99
        assert hit.average_rating == 4.5
        assert hit.main_category == "All Beauty"


class TestSemanticSearch:
    @pytest.mark.asyncio
    async def test_returns_semantic_mode(self, mock_client):
        result = await semantic_search(mock_client, SAMPLE_VECTOR, INDEX)
        assert result.mode == SearchMode.semantic

    @pytest.mark.asyncio
    async def test_returns_hits(self, mock_client):
        result = await semantic_search(mock_client, SAMPLE_VECTOR, INDEX)
        assert result.total == 1
        assert len(result.hits) == 1

    @pytest.mark.asyncio
    async def test_calls_client_search(self, mock_client):
        await semantic_search(mock_client, SAMPLE_VECTOR, INDEX)
        mock_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_results(self, empty_client):
        result = await semantic_search(empty_client, SAMPLE_VECTOR, INDEX)
        assert result.total == 0

    @pytest.mark.asyncio
    async def test_with_filters(self, mock_client):
        result = await semantic_search(
            mock_client,
            SAMPLE_VECTOR,
            INDEX,
            category="Beauty",
            min_price=0.0,
            max_price=100.0,
            min_rating=3.0,
        )
        assert result.mode == SearchMode.semantic

    @pytest.mark.asyncio
    async def test_custom_k_and_size(self, mock_client):
        await semantic_search(mock_client, SAMPLE_VECTOR, INDEX, size=20, k=50)
        call_kwargs = mock_client.search.call_args
        body = call_kwargs[1]["body"] if "body" in call_kwargs[1] else call_kwargs[0][1]
        assert body["size"] == 20
        assert body["query"]["knn"]["embedding_vector"]["k"] == 50


class TestHybridSearch:
    @pytest.mark.asyncio
    async def test_returns_hybrid_mode_on_success(self, mock_client):
        result = await hybrid_search(mock_client, "shampoo", SAMPLE_VECTOR, INDEX)
        assert result.mode == SearchMode.hybrid

    @pytest.mark.asyncio
    async def test_returns_hits(self, mock_client):
        result = await hybrid_search(mock_client, "shampoo", SAMPLE_VECTOR, INDEX)
        assert result.total == 1
        assert len(result.hits) == 1

    @pytest.mark.asyncio
    async def test_falls_back_to_lexical_on_exception(self, mock_client):
        """
        When the hybrid query raises (e.g. pipeline not registered),
        hybrid_search() should fall back to lexical and still return results.
        """
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Hybrid pipeline not found")
            return _make_os_response(hits=[_make_hit()], total=1)

        mock_client.search.side_effect = side_effect
        result = await hybrid_search(mock_client, "shampoo", SAMPLE_VECTOR, INDEX)
        # Fallback returns lexical mode
        assert result.mode == SearchMode.lexical
        assert result.total == 1

    @pytest.mark.asyncio
    async def test_with_all_params(self, mock_client):
        result = await hybrid_search(
            mock_client,
            query="moisturizer",
            query_vector=SAMPLE_VECTOR,
            index_name=INDEX,
            category="All Beauty",
            min_price=5.0,
            max_price=40.0,
            min_rating=4.0,
            boost_terms=["vegan", "cruelty-free"],
            size=15,
            from_=0,
        )
        assert result.mode == SearchMode.hybrid

    @pytest.mark.asyncio
    async def test_empty_results(self, empty_client):
        result = await hybrid_search(empty_client, "unknown", SAMPLE_VECTOR, INDEX)
        assert result.total == 0
        assert result.hits == []

    @pytest.mark.asyncio
    async def test_pipeline_param_passed(self, mock_client):
        await hybrid_search(mock_client, "serum", SAMPLE_VECTOR, INDEX)
        call_kwargs = mock_client.search.call_args
        # params should include search_pipeline
        params = call_kwargs[1].get("params", {})
        assert "search_pipeline" in params


# ── ProductHit model tests ────────────────────────────────────────────────────

class TestProductHit:
    def test_from_hit_basic(self):
        raw = _make_hit("ASIN001", score=2.5, title="Great Shampoo", price=12.99)
        hit = ProductHit.from_hit(raw)
        assert hit.id == "ASIN001"
        assert hit.score == 2.5
        assert hit.title == "Great Shampoo"
        assert hit.price == 12.99

    def test_from_hit_missing_optional_fields(self):
        raw = {"_id": "X", "_score": 0.5, "_source": {"title": "Minimal"}}
        hit = ProductHit.from_hit(raw)
        assert hit.id == "X"
        assert hit.price is None
        assert hit.average_rating is None
        assert hit.description == ""

    def test_from_hit_null_score(self):
        raw = {"_id": "Y", "_score": None, "_source": {}}
        hit = ProductHit.from_hit(raw)
        assert hit.score == 0.0
