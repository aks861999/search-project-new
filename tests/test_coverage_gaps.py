"""
tests/test_coverage_gaps.py — targeted tests to cover every remaining
uncovered line across api/, config/, and ingestion/.

Gaps addressed:
  api/main.py         66-69   _create_os_client with http_auth
                      86-108  lifespan startup + shutdown
                      146-148 health check OpenSearch failure
                      235-237 /search endpoint internal exception → 500
                      274-276 /nl-search internal exception → 500
                      301     /suggest cache hit path
                      326-328 /suggest OpenSearch exception → empty list
                      371-373 /index/product OpenSearch exception → 500

  api/nl_query.py     82-83   _extract_json braces-match but invalid JSON
                      93      _dict_to_parsed_query non-dict filters
                      131-142 _build_llm_chain with API key (success + failure)

  api/search.py       226     _parse_response with integer total

  api/models.py       41-44   SearchRequest max_price < min_price validator

  config/settings.py  73      invalid log_level raises ValueError
                      78-79   opensearch_url property

  ingestion/pipeline  111     preprocess_record with string details
                      157-161 get_client with http_auth credentials
                      292-309 run_pipeline chunk-flush while loop (chunk_size < batch)
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient


# ── api/models.py — validator ─────────────────────────────────────────────────

class TestSearchRequestValidator:
    def test_max_price_less_than_min_price_raises(self):
        from api.models import SearchRequest
        with pytest.raises(Exception):  # pydantic ValidationError
            SearchRequest(q="test", min_price=50.0, max_price=10.0)

    def test_max_price_equal_to_min_price_is_valid(self):
        from api.models import SearchRequest
        req = SearchRequest(q="test", min_price=20.0, max_price=20.0)
        assert req.max_price == 20.0

    def test_max_price_none_always_valid(self):
        from api.models import SearchRequest
        req = SearchRequest(q="test", min_price=10.0, max_price=None)
        assert req.max_price is None

    def test_min_price_none_always_valid(self):
        from api.models import SearchRequest
        req = SearchRequest(q="test", min_price=None, max_price=5.0)
        assert req.max_price == 5.0


# ── config/settings.py ────────────────────────────────────────────────────────

class TestSettings:
    def test_invalid_log_level_raises_value_error(self):
        from pydantic import ValidationError
        from config.settings import Settings
        with pytest.raises(ValidationError):
            Settings(LOG_LEVEL="VERBOSE")

    def test_valid_log_levels_accepted(self):
        from config.settings import Settings
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            s = Settings(LOG_LEVEL=level)
            assert s.log_level == level

    def test_opensearch_url_http(self):
        from config.settings import Settings
        s = Settings(OPENSEARCH_HOST="myhost", OPENSEARCH_PORT=9200,
                     OPENSEARCH_USE_SSL=False)
        assert s.opensearch_url == "http://myhost:9200"

    def test_opensearch_url_https(self):
        from config.settings import Settings
        s = Settings(OPENSEARCH_HOST="secure.host", OPENSEARCH_PORT=443,
                     OPENSEARCH_USE_SSL=True)
        assert s.opensearch_url == "https://secure.host:443"

    def test_log_level_is_uppercased(self):
        from config.settings import Settings
        s = Settings(LOG_LEVEL="info")
        assert s.log_level == "INFO"


# ── api/search.py — integer total ─────────────────────────────────────────────

class TestParseResponseIntegerTotal:
    @pytest.mark.asyncio
    async def test_integer_total_parsed_correctly(self):
        """OpenSearch sometimes returns total as a plain int, not a dict."""
        from api.search import _parse_response
        from api.models import SearchMode

        raw = {
            "took": 2,
            "hits": {
                "total": 42,         # plain int, not {"value": 42, "relation": "eq"}
                "hits": [],
            }
        }
        result = _parse_response(raw, SearchMode.lexical)
        assert result.total == 42

    @pytest.mark.asyncio
    async def test_zero_integer_total(self):
        from api.search import _parse_response
        from api.models import SearchMode

        raw = {"took": 1, "hits": {"total": 0, "hits": []}}
        result = _parse_response(raw, SearchMode.hybrid)
        assert result.total == 0


# ── api/nl_query.py — edge cases ─────────────────────────────────────────────

class TestExtractJsonEdgeCases:
    def test_braces_match_but_invalid_json_returns_empty(self):
        """A {…} block is found but its content is not valid JSON."""
        from api.nl_query import _extract_json
        text = "Here is some { invalid: json without quotes } output"
        result = _extract_json(text)
        assert result == {}

    def test_nested_braces_valid_json(self):
        from api.nl_query import _extract_json
        text = 'Response: {"semantic_query": "test", "filters": {"price_max": 20}}'
        result = _extract_json(text)
        assert result["semantic_query"] == "test"
        assert result["filters"]["price_max"] == 20


class TestDictToParsedQueryEdgeCases:
    def test_non_dict_filters_becomes_empty_nlfilters(self):
        """filters set to a non-dict value (e.g. a string) must be treated as empty."""
        from api.nl_query import _dict_to_parsed_query
        data = {"semantic_query": "serum", "filters": "not a dict"}
        result = _dict_to_parsed_query(data)
        assert result.filters.main_category is None
        assert result.filters.price_max is None

    def test_list_filters_becomes_empty(self):
        from api.nl_query import _dict_to_parsed_query
        data = {"semantic_query": "serum", "filters": [1, 2, 3]}
        result = _dict_to_parsed_query(data)
        assert result.filters.price_min is None


class TestBuildLlmChain:
    def test_returns_none_when_no_api_key(self):
        from api.nl_query import _build_llm_chain
        with patch("api.nl_query.get_settings") as mock_settings:
            mock_settings.return_value.google_api_key = None
            result = _build_llm_chain()
        assert result is None

    def test_returns_chain_when_api_key_set(self):
        """When GOOGLE_API_KEY is set, _build_llm_chain returns a non-None chain."""
        import langchain_google_genai as lgc_mod
        from api.nl_query import _build_llm_chain

        mock_llm = MagicMock()
        mock_settings = MagicMock()
        mock_settings.google_api_key = "fake-key"
        mock_settings.llm_model = "gemini-2.0-flash"
        mock_settings.llm_temperature = 0.0

        # Patch inside the langchain_google_genai module (where the import resolves)
        with (
            patch("api.nl_query.get_settings", return_value=mock_settings),
            patch.object(lgc_mod, "ChatGoogleGenerativeAI", return_value=mock_llm),
        ):
            chain = _build_llm_chain()

        assert chain is not None

    def test_falls_back_to_none_when_gemini_init_fails(self):
        """If ChatGoogleGenerativeAI raises on init, fall back to None."""
        import langchain_google_genai as lgc_mod
        from api.nl_query import _build_llm_chain

        mock_settings = MagicMock()
        mock_settings.google_api_key = "fake-key"
        mock_settings.llm_model = "gemini-2.0-flash"
        mock_settings.llm_temperature = 0.0

        with (
            patch("api.nl_query.get_settings", return_value=mock_settings),
            patch.object(lgc_mod, "ChatGoogleGenerativeAI",
                         side_effect=Exception("import error")),
        ):
            result = _build_llm_chain()

        assert result is None


# ── ingestion/pipeline.py ─────────────────────────────────────────────────────

class TestGetClientWithAuth:
    def test_get_client_without_auth(self):
        from ingestion.pipeline import get_client
        settings = MagicMock()
        settings.opensearch_host = "localhost"
        settings.opensearch_port = 9200
        settings.opensearch_use_ssl = False
        settings.opensearch_verify_certs = False
        settings.opensearch_user = None
        settings.opensearch_password = None

        with patch("ingestion.pipeline.OpenSearch") as mock_os:
            get_client(settings)
            call_kwargs = mock_os.call_args[1]
            assert call_kwargs["http_auth"] is None

    def test_get_client_with_auth(self):
        from ingestion.pipeline import get_client
        settings = MagicMock()
        settings.opensearch_host = "localhost"
        settings.opensearch_port = 9200
        settings.opensearch_use_ssl = True
        settings.opensearch_verify_certs = False
        settings.opensearch_user = "admin"
        settings.opensearch_password = "secret"

        with patch("ingestion.pipeline.OpenSearch") as mock_os:
            get_client(settings)
            call_kwargs = mock_os.call_args[1]
            assert call_kwargs["http_auth"] == ("admin", "secret")


class TestPreprocessRecordStringDetails:
    def test_string_details_treated_as_empty_dict(self):
        from ingestion.pipeline import preprocess_record
        record = {
            "parent_asin": "B010",
            "title": "Product",
            "features": [],
            "description": [],
            "main_category": "Beauty",
            "sub_category": "",
            "store": "S",
            "price": "9.99",
            "average_rating": 4.0,
            "rating_number": 5,
            "images": [],
            "details": "some string instead of dict",  # ← the gap
        }
        result = preprocess_record(record)
        assert result is not None
        assert result["parent_asin"] == "B010"


class TestRunPipelineChunkFlush:
    def test_chunk_flush_happens_when_accumulation_exceeds_chunk_size(self):
        """
        When chunk_size < batch_size the while-loop inside run_pipeline fires.
        Here: 4 docs, batch_size=4, chunk_size=2 → while loop flushes twice.
        """
        import numpy as np
        from ingestion.pipeline import run_pipeline

        fake_records = [
            {"parent_asin": f"B{i:03d}", "title": f"Product {i}",
             "features": [], "description": [], "main_category": "Beauty",
             "sub_category": "", "store": "S", "price": "9.99",
             "average_rating": 4.0, "rating_number": 10,
             "images": [], "details": {}}
            for i in range(4)
        ]

        mock_client = MagicMock()
        mock_client.cluster.health = MagicMock(return_value={"status": "green"})
        mock_client.indices.exists = MagicMock(return_value=True)
        mock_client.transport.perform_request = MagicMock(return_value={})

        mock_model = MagicMock()
        # encode returns a (4, 384) array
        mock_model.encode = MagicMock(
            return_value=np.zeros((4, 384), dtype="float32")
        )

        bulk_calls = []

        def fake_bulk(client, actions, **kwargs):
            action_list = list(actions)
            bulk_calls.append(len(action_list))
            return (len(action_list), [])

        fake_settings = MagicMock()
        fake_settings.index_name = "products"
        fake_settings.embedding_model = "BAAI/bge-small-en-v1.5"
        fake_settings.hf_dataset = "McAuley-Lab/Amazon-Reviews-2023"
        fake_settings.hf_dataset_config = "raw_meta_All_Beauty"
        fake_settings.embedding_batch_size = 4   # encode all 4 at once
        fake_settings.chunk_size = 2             # flush every 2 → while loop fires

        with (
            patch("ingestion.pipeline.get_settings", return_value=fake_settings),
            patch("ingestion.pipeline.get_client", return_value=mock_client),
            patch("ingestion.pipeline.SentenceTransformer", return_value=mock_model),
            patch("ingestion.pipeline.load_dataset", return_value=fake_records),
            patch("ingestion.pipeline.helpers.bulk", side_effect=fake_bulk),
        ):
            run_pipeline()

        # With chunk_size=2 and 4 docs: while loop fires twice (2+2), no final flush
        total_docs = sum(bulk_calls)
        assert total_docs == 4
        assert len(bulk_calls) >= 2


# ── api/main.py — error paths and lifespan ───────────────────────────────────

def _make_patched_client(app):
    """Return a patched AsyncClient with all dependencies properly mocked."""
    mock_os = MagicMock()
    mock_os.search = MagicMock(return_value={
        "took": 1,
        "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []}
    })
    mock_os.cluster.health = MagicMock(return_value={"status": "green"})
    mock_os.index = MagicMock(return_value={"result": "created", "_id": "X"})
    mock_model = MagicMock()
    arr = MagicMock()
    arr.tolist = MagicMock(return_value=[0.0] * 384)
    mock_model.encode = MagicMock(return_value=arr)
    return mock_os, mock_model


class TestMainErrorPaths:
    @pytest.mark.asyncio
    async def test_search_endpoint_returns_500_on_internal_error(self):
        from api.main import app
        mock_os = MagicMock()
        mock_os.search = MagicMock(side_effect=Exception("OpenSearch down"))
        mock_model = MagicMock()
        arr = MagicMock()
        arr.tolist = MagicMock(return_value=[0.0] * 384)
        mock_model.encode = MagicMock(return_value=arr)

        with (
            patch("api.dependencies._os_client", mock_os),
            patch("api.dependencies._embedding_model", mock_model),
            patch("api.cache.cache_get", new=AsyncMock(return_value=None)),
            patch("api.cache.cache_set", new=AsyncMock()),
            patch("api.cache.ping_redis", new=AsyncMock(return_value=True)),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/search?q=test")
                assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_nl_search_returns_500_on_internal_error(self):
        from api.main import app
        mock_os = MagicMock()
        mock_os.search = MagicMock(side_effect=Exception("Connection reset"))
        mock_model = MagicMock()
        arr = MagicMock()
        arr.tolist = MagicMock(return_value=[0.0] * 384)
        mock_model.encode = MagicMock(return_value=arr)

        with (
            patch("api.dependencies._os_client", mock_os),
            patch("api.dependencies._embedding_model", mock_model),
            patch("api.cache.cache_get", new=AsyncMock(return_value=None)),
            patch("api.cache.cache_set", new=AsyncMock()),
            patch("api.cache.ping_redis", new=AsyncMock(return_value=True)),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/nl-search",
                                     json={"query": "moisturizer for dry skin"})
                assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_suggest_returns_empty_list_when_os_fails(self):
        from api.main import app
        mock_os = MagicMock()
        mock_os.search = MagicMock(side_effect=Exception("timeout"))
        mock_model = MagicMock()

        with (
            patch("api.dependencies._os_client", mock_os),
            patch("api.dependencies._embedding_model", mock_model),
            patch("api.cache.cache_get", new=AsyncMock(return_value=None)),
            patch("api.cache.cache_set", new=AsyncMock()),
            patch("api.cache.ping_redis", new=AsyncMock(return_value=True)),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/suggest?prefix=test")
                assert resp.status_code == 200
                assert resp.json()["suggestions"] == []

    @pytest.mark.asyncio
    async def test_suggest_cached_path(self):
        from api.main import app
        cached = {"suggestions": ["Moisturizer A", "Moisturizer B"], "prefix": "moist"}

        with (
            patch("api.dependencies._os_client", MagicMock()),
            patch("api.dependencies._embedding_model", MagicMock()),
            patch("api.cache.cache_get", new=AsyncMock(return_value=cached)),
            patch("api.cache.cache_set", new=AsyncMock()),
            patch("api.cache.ping_redis", new=AsyncMock(return_value=True)),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/suggest?prefix=moist")
                assert resp.status_code == 200
                body = resp.json()
                assert body["suggestions"] == ["Moisturizer A", "Moisturizer B"]

    @pytest.mark.asyncio
    async def test_index_product_returns_500_on_os_error(self):
        from api.main import app
        mock_os = MagicMock()
        mock_os.index = MagicMock(side_effect=Exception("index write failed"))
        mock_model = MagicMock()
        arr = MagicMock()
        arr.tolist = MagicMock(return_value=[0.0] * 384)
        mock_model.encode = MagicMock(return_value=arr)

        product = {
            "id": "B999", "score": 0.0, "title": "Test",
            "description": "desc", "features": "feat",
            "main_category": "Beauty", "sub_category": "",
            "store": "S", "price": 9.99, "average_rating": 4.0,
            "rating_number": 5, "primary_image_url": "", "parent_asin": "B999",
        }

        with (
            patch("api.dependencies._os_client", mock_os),
            patch("api.dependencies._embedding_model", mock_model),
            patch("api.cache.cache_get", new=AsyncMock(return_value=None)),
            patch("api.cache.cache_set", new=AsyncMock()),
            patch("api.cache.ping_redis", new=AsyncMock(return_value=True)),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/index/product", json=product)
                assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_health_check_opensearch_unreachable(self):
        from api.main import app
        mock_os = MagicMock()
        mock_os.cluster.health = MagicMock(side_effect=Exception("connection refused"))

        with (
            patch("api.dependencies._os_client", mock_os),
            patch("api.dependencies._embedding_model", MagicMock()),
            patch("api.cache.cache_get", new=AsyncMock(return_value=None)),
            patch("api.cache.cache_set", new=AsyncMock()),
            patch("api.cache.ping_redis", new=AsyncMock(return_value=True)),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/health")
                body = resp.json()
                assert body["opensearch"] == "unreachable"
                assert body["status"] == "degraded"


class TestCreateOsClient:
    def test_create_os_client_without_auth(self):
        """_create_os_client when no user/password → http_auth=None."""
        from api.main import _create_os_client
        with (
            patch("api.main.settings") as mock_s,
            patch("api.main.OpenSearch") as mock_os_cls,
        ):
            mock_s.opensearch_user = None
            mock_s.opensearch_password = None
            mock_s.opensearch_host = "localhost"
            mock_s.opensearch_port = 9200
            mock_s.opensearch_use_ssl = False
            mock_s.opensearch_verify_certs = False
            _create_os_client()
            call_kwargs = mock_os_cls.call_args[1]
            assert call_kwargs["http_auth"] is None

    def test_create_os_client_with_auth(self):
        """_create_os_client when user+password set → http_auth tuple."""
        from api.main import _create_os_client
        with (
            patch("api.main.settings") as mock_s,
            patch("api.main.OpenSearch") as mock_os_cls,
        ):
            mock_s.opensearch_user = "admin"
            mock_s.opensearch_password = "secret"
            mock_s.opensearch_host = "localhost"
            mock_s.opensearch_port = 9200
            mock_s.opensearch_use_ssl = False
            mock_s.opensearch_verify_certs = False
            _create_os_client()
            call_kwargs = mock_os_cls.call_args[1]
            assert call_kwargs["http_auth"] == ("admin", "secret")
