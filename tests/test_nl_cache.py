"""
Tests for api/nl_query.py and api/cache.py.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.cache import make_cache_key
from api.models import ParsedNLQuery, SearchMode
from api.nl_query import _dict_to_parsed_query, _extract_json, _stub_parse


# ── JSON extraction tests ─────────────────────────────────────────────────────

class TestExtractJson:
    def test_clean_json(self):
        text = '{"semantic_query": "moisturizer", "filters": {}, "boost_terms": []}'
        result = _extract_json(text)
        assert result["semantic_query"] == "moisturizer"

    def test_markdown_fenced_json(self):
        text = '```json\n{"semantic_query": "serum", "boost_terms": []}\n```'
        result = _extract_json(text)
        assert result["semantic_query"] == "serum"

    def test_json_with_surrounding_text(self):
        text = 'Here is the JSON:\n{"semantic_query": "shampoo"}\nEnd of response.'
        result = _extract_json(text)
        assert result["semantic_query"] == "shampoo"

    def test_invalid_json_returns_empty_dict(self):
        result = _extract_json("Not JSON at all.")
        assert result == {}

    def test_empty_string_returns_empty_dict(self):
        result = _extract_json("")
        assert result == {}


# ── Dict-to-ParsedNLQuery tests ───────────────────────────────────────────────

class TestDictToParsedQuery:
    def test_full_dict(self):
        data = {
            "semantic_query": "organic hair care",
            "filters": {
                "main_category": "All Beauty",
                "price_min": 5.0,
                "price_max": 30.0,
                "rating_min": 4.0,
            },
            "boost_terms": ["organic", "natural"],
            "search_mode": "hybrid",
        }
        parsed = _dict_to_parsed_query(data)
        assert parsed.semantic_query == "organic hair care"
        assert parsed.filters.main_category == "All Beauty"
        assert parsed.filters.price_min == 5.0
        assert parsed.filters.price_max == 30.0
        assert parsed.filters.rating_min == 4.0
        assert parsed.boost_terms == ["organic", "natural"]
        assert parsed.search_mode == SearchMode.hybrid

    def test_empty_dict_gives_defaults(self):
        parsed = _dict_to_parsed_query({})
        assert parsed.semantic_query == ""
        assert parsed.filters.main_category is None
        assert parsed.boost_terms == []
        assert parsed.search_mode == SearchMode.hybrid

    def test_invalid_search_mode_defaults_to_hybrid(self):
        parsed = _dict_to_parsed_query({"search_mode": "nonexistent"})
        assert parsed.search_mode == SearchMode.hybrid

    def test_null_filters_handled(self):
        parsed = _dict_to_parsed_query({"filters": None})
        assert parsed.filters.main_category is None


# ── Stub parser tests ─────────────────────────────────────────────────────────

class TestStubParse:
    def test_price_max_extracted(self):
        result = _stub_parse("moisturizer under $25")
        assert result.filters.price_max == 25.0

    def test_price_min_extracted(self):
        result = _stub_parse("serum over $10")
        assert result.filters.price_min == 10.0

    def test_price_range_extracted(self):
        result = _stub_parse("shampoo between $5 and $20")
        assert result.filters.price_min == 5.0
        assert result.filters.price_max == 20.0

    def test_high_rating_triggers_filter(self):
        result = _stub_parse("top rated face cream")
        assert result.filters.rating_min == 4.0

    def test_best_triggers_rating_filter(self):
        result = _stub_parse("best anti-aging serum")
        assert result.filters.rating_min == 4.0

    def test_no_rating_filter_when_not_mentioned(self):
        result = _stub_parse("blue shampoo bottle")
        assert result.filters.rating_min is None

    def test_boost_terms_extracted(self):
        result = _stub_parse("organic sulfate-free shampoo")
        assert "organic" in result.boost_terms
        assert any("sulfate" in t for t in result.boost_terms)

    def test_returns_parsed_nl_query_type(self):
        result = _stub_parse("any query")
        assert isinstance(result, ParsedNLQuery)

    def test_search_mode_is_hybrid(self):
        result = _stub_parse("cheap moisturizer")
        assert result.search_mode == SearchMode.hybrid

    def test_semantic_query_not_empty_for_valid_input(self):
        result = _stub_parse("shampoo for dry hair")
        assert len(result.semantic_query) > 0


# ── parse_nl_query (chain paths) ──────────────────────────────────────────────

class TestParseNLQueryChainPaths:
    @pytest.mark.asyncio
    async def test_uses_stub_when_no_chain(self):
        from api.nl_query import parse_nl_query
        with patch("api.nl_query.get_nl_chain", return_value=None):
            result = await parse_nl_query("cheap shampoo under $15")
        assert result.filters.price_max == 15.0

    @pytest.mark.asyncio
    async def test_uses_llm_chain_when_available(self):
        from api.nl_query import parse_nl_query

        llm_response = json.dumps({
            "semantic_query": "shampoo",
            "filters": {"main_category": "All Beauty", "price_max": 20.0,
                        "price_min": None, "rating_min": None},
            "boost_terms": ["organic"],
            "search_mode": "hybrid",
        })
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(return_value=llm_response)

        with patch("api.nl_query.get_nl_chain", return_value=mock_chain):
            result = await parse_nl_query("organic shampoo under $20")

        assert result.semantic_query == "shampoo"
        assert result.filters.price_max == 20.0
        assert "organic" in result.boost_terms

    @pytest.mark.asyncio
    async def test_falls_back_to_stub_when_chain_raises(self):
        from api.nl_query import parse_nl_query

        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(side_effect=Exception("API timeout"))

        with patch("api.nl_query.get_nl_chain", return_value=mock_chain):
            result = await parse_nl_query("top rated moisturizer")

        # Stub parser handles it — rating_min set from "top rated"
        assert result.filters.rating_min == 4.0


# ── nl_search end-to-end ──────────────────────────────────────────────────────

class TestNLSearch:
    @pytest.mark.asyncio
    async def test_nl_search_calls_hybrid_search(self):
        import numpy as np
        from api.nl_query import nl_search

        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.encode = MagicMock(return_value=np.zeros(384, dtype="float32"))

        mock_search_result = MagicMock()
        mock_search_result.total = 2
        mock_search_result.hits = []
        mock_search_result.took_ms = 5

        with (
            patch("api.nl_query.get_nl_chain", return_value=None),
            patch("api.nl_query.hybrid_search", new=AsyncMock(return_value=mock_search_result)),
        ):
            result = await nl_search(
                client=mock_client,
                model=mock_model,
                query="organic shampoo",
                index_name="products",
                size=5,
            )

        assert result.total == 2
        assert result.took_ms == 5

    @pytest.mark.asyncio
    async def test_nl_search_passes_filters_to_hybrid(self):
        import numpy as np
        from api.nl_query import nl_search

        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.encode = MagicMock(return_value=np.zeros(384, dtype="float32"))

        mock_result = MagicMock()
        mock_result.total = 1
        mock_result.hits = []
        mock_result.took_ms = 3

        captured = {}

        async def mock_hybrid(**kwargs):
            captured.update(kwargs)
            return mock_result

        with (
            patch("api.nl_query.get_nl_chain", return_value=None),
            patch("api.nl_query.hybrid_search", new=mock_hybrid),
        ):
            await nl_search(
                client=mock_client,
                model=mock_model,
                query="top rated serum under $30",
                index_name="products",
                size=10,
            )

        # stub parser should have extracted price_max=30 and rating_min=4.0
        assert captured["max_price"] == 30.0
        assert captured["min_rating"] == 4.0

    @pytest.mark.asyncio
    async def test_nl_search_uses_full_query_as_fallback(self):
        """When semantic_query is empty the original query is used."""
        import numpy as np
        from api.nl_query import nl_search

        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.encode = MagicMock(return_value=np.zeros(384, dtype="float32"))

        mock_result = MagicMock()
        mock_result.total = 0
        mock_result.hits = []
        mock_result.took_ms = 1

        from api.models import NLFilters, ParsedNLQuery, SearchMode

        empty_parsed = ParsedNLQuery(
            semantic_query="",
            filters=NLFilters(),
            boost_terms=[],
            search_mode=SearchMode.hybrid,
        )

        with (
            patch("api.nl_query.parse_nl_query", new=AsyncMock(return_value=empty_parsed)),
            patch("api.nl_query.hybrid_search", new=AsyncMock(return_value=mock_result)),
        ):
            result = await nl_search(
                client=mock_client,
                model=mock_model,
                query="original query",
                index_name="products",
                size=5,
            )

        assert result.total == 0



class TestMakeCacheKey:
    def test_key_has_prefix(self):
        key = make_cache_key("search", q="shampoo")
        assert key.startswith("search:")

    def test_same_params_produce_same_key(self):
        key1 = make_cache_key("search", q="shampoo", size=10, mode="hybrid")
        key2 = make_cache_key("search", q="shampoo", size=10, mode="hybrid")
        assert key1 == key2

    def test_different_params_produce_different_keys(self):
        key1 = make_cache_key("search", q="shampoo")
        key2 = make_cache_key("search", q="conditioner")
        assert key1 != key2

    def test_none_values_excluded(self):
        key1 = make_cache_key("search", q="shampoo", category=None)
        key2 = make_cache_key("search", q="shampoo")
        assert key1 == key2

    def test_param_order_does_not_matter(self):
        key1 = make_cache_key("search", q="shampoo", size=10, mode="hybrid")
        key2 = make_cache_key("search", mode="hybrid", size=10, q="shampoo")
        assert key1 == key2

    def test_different_prefixes_produce_different_keys(self):
        key1 = make_cache_key("search", q="shampoo")
        key2 = make_cache_key("nl-search", q="shampoo")
        assert key1 != key2
