"""
Tests for api/cache.py — async Redis caching utilities.

All Redis I/O is mocked; no live Redis instance is required.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.cache import cache_get, cache_set, close_redis, make_cache_key, ping_redis


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_pool():
    """Reset the module-level Redis pool before each test."""
    import api.cache as cache_mod
    cache_mod._redis_pool = None
    yield
    cache_mod._redis_pool = None


def _make_redis_mock(get_return=None, ping_return=True):
    r = AsyncMock()
    r.get = AsyncMock(return_value=get_return)
    r.setex = AsyncMock(return_value=True)
    r.ping = AsyncMock(return_value=ping_return)
    r.aclose = AsyncMock()
    return r


# ── get_redis ─────────────────────────────────────────────────────────────────

class TestGetRedis:
    @pytest.mark.asyncio
    async def test_creates_pool_on_first_call(self):
        import api.cache as cache_mod

        mock_pool = AsyncMock()
        with patch("api.cache.aioredis") as mock_aioredis:
            mock_aioredis.from_url = AsyncMock(return_value=mock_pool)
            result = await cache_mod.get_redis()

        assert result is mock_pool
        mock_aioredis.from_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_same_pool_on_second_call(self):
        import api.cache as cache_mod

        mock_pool = AsyncMock()
        with patch("api.cache.aioredis") as mock_aioredis:
            mock_aioredis.from_url = AsyncMock(return_value=mock_pool)
            r1 = await cache_mod.get_redis()
            r2 = await cache_mod.get_redis()

        # from_url called only once despite two get_redis calls
        assert mock_aioredis.from_url.call_count == 1
        assert r1 is r2

    @pytest.mark.asyncio
    async def test_pool_reused_when_already_set(self):
        import api.cache as cache_mod

        existing = AsyncMock(name="existing_pool")
        cache_mod._redis_pool = existing
        result = await cache_mod.get_redis()
        assert result is existing



class TestCacheGet:
    @pytest.mark.asyncio
    async def test_returns_none_on_miss(self):
        mock_redis = _make_redis_mock(get_return=None)
        with patch("api.cache.get_redis", new=AsyncMock(return_value=mock_redis)):
            result = await cache_get("missing:key")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_deserialized_value_on_hit(self):
        payload = {"total": 5, "hits": []}
        mock_redis = _make_redis_mock(get_return=json.dumps(payload))
        with patch("api.cache.get_redis", new=AsyncMock(return_value=mock_redis)):
            result = await cache_get("search:abc")
        assert result == payload

    @pytest.mark.asyncio
    async def test_returns_none_on_redis_error(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=Exception("connection refused"))
        with patch("api.cache.get_redis", new=AsyncMock(return_value=mock_redis)):
            result = await cache_get("search:xyz")
        assert result is None

    @pytest.mark.asyncio
    async def test_handles_nested_dict(self):
        payload = {"a": {"b": [1, 2, 3]}, "c": None}
        mock_redis = _make_redis_mock(get_return=json.dumps(payload))
        with patch("api.cache.get_redis", new=AsyncMock(return_value=mock_redis)):
            result = await cache_get("key")
        assert result == payload

    @pytest.mark.asyncio
    async def test_calls_redis_get_with_correct_key(self):
        mock_redis = _make_redis_mock(get_return=None)
        with patch("api.cache.get_redis", new=AsyncMock(return_value=mock_redis)):
            await cache_get("my:exact:key")
        mock_redis.get.assert_called_once_with("my:exact:key")


# ── cache_set ─────────────────────────────────────────────────────────────────

class TestCacheSet:
    @pytest.mark.asyncio
    async def test_serialises_and_stores_value(self):
        mock_redis = _make_redis_mock()
        payload = {"hits": [{"id": "B001"}]}
        with patch("api.cache.get_redis", new=AsyncMock(return_value=mock_redis)):
            await cache_set("search:key", payload)
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args[0]
        assert call_args[0] == "search:key"
        assert call_args[1] == 300  # default TTL
        assert json.loads(call_args[2]) == payload

    @pytest.mark.asyncio
    async def test_respects_custom_ttl(self):
        mock_redis = _make_redis_mock()
        with patch("api.cache.get_redis", new=AsyncMock(return_value=mock_redis)):
            await cache_set("suggest:key", ["a", "b"], ttl=60)
        call_args = mock_redis.setex.call_args[0]
        assert call_args[1] == 60

    @pytest.mark.asyncio
    async def test_does_not_raise_on_redis_error(self):
        mock_redis = AsyncMock()
        mock_redis.setex = AsyncMock(side_effect=Exception("timeout"))
        with patch("api.cache.get_redis", new=AsyncMock(return_value=mock_redis)):
            await cache_set("key", {"data": 1})  # must NOT raise

    @pytest.mark.asyncio
    async def test_handles_none_value(self):
        mock_redis = _make_redis_mock()
        with patch("api.cache.get_redis", new=AsyncMock(return_value=mock_redis)):
            await cache_set("key", None)
        mock_redis.setex.assert_called_once()


# ── ping_redis ────────────────────────────────────────────────────────────────

class TestPingRedis:
    @pytest.mark.asyncio
    async def test_returns_true_when_redis_responds(self):
        mock_redis = _make_redis_mock(ping_return=True)
        with patch("api.cache.get_redis", new=AsyncMock(return_value=mock_redis)):
            result = await ping_redis()
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_redis_unreachable(self):
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=Exception("connection refused"))
        with patch("api.cache.get_redis", new=AsyncMock(return_value=mock_redis)):
            result = await ping_redis()
        assert result is False


# ── close_redis ───────────────────────────────────────────────────────────────

class TestCloseRedis:
    @pytest.mark.asyncio
    async def test_closes_pool_and_sets_none(self):
        import api.cache as cache_mod

        mock_redis = _make_redis_mock()
        cache_mod._redis_pool = mock_redis

        await close_redis()

        mock_redis.aclose.assert_called_once()
        assert cache_mod._redis_pool is None

    @pytest.mark.asyncio
    async def test_noop_when_pool_is_none(self):
        import api.cache as cache_mod
        cache_mod._redis_pool = None
        await close_redis()  # must not raise


# ── make_cache_key ────────────────────────────────────────────────────────────

class TestMakeCacheKeyExtended:
    def test_none_values_excluded_from_hash(self):
        k1 = make_cache_key("search", q="shampoo", category=None)
        k2 = make_cache_key("search", q="shampoo")
        assert k1 == k2

    def test_list_values_serialized_correctly(self):
        k = make_cache_key("search", boost_terms=["organic", "vegan"])
        assert k.startswith("search:")
        assert len(k) == len("search:") + 32  # md5 hex digest

    def test_float_and_int_produce_stable_key(self):
        k1 = make_cache_key("search", min_price=5.0, max_price=30.0)
        k2 = make_cache_key("search", min_price=5.0, max_price=30.0)
        assert k1 == k2

    def test_prefix_kwarg_does_not_collide(self):
        """Critical: 'prefix' must be passable as a kwarg without TypeError."""
        k = make_cache_key("suggest", prefix="moist", size=5)
        assert k.startswith("suggest:")

    def test_key_length_is_fixed(self):
        k = make_cache_key("nl-search", query="moisturizer", size=10)
        expected_len = len("nl-search:") + 32
        assert len(k) == expected_len
