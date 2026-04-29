"""
tests/test_100pct.py — Three targeted tests to close the final 6 uncovered lines.

  api/main.py:469-470   POST /mlops/experiments → 500 on unexpected exception
  api/main.py:591       _log_search_event early return when os_client is None
  mlops/experiments/ab_framework.py:133   start_experiment raises when not found
  mlops/experiments/ab_framework.py:208   get_config_for_session returns None
                                           when experiment disappears after assign
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.main import app


# ── Shared fixture ────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client():
    with (
        patch("api.dependencies._os_client", MagicMock()),
        patch("api.dependencies._embedding_model", MagicMock()),
        patch("api.cache.cache_get", new=AsyncMock(return_value=None)),
        patch("api.cache.cache_set", new=AsyncMock()),
        patch("api.cache.ping_redis", new=AsyncMock(return_value=True)),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


# ── api/main.py lines 469-470: POST /mlops/experiments → 500 ─────────────────

class TestCreateExperimentGenericError:
    @pytest.mark.asyncio
    async def test_create_experiment_returns_500_on_unexpected_exception(
        self, client
    ):
        """
        When _build_ab_framework raises an unexpected exception (not KeyError/ValueError)
        the endpoint should return 500, covering lines 469-470.
        """
        with patch(
            "api.main._build_ab_framework",
            side_effect=RuntimeError("Redis connection refused"),
        ):
            resp = await client.post(
                "/mlops/experiments",
                json={
                    "experiment_id": "crash-test",
                    "description": "Should 500",
                    "variants": [
                        {"variant_id": "a", "weight": 0.5, "config": {}},
                        {"variant_id": "b", "weight": 0.5, "config": {}},
                    ],
                },
            )
        assert resp.status_code == 500
        assert "Redis connection refused" in resp.json()["detail"]


# ── api/main.py line 591: _log_search_event early return ─────────────────────

class TestLogSearchEventEarlyReturn:
    @pytest.mark.asyncio
    async def test_search_event_skipped_when_os_client_none(self):
        """
        _log_search_event should return immediately when _deps._os_client is None,
        covering line 591 without crashing the request.
        """
        import api.dependencies as _deps

        mock_os = MagicMock()
        mock_os.search = MagicMock(return_value={
            "took": 2,
            "hits": {"total": {"value": 1, "relation": "eq"}, "hits": [
                {"_id": "B001", "_score": 1.0, "_source": {
                    "title": "Test", "parent_asin": "B001",
                    "description": "", "features": "",
                    "main_category": "Beauty", "sub_category": "",
                    "store": "S", "price": 9.99, "average_rating": 4.0,
                    "rating_number": 10, "primary_image_url": "",
                }}
            ]},
        })
        mock_model = MagicMock()
        arr = MagicMock()
        arr.tolist = MagicMock(return_value=[0.0] * 384)
        mock_model.encode = MagicMock(return_value=arr)

        # Deliberately set the dep-injection os_client to None to trigger
        # the early-return branch in _log_search_event
        original = _deps._os_client
        try:
            _deps._os_client = None  # triggers line 591

            with (
                patch("api.dependencies._os_client", mock_os),
                patch("api.dependencies._embedding_model", mock_model),
                patch("api.cache.cache_get", new=AsyncMock(return_value=None)),
                patch("api.cache.cache_set", new=AsyncMock()),
                patch("api.cache.ping_redis", new=AsyncMock(return_value=True)),
                # Make _log_search_event see None for os_client
                patch("api.main._deps") as mock_deps,
            ):
                mock_deps._os_client = None
                mock_deps.get_os_client = MagicMock(return_value=mock_os)
                mock_deps.get_model = MagicMock(return_value=mock_model)

                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as ac:
                    resp = await ac.get("/search?q=test")
                    # Request should complete successfully even though logging
                    # skipped due to None os_client
                    assert resp.status_code == 200
        finally:
            _deps._os_client = original

    @pytest.mark.asyncio
    async def test_log_search_event_none_client_does_not_raise(self):
        """Direct unit test of the _log_search_event coroutine."""
        import api.main as main_mod
        import api.dependencies as _deps

        original = _deps._os_client
        try:
            _deps._os_client = None  # triggers the early return at line 591
            await main_mod._log_search_event(
                query_text="test",
                mode="hybrid",
                hits=[],
                took_ms=10.0,
                cached=False,
                session_id=None,
            )
            # Should return silently — no exception
        finally:
            _deps._os_client = original


# ── ab_framework.py line 133: start_experiment raises for unknown id ──────────

class TestABFrameworkFinalGaps:
    def test_start_experiment_raises_for_unknown_id(self):
        """
        start_experiment() calls get_experiment() which returns None for
        an unknown ID, then raises ValueError — covers line 133.
        """
        from mlops.experiments.ab_framework import ABFramework

        mock_redis = MagicMock()
        mock_redis.get = MagicMock(return_value=None)  # experiment not found
        framework = ABFramework(mock_redis)

        with pytest.raises(ValueError, match="not found"):
            framework.start_experiment("nonexistent-experiment-id")

    def test_get_config_for_session_returns_none_when_experiment_deleted(self):
        """
        get_config_for_session() can return None if the experiment is deleted
        between assign_variant() and get_experiment() — covers line 208.
        """
        from mlops.experiments.ab_framework import ABFramework

        mock_redis = MagicMock()
        # assign_variant returns a variant_id from cache
        mock_redis.get = MagicMock(return_value="control")
        # But then get_experiment finds nothing (race condition / deletion)
        framework = ABFramework(mock_redis)

        with patch.object(framework, "get_experiment", return_value=None):
            result = framework.get_config_for_session("deleted-exp", "sess-123")

        assert result is None
