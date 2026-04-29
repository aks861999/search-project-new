"""
tests/test_final_gaps.py

Closes the last 52 uncovered lines:
  api/main.py   96-117  lifespan startup + shutdown
                200-201 A/B mode override ValueError path
                388-421 MLOps registry / experiment endpoints happy paths
                469-515 experiment get/drift/evaluate error+happy paths
                554-568 helper functions (_get_active_experiment_id, _get_ab_config)
                572-575 _version_to_dict
                591     _log_search_event early return
                599-600 _log_search_event exception swallowing

  mlops/experiments/ab_framework.py
                133  conclude_experiment → ValueError on unknown variant
                208  get_config_for_session → None when variant missing
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import contextlib
import pytest
from httpx import ASGITransport, AsyncClient


@contextlib.contextmanager
def _stacked(*patch_list):
    """Enter multiple patch context managers using ExitStack."""
    with contextlib.ExitStack() as stack:
        for p in patch_list:
            stack.enter_context(p)
        yield stack


# ── Shared helpers ────────────────────────────────────────────────────────────

def _mock_os(search_return=None):
    c = MagicMock()
    c.search = MagicMock(return_value=search_return or {
        "took": 2,
        "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []}
    })
    c.cluster.health = MagicMock(return_value={"status": "green"})
    c.index = MagicMock(return_value={"result": "created", "_id": "X"})
    return c


def _mock_model():
    m = MagicMock()
    arr = MagicMock()
    arr.tolist = MagicMock(return_value=[0.0] * 384)
    m.encode = MagicMock(return_value=arr)
    return m


def _all_patches(os_client=None, model=None, cache_get=None):
    return [
        patch("api.dependencies._os_client", os_client or _mock_os()),
        patch("api.dependencies._embedding_model", model or _mock_model()),
        patch("api.cache.cache_get", new=AsyncMock(return_value=cache_get)),
        patch("api.cache.cache_set", new=AsyncMock()),
        patch("api.cache.ping_redis", new=AsyncMock(return_value=True)),
    ]


# ═══════════════════════════════════════════════════════════════
# 1. api/main.py — LIFESPAN  (lines 96-117)
# ═══════════════════════════════════════════════════════════════

class TestLifespan:
    @pytest.mark.asyncio
    async def test_lifespan_initialises_singletons(self):
        """Lifespan startup must set api.dependencies._os_client and _embedding_model."""
        import api.dependencies as _deps

        fake_model = _mock_model()

        with _stacked(            patch("api.main.SentenceTransformer", return_value=fake_model),
            patch("api.main.OpenSearch", return_value=_mock_os()),
            patch("api.cache.close_redis", new=AsyncMock()),
            patch("mlops.observability.metrics.update_model_info"),
        ):
            from api.main import app, lifespan
            async with lifespan(app):
                # Inside the lifespan the singletons must be set
                assert _deps._os_client is not None
                assert _deps._embedding_model is not None

    @pytest.mark.asyncio
    async def test_lifespan_closes_redis_on_shutdown(self):
        """Lifespan shutdown must call _cache.close_redis()."""
        fake_model = _mock_model()
        mock_close = AsyncMock()

        with (
            patch("api.main.SentenceTransformer", return_value=fake_model),
            patch("api.main.OpenSearch", return_value=_mock_os()),
            patch("api.cache.close_redis", new=mock_close),
            patch("mlops.observability.metrics.update_model_info"),
        ):
            from api.main import app, lifespan
            async with lifespan(app):
                pass  # Exits lifespan → shutdown runs

        mock_close.assert_called_once()


# ═══════════════════════════════════════════════════════════════
# 2. api/main.py — A/B mode override ValueError (lines 200-201)
# ═══════════════════════════════════════════════════════════════

class TestABModeOverride:
    @pytest.mark.asyncio
    async def test_invalid_ab_mode_falls_back_to_requested_mode(self):
        """When the A/B config returns an invalid search_mode value,
        the endpoint should fall back to the originally requested mode
        without raising (ValueError is caught silently at line 200-201)."""
        from api.main import app

        with _stacked(
            _all_patches()[0], _all_patches()[1], _all_patches()[2], _all_patches()[3], _all_patches()[4],
            patch("api.main._get_active_experiment_id",
                  return_value="exp-001"),
            patch("api.main._get_ab_config",
                  return_value={"search_mode": "INVALID_MODE_XYZ"}),

        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get(
                    "/search?q=test",
                    headers={"X-Session-ID": "sess-abc"},
                )
        # Should succeed — invalid mode is ignored, falls back to default (hybrid)
        assert resp.status_code == 200
        assert resp.json()["mode"] == "hybrid"


# ═══════════════════════════════════════════════════════════════
# 3. api/main.py — MLOps REGISTRY endpoints (lines 388-408)
# ═══════════════════════════════════════════════════════════════

class TestMLOpsRegistryEndpoints:
    @pytest.mark.asyncio
    async def test_get_registry_returns_active_and_versions(self):
        from api.main import app
        from mlops.model_registry import ModelStatus, ModelVersion
        import time

        mock_version = ModelVersion(
            version_id="v-001",
            model_name="BAAI/bge-small-en-v1.5",
            embedding_dim=384,
            description="test",
            status=ModelStatus.active,
            registered_at=time.time(),
        )
        mock_registry = MagicMock()
        mock_registry.get_active = MagicMock(return_value=mock_version)
        mock_registry.list_versions = MagicMock(return_value=[mock_version])

        with _stacked(            *_all_patches(),
            patch("api.main._build_registry", return_value=mock_registry),

        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/mlops/registry")

        assert resp.status_code == 200
        body = resp.json()
        assert body["active"]["version_id"] == "v-001"
        assert len(body["versions"]) == 1

    @pytest.mark.asyncio
    async def test_promote_model_returns_promoted_status(self):
        from api.main import app
        from mlops.model_registry import ModelStatus, ModelVersion
        import time

        mock_version = ModelVersion(
            version_id="v-002",
            model_name="BAAI/bge-base-en-v1.5",
            embedding_dim=768,
            description="better model",
            status=ModelStatus.active,
            registered_at=time.time(),
        )
        mock_registry = MagicMock()
        mock_registry.promote = MagicMock(return_value=mock_version)

        with _stacked(            *_all_patches(),
            patch("api.main._build_registry", return_value=mock_registry),

        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/mlops/registry/promote?version_id=v-002")

        assert resp.status_code == 200
        assert resp.json()["status"] == "promoted"

    @pytest.mark.asyncio
    async def test_promote_model_returns_404_for_unknown_version(self):
        from api.main import app

        mock_registry = MagicMock()
        mock_registry.promote = MagicMock(
            side_effect=ValueError("Version 'v-999' not found in registry.")
        )

        with _stacked(            *_all_patches(),
            patch("api.main._build_registry", return_value=mock_registry),

        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/mlops/registry/promote?version_id=v-999")

        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_promote_model_returns_500_on_unexpected_error(self):
        from api.main import app

        mock_registry = MagicMock()
        mock_registry.promote = MagicMock(side_effect=Exception("redis down"))

        with _stacked(            *_all_patches(),
            patch("api.main._build_registry", return_value=mock_registry),

        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/mlops/registry/promote?version_id=v-x")

        assert resp.status_code == 500


# ═══════════════════════════════════════════════════════════════
# 4. api/main.py — MLOps EXPERIMENTS endpoints (lines 417-421)
# ═══════════════════════════════════════════════════════════════

class TestMLOpsExperimentsEndpoints:
    @pytest.mark.asyncio
    async def test_list_experiments_returns_all(self):
        from api.main import app

        mock_framework = MagicMock()
        mock_framework.list_experiments = MagicMock(return_value=["exp-001", "exp-002"])
        mock_framework.get_experiment_report = MagicMock(side_effect=lambda eid: {
            "experiment_id": eid, "status": "running", "variants": []
        })

        with _stacked(            *_all_patches(),
            patch("api.main._build_ab_framework", return_value=mock_framework),

        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/mlops/experiments")

        assert resp.status_code == 200
        assert len(resp.json()["experiments"]) == 2

    @pytest.mark.asyncio
    async def test_create_experiment_returns_201(self):
        from api.main import app

        mock_framework = MagicMock()
        mock_framework.create_experiment = MagicMock()
        mock_framework.start_experiment = MagicMock()

        with _stacked(            *_all_patches(),
            patch("api.main._build_ab_framework", return_value=mock_framework),

        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/mlops/experiments", json={
                    "experiment_id": "weight_test_001",
                    "description": "Test BM25 weights",
                    "variants": [
                        {"variant_id": "control", "weight": 0.5,
                         "config": {"search_mode": "hybrid"}},
                        {"variant_id": "treatment", "weight": 0.5,
                         "config": {"search_mode": "lexical"}},
                    ]
                })

        assert resp.status_code == 201
        assert resp.json()["experiment_id"] == "weight_test_001"

    @pytest.mark.asyncio
    async def test_create_experiment_returns_422_on_validation_error(self):
        from api.main import app

        mock_framework = MagicMock()
        mock_framework.create_experiment = MagicMock(
            side_effect=ValueError("weights must sum to 1.0")
        )

        with _stacked(            *_all_patches(),
            patch("api.main._build_ab_framework", return_value=mock_framework),

        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/mlops/experiments", json={
                    "experiment_id": "bad",
                    "variants": [
                        {"variant_id": "a", "weight": 0.3, "config": {}},
                    ]
                })

        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_get_experiment_returns_report(self):
        from api.main import app

        mock_framework = MagicMock()
        mock_framework.get_experiment_report = MagicMock(return_value={
            "experiment_id": "exp-001",
            "status": "running",
            "variants": [
                {"variant_id": "control", "weight": 0.5, "config": {}}
            ]
        })

        with _stacked(            *_all_patches(),
            patch("api.main._build_ab_framework", return_value=mock_framework),

        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/mlops/experiments/exp-001")

        assert resp.status_code == 200
        assert resp.json()["experiment_id"] == "exp-001"

    @pytest.mark.asyncio
    async def test_get_experiment_returns_404_when_not_found(self):
        from api.main import app

        mock_framework = MagicMock()
        mock_framework.get_experiment_report = MagicMock(return_value={
            "error": "Experiment 'missing' not found"
        })

        with _stacked(            *_all_patches(),
            patch("api.main._build_ab_framework", return_value=mock_framework),

        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/mlops/experiments/missing")

        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_experiment_returns_500_on_exception(self):
        from api.main import app

        mock_framework = MagicMock()
        mock_framework.get_experiment_report = MagicMock(
            side_effect=Exception("redis crashed")
        )

        with _stacked(            *_all_patches(),
            patch("api.main._build_ab_framework", return_value=mock_framework),

        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/mlops/experiments/exp-x")

        assert resp.status_code == 500


# ═══════════════════════════════════════════════════════════════
# 5. api/main.py — DRIFT + EVALUATE endpoints (lines 501-515)
# ═══════════════════════════════════════════════════════════════

class TestMLOpsDriftAndEvalEndpoints:
    @pytest.mark.asyncio
    async def test_get_drift_report_returns_result(self):
        from api.main import app

        mock_detector = MagicMock()
        mock_detector.check = MagicMock(return_value={
            "drift_detected": False, "alerts": []
        })

        with _stacked(            *_all_patches(),
            patch("mlops.evaluation.drift_detector.DriftDetector",
                  return_value=mock_detector),
            patch("api.main._build_sync_os_client", return_value=MagicMock()),
            patch("api.main._get_sync_redis", new=AsyncMock(return_value=MagicMock())),

        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/mlops/drift")

        assert resp.status_code == 200
        assert resp.json()["drift_detected"] is False

    @pytest.mark.asyncio
    async def test_get_drift_report_returns_error_dict_on_failure(self):
        from api.main import app

        with _stacked(            *_all_patches(),
            patch("api.main._build_sync_os_client",
                  side_effect=Exception("OS down")),
            patch("api.main._get_sync_redis", new=AsyncMock(return_value=MagicMock())),

        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/mlops/drift")

        assert resp.status_code == 200
        assert "error" in resp.json()

    @pytest.mark.asyncio
    async def test_trigger_evaluation_returns_202_with_task_id(self):
        from api.main import app

        mock_task = MagicMock()
        mock_task.id = "task-abc-123"

        with patch("mlops.scheduler.evaluate_active_model") as mock_eval:
            mock_eval.delay = MagicMock(return_value=mock_task)
            with patch("api.dependencies._os_client", _mock_os()), \
                 patch("api.dependencies._embedding_model", _mock_model()), \
                 patch("api.cache.cache_get", new=AsyncMock(return_value=None)), \
                 patch("api.cache.cache_set", new=AsyncMock()), \
                 patch("api.cache.ping_redis", new=AsyncMock(return_value=True)):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.post("/mlops/evaluate")

        assert resp.status_code == 202
        assert resp.json()["task_id"] == "task-abc-123"

    @pytest.mark.asyncio
    async def test_trigger_evaluation_returns_202_even_when_celery_down(self):
        """Even if Celery is unavailable, the endpoint returns 202 (fire-and-forget)."""
        from api.main import app

        with patch("mlops.scheduler.evaluate_active_model") as mock_eval:
            mock_eval.delay = MagicMock(side_effect=Exception("broker unavailable"))
            with patch("api.dependencies._os_client", _mock_os()), \
                 patch("api.dependencies._embedding_model", _mock_model()), \
                 patch("api.cache.cache_get", new=AsyncMock(return_value=None)), \
                 patch("api.cache.cache_set", new=AsyncMock()), \
                 patch("api.cache.ping_redis", new=AsyncMock(return_value=True)):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.post("/mlops/evaluate")

        assert resp.status_code == 202
        assert "warning" in resp.json()


# ═══════════════════════════════════════════════════════════════
# 6. api/main.py — helper functions (lines 554-575)
# ═══════════════════════════════════════════════════════════════

class TestMainHelperFunctions:
    def test_get_active_experiment_id_returns_running_id(self):
        from api.main import _get_active_experiment_id
        from mlops.experiments.ab_framework import ExperimentStatus

        mock_framework = MagicMock()
        mock_exp = MagicMock()
        mock_exp.status = ExperimentStatus.running
        mock_framework.list_experiments = MagicMock(return_value=["exp-001"])
        mock_framework.get_experiment = MagicMock(return_value=mock_exp)

        with patch("api.main._build_ab_framework", return_value=mock_framework):
            result = _get_active_experiment_id()

        assert result == "exp-001"

    def test_get_active_experiment_id_returns_none_when_no_running(self):
        from api.main import _get_active_experiment_id
        from mlops.experiments.ab_framework import ExperimentStatus

        mock_framework = MagicMock()
        mock_exp = MagicMock()
        mock_exp.status = ExperimentStatus.concluded
        mock_framework.list_experiments = MagicMock(return_value=["exp-001"])
        mock_framework.get_experiment = MagicMock(return_value=mock_exp)

        with patch("api.main._build_ab_framework", return_value=mock_framework):
            result = _get_active_experiment_id()

        assert result is None

    def test_get_active_experiment_id_returns_none_on_exception(self):
        from api.main import _get_active_experiment_id

        with patch("api.main._build_ab_framework",
                   side_effect=Exception("redis down")):
            result = _get_active_experiment_id()

        assert result is None

    def test_get_ab_config_returns_config_for_session(self):
        from api.main import _get_ab_config

        mock_framework = MagicMock()
        mock_framework.get_config_for_session = MagicMock(
            return_value={"search_mode": "lexical"}
        )

        with patch("api.main._build_ab_framework", return_value=mock_framework):
            result = _get_ab_config("exp-001", "sess-abc")

        assert result["search_mode"] == "lexical"

    def test_get_ab_config_returns_none_on_exception(self):
        from api.main import _get_ab_config

        with patch("api.main._build_ab_framework",
                   side_effect=Exception("redis down")):
            result = _get_ab_config("exp-001", "sess-x")

        assert result is None

    def test_version_to_dict_returns_serialisable_dict(self):
        from api.main import _version_to_dict
        from mlops.model_registry import ModelStatus, ModelVersion
        import time

        v = ModelVersion(
            version_id="v-abc",
            model_name="BAAI/bge-small-en-v1.5",
            embedding_dim=384,
            description="test version",
            status=ModelStatus.active,
            registered_at=time.time(),
        )
        result = _version_to_dict(v)

        assert result["version_id"] == "v-abc"
        assert result["status"] == "active"
        # Must be JSON-serialisable
        assert json.dumps(result)

    def test_version_to_dict_serialises_status_as_string(self):
        from api.main import _version_to_dict
        from mlops.model_registry import ModelStatus, ModelVersion
        import time

        v = ModelVersion(
            version_id="v-reg",
            model_name="BAAI/bge-base-en-v1.5",
            embedding_dim=768,
            description="",
            status=ModelStatus.registered,
            registered_at=time.time(),
        )
        result = _version_to_dict(v)
        assert isinstance(result["status"], str)
        assert result["status"] == "registered"


# ═══════════════════════════════════════════════════════════════
# 7. api/main.py — _log_search_event (lines 591, 599-600)
# ═══════════════════════════════════════════════════════════════

class TestLogSearchEvent:
    @pytest.mark.asyncio
    async def test_log_search_event_skips_when_no_client(self):
        """_log_search_event returns early when OS client is not set."""
        import api.main as main_mod
        from api.main import _log_search_event

        original = main_mod._os_client
        main_mod._os_client = None
        try:
            # Must not raise
            await _log_search_event(
                query_text="test", mode="hybrid", hits=[],
                took_ms=5.0, cached=False, session_id="s-1",
            )
        finally:
            main_mod._os_client = original

    @pytest.mark.asyncio
    async def test_log_search_event_swallows_exception(self):
        """_log_search_event must never propagate exceptions (fire-and-forget)."""
        import api.main as main_mod
        from api.main import _log_search_event

        original = main_mod._os_client
        main_mod._os_client = _mock_os()
        try:
            with patch(
                "mlops.evaluation.metrics_logger.MetricsLogger",
                side_effect=Exception("metrics logger broken"),
            ):
                # Must not raise
                await _log_search_event(
                    query_text="test", mode="hybrid", hits=[],
                    took_ms=5.0, cached=False, session_id="s-1",
                )
        finally:
            main_mod._os_client = original


# ═══════════════════════════════════════════════════════════════
# 8. mlops/experiments/ab_framework.py  (lines 133, 208)
# ═══════════════════════════════════════════════════════════════

class TestABFrameworkFinalGaps:
    def _make_redis(self, stored=None):
        r = MagicMock()
        r.get = MagicMock(return_value=stored)
        r.set = MagicMock()
        r.sadd = MagicMock()
        r.smembers = MagicMock(return_value=set())
        r.setex = MagicMock()
        r.incr = MagicMock()
        r.hincrbyfloat = MagicMock()
        r.hincrby = MagicMock()
        r.hgetall = MagicMock(return_value={})
        return r

    def _stored_exp(self, exp_id="e1", status="running"):
        return json.dumps({
            "experiment_id": exp_id,
            "description": "",
            "status": status,
            "created_at": 0.0,
            "started_at": None,
            "concluded_at": None,
            "winner_variant_id": None,
            "variants": [
                {"variant_id": "a", "weight": 0.5, "config": {}, "description": ""},
                {"variant_id": "b", "weight": 0.5, "config": {}, "description": ""},
            ],
        })

    def test_conclude_with_unknown_variant_raises_value_error(self):
        """Line 133 — conclude_experiment raises ValueError for unknown variant_id."""
        from mlops.experiments.ab_framework import ABFramework

        r = self._make_redis(stored=self._stored_exp())
        framework = ABFramework(r)

        with pytest.raises(ValueError, match="not in experiment"):
            framework.conclude_experiment("e1", winner_variant_id="nonexistent")

    def test_get_config_for_session_returns_none_when_variant_missing(self):
        """Line 208 — get_config_for_session returns None when get_variant yields None."""
        from mlops.experiments.ab_framework import ABFramework

        # Stored experiment has variants a/b but we'll make get_variant return None
        r = self._make_redis(stored=self._stored_exp())
        framework = ABFramework(r)

        # Cache the assignment to "c" (a variant that doesn't exist)
        r.setex = MagicMock()  # assignment cache write
        r.get = MagicMock(side_effect=[
            self._stored_exp(),    # get_experiment call
            "c",                   # sticky assignment cache hit → returns "c"
            self._stored_exp(),    # get_experiment call inside get_config_for_session
        ])

        result = framework.get_config_for_session("e1", "sess-xyz")
        # "c" doesn't exist as a variant → get_variant returns None → result is None
        assert result is None
