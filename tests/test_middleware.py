"""
Tests for api/middleware.py — RequestTimingMiddleware and CorrelationIdMiddleware.

Uses a minimal Starlette test app so no FastAPI lifespan or OS client is needed.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from api.middleware import CorrelationIdMiddleware, RequestTimingMiddleware, add_middleware


# ── Minimal test app ──────────────────────────────────────────────────────────

async def _echo(request: Request):
    return JSONResponse({"path": request.url.path})


async def _slow(request: Request):
    import asyncio
    await asyncio.sleep(0.01)
    return JSONResponse({"ok": True})


def _make_app():
    app = Starlette(routes=[
        Route("/echo", _echo),
        Route("/slow", _slow),
        Route("/health", _echo),
    ])
    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(RequestTimingMiddleware)
    return app


@pytest.fixture
def app():
    return _make_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── RequestTimingMiddleware ───────────────────────────────────────────────────

class TestRequestTimingMiddleware:
    @pytest.mark.asyncio
    async def test_adds_process_time_header(self, client):
        resp = await client.get("/echo")
        assert "x-process-time-ms" in resp.headers

    @pytest.mark.asyncio
    async def test_process_time_is_numeric(self, client):
        resp = await client.get("/echo")
        ms = float(resp.headers["x-process-time-ms"])
        assert ms >= 0

    @pytest.mark.asyncio
    async def test_slow_endpoint_shows_nonzero_time(self, client):
        resp = await client.get("/slow")
        ms = float(resp.headers["x-process-time-ms"])
        assert ms > 0

    @pytest.mark.asyncio
    async def test_health_endpoint_still_returns_header(self, client):
        """Even skipped paths get the timing header (it's set on the response)."""
        resp = await client.get("/health")
        assert "x-process-time-ms" in resp.headers


# ── CorrelationIdMiddleware ───────────────────────────────────────────────────

class TestCorrelationIdMiddleware:
    @pytest.mark.asyncio
    async def test_generates_correlation_id_when_absent(self, client):
        resp = await client.get("/echo")
        assert "x-correlation-id" in resp.headers
        cid = resp.headers["x-correlation-id"]
        assert len(cid) == 36  # UUID4 format

    @pytest.mark.asyncio
    async def test_forwards_existing_correlation_id(self, client):
        cid = "my-trace-id-12345"
        resp = await client.get("/echo", headers={"X-Correlation-ID": cid})
        assert resp.headers["x-correlation-id"] == cid

    @pytest.mark.asyncio
    async def test_different_requests_get_different_ids(self, client):
        r1 = await client.get("/echo")
        r2 = await client.get("/echo")
        assert r1.headers["x-correlation-id"] != r2.headers["x-correlation-id"]

    @pytest.mark.asyncio
    async def test_correlation_id_is_valid_uuid(self, client):
        import uuid
        resp = await client.get("/echo")
        cid = resp.headers["x-correlation-id"]
        # Should not raise
        parsed = uuid.UUID(cid)
        assert str(parsed) == cid


# ── add_middleware helper ─────────────────────────────────────────────────────

class TestAddMiddleware:
    def test_add_middleware_registers_both(self):
        """add_middleware() should not raise and should be callable on a FastAPI app."""
        from fastapi import FastAPI
        app = FastAPI()
        add_middleware(app)  # must not raise

    @pytest.mark.asyncio
    async def test_full_app_has_both_headers(self):
        """When both middleware are registered the response has both headers."""
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        fapp = FastAPI()
        add_middleware(fapp)

        @fapp.get("/ping")
        async def ping():
            return {"pong": True}

        transport = ASGITransport(app=fapp)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/ping")
            assert "x-process-time-ms" in resp.headers
            assert "x-correlation-id" in resp.headers
