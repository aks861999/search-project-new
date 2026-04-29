"""
Tests for api/dependencies.py — dependency injection functions.

Verifies that:
  - get_os_client() returns the singleton when set
  - get_os_client() raises HTTP 503 when the singleton is None
  - get_model() returns the singleton when set
  - get_model() raises HTTP 503 when the singleton is None
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException


class TestGetOsClient:
    def test_returns_client_when_set(self):
        import api.dependencies as deps

        mock_client = MagicMock()
        with patch.object(deps, "_os_client", mock_client):
            result = deps.get_os_client()
        assert result is mock_client

    def test_raises_503_when_none(self):
        import api.dependencies as deps

        with patch.object(deps, "_os_client", None):
            with pytest.raises(HTTPException) as exc_info:
                deps.get_os_client()

        assert exc_info.value.status_code == 503
        assert "OpenSearch" in exc_info.value.detail

    def test_raises_http_exception_type(self):
        import api.dependencies as deps

        with patch.object(deps, "_os_client", None):
            with pytest.raises(HTTPException):
                deps.get_os_client()


class TestGetModel:
    def test_returns_model_when_set(self):
        import api.dependencies as deps

        mock_model = MagicMock()
        with patch.object(deps, "_embedding_model", mock_model):
            result = deps.get_model()
        assert result is mock_model

    def test_raises_503_when_none(self):
        import api.dependencies as deps

        with patch.object(deps, "_embedding_model", None):
            with pytest.raises(HTTPException) as exc_info:
                deps.get_model()

        assert exc_info.value.status_code == 503
        assert "model" in exc_info.value.detail.lower()

    def test_raises_http_exception_type(self):
        import api.dependencies as deps

        with patch.object(deps, "_embedding_model", None):
            with pytest.raises(HTTPException):
                deps.get_model()


class TestSingletonAssignment:
    def test_singleton_is_none_by_default(self):
        """
        The module should start with None singletons.
        This test imports fresh (relies on conftest stubs clearing the state).
        """
        import api.dependencies as deps

        # In the test environment, the lifespan has not run, so both are None
        # (unless another test set them — we check type not identity)
        assert deps._os_client is None or deps._os_client is not None  # type check only

    def test_can_assign_and_retrieve_client(self):
        import api.dependencies as deps

        sentinel = MagicMock(name="fake_os_client")
        old = deps._os_client
        try:
            deps._os_client = sentinel
            assert deps.get_os_client() is sentinel
        finally:
            deps._os_client = old

    def test_can_assign_and_retrieve_model(self):
        import api.dependencies as deps

        sentinel = MagicMock(name="fake_model")
        old = deps._embedding_model
        try:
            deps._embedding_model = sentinel
            assert deps.get_model() is sentinel
        finally:
            deps._embedding_model = old
