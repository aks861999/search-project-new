"""
Tests for ingestion/pipeline.py — pipeline integration and error-path coverage.

Extends test_pipeline.py (which covers pure helper functions) with:
  - wait_for_opensearch retry logic
  - ensure_index (create vs skip)
  - ensure_search_pipeline (create vs skip)
  - run_pipeline end-to-end with everything mocked
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from ingestion.pipeline import (
    ensure_index,
    ensure_search_pipeline,
    wait_for_opensearch,
)


# ── wait_for_opensearch ───────────────────────────────────────────────────────

class TestWaitForOpensearch:
    def test_succeeds_immediately_when_healthy(self):
        client = MagicMock()
        client.cluster.health = MagicMock(return_value={"status": "yellow"})
        wait_for_opensearch(client, retries=3, delay=0)
        client.cluster.health.assert_called_once()

    def test_retries_then_succeeds(self):
        client = MagicMock()
        # Fail twice, succeed on third attempt
        client.cluster.health = MagicMock(
            side_effect=[Exception("not ready"), Exception("still not"), {"status": "green"}]
        )
        wait_for_opensearch(client, retries=5, delay=0)
        assert client.cluster.health.call_count == 3

    def test_raises_after_all_retries_exhausted(self):
        client = MagicMock()
        client.cluster.health = MagicMock(side_effect=Exception("never ready"))
        with pytest.raises(RuntimeError, match="did not become ready"):
            wait_for_opensearch(client, retries=3, delay=0)
        assert client.cluster.health.call_count == 3


# ── ensure_index ─────────────────────────────────────────────────────────────

class TestEnsureIndex:
    def test_creates_index_when_absent(self):
        client = MagicMock()
        client.indices.exists = MagicMock(return_value=False)
        client.indices.create = MagicMock(return_value={"acknowledged": True})

        ensure_index(client, "products")

        client.indices.create.assert_called_once()
        call_kwargs = client.indices.create.call_args[1]
        assert call_kwargs["index"] == "products"

    def test_skips_creation_when_index_exists(self):
        client = MagicMock()
        client.indices.exists = MagicMock(return_value=True)

        ensure_index(client, "products")

        client.indices.create.assert_not_called()

    def test_passes_index_settings(self):
        from ingestion.schema import INDEX_SETTINGS

        client = MagicMock()
        client.indices.exists = MagicMock(return_value=False)
        client.indices.create = MagicMock(return_value={"acknowledged": True})

        ensure_index(client, "products")

        _, kwargs = client.indices.create.call_args
        assert kwargs["body"] == INDEX_SETTINGS


# ── ensure_search_pipeline ────────────────────────────────────────────────────

class TestEnsureSearchPipeline:
    def test_skips_when_pipeline_exists(self):
        client = MagicMock()
        client.transport.perform_request = MagicMock(return_value={"hybrid-search-pipeline": {}})

        ensure_search_pipeline(client)

        # GET called once (existence check), PUT not called
        assert client.transport.perform_request.call_count == 1
        get_call = client.transport.perform_request.call_args
        assert get_call[0][0] == "GET"

    def test_registers_pipeline_when_absent(self):
        client = MagicMock()
        # First call (GET) raises → pipeline absent; second call (PUT) succeeds
        client.transport.perform_request = MagicMock(
            side_effect=[Exception("not found"), {"acknowledged": True}]
        )

        ensure_search_pipeline(client)

        assert client.transport.perform_request.call_count == 2
        put_call = client.transport.perform_request.call_args
        assert put_call[0][0] == "PUT"

    def test_pipeline_body_matches_schema(self):
        from ingestion.schema import HYBRID_PIPELINE_ID, HYBRID_SEARCH_PIPELINE

        client = MagicMock()
        client.transport.perform_request = MagicMock(
            side_effect=[Exception("not found"), {"acknowledged": True}]
        )

        ensure_search_pipeline(client)

        put_call = client.transport.perform_request.call_args
        assert HYBRID_PIPELINE_ID in put_call[0][1]
        assert put_call[1]["body"] == HYBRID_SEARCH_PIPELINE


# ── run_pipeline integration ──────────────────────────────────────────────────

class TestRunPipeline:
    def test_run_pipeline_indexes_all_docs(self):
        """
        Full integration: mock dataset, model, and OS client.
        Verify helpers.bulk is called with the right number of actions.
        """
        import numpy as np
        from ingestion.pipeline import run_pipeline

        # Two fake dataset records
        fake_records = [
            {
                "parent_asin": "B001",
                "title": "Shampoo",
                "features": ["sulfate-free"],
                "description": ["Great for dry hair"],
                "main_category": "All Beauty",
                "sub_category": "",
                "store": "BrandX",
                "price": "14.99",
                "average_rating": 4.5,
                "rating_number": 100,
                "images": [{"large": "http://img.example.com/1.jpg"}],
                "details": {},
            },
            {
                "parent_asin": "B002",
                "title": "Conditioner",
                "features": ["moisturizing"],
                "description": ["Deep treatment"],
                "main_category": "All Beauty",
                "sub_category": "",
                "store": "BrandY",
                "price": "12.50",
                "average_rating": 4.2,
                "rating_number": 50,
                "images": [],
                "details": {},
            },
        ]

        mock_client = MagicMock()
        mock_client.cluster.health = MagicMock(return_value={"status": "green"})
        mock_client.indices.exists = MagicMock(return_value=False)
        mock_client.indices.create = MagicMock(return_value={"acknowledged": True})
        mock_client.transport.perform_request = MagicMock(
            side_effect=[Exception("no pipeline"), {"acknowledged": True}]
        )

        mock_model = MagicMock()
        mock_model.encode = MagicMock(
            return_value=np.zeros((2, 384), dtype="float32")
        )

        with (
            patch("ingestion.pipeline.get_client", return_value=mock_client),
            patch("ingestion.pipeline.SentenceTransformer", return_value=mock_model),
            patch("ingestion.pipeline.load_dataset", return_value=fake_records),
            patch("ingestion.pipeline.helpers.bulk", return_value=(2, [])) as mock_bulk,
        ):
            run_pipeline()

        # bulk should have been called at least once
        mock_bulk.assert_called()
        # All actions passed to bulk should have embedding_vector
        all_actions = []
        for c in mock_bulk.call_args_list:
            all_actions.extend(c[0][1])  # second positional arg is the actions list
        for action in all_actions:
            assert "embedding_vector" in action["_source"]
            assert len(action["_source"]["embedding_vector"]) == 384

    def test_run_pipeline_skips_records_without_asin(self):
        """Records with no ASIN are silently skipped."""
        import numpy as np
        from ingestion.pipeline import run_pipeline

        fake_records = [
            {"parent_asin": None, "title": "No ASIN product"},
            {"parent_asin": "B003", "title": "Valid", "features": [],
             "description": [], "main_category": "Beauty", "sub_category": "",
             "store": "S", "price": "10", "average_rating": 4.0,
             "rating_number": 10, "images": [], "details": {}},
        ]

        mock_client = MagicMock()
        mock_client.cluster.health = MagicMock(return_value={"status": "green"})
        mock_client.indices.exists = MagicMock(return_value=True)
        mock_client.transport.perform_request = MagicMock(return_value={})

        mock_model = MagicMock()
        mock_model.encode = MagicMock(
            return_value=__import__("numpy").zeros((1, 384), dtype="float32")
        )

        with (
            patch("ingestion.pipeline.get_client", return_value=mock_client),
            patch("ingestion.pipeline.SentenceTransformer", return_value=mock_model),
            patch("ingestion.pipeline.load_dataset", return_value=fake_records),
            patch("ingestion.pipeline.helpers.bulk", return_value=(1, [])) as mock_bulk,
        ):
            run_pipeline()

        # Only 1 valid document should have been indexed
        all_actions = []
        for c in mock_bulk.call_args_list:
            all_actions.extend(c[0][1])
        assert len(all_actions) == 1
        assert all_actions[0]["_id"] == "B003"
