"""
Tests for ingestion/pipeline.py — preprocessing helpers and pipeline logic.

No network calls or HuggingFace downloads are made; all external I/O is mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from ingestion.pipeline import (
    _extract_image_url,
    _safe_float,
    _safe_int,
    _safe_str,
    build_bulk_actions,
    iter_batches,
    preprocess_record,
)


# ── _safe_str ─────────────────────────────────────────────────────────────────

class TestSafeStr:
    def test_plain_string_returned(self):
        assert _safe_str("hello") == "hello"

    def test_none_returns_empty(self):
        assert _safe_str(None) == ""

    def test_empty_string_returns_empty(self):
        assert _safe_str("") == ""

    def test_list_joined_with_space(self):
        assert _safe_str(["a", "b", "c"]) == "a b c"

    def test_list_with_none_skips_none(self):
        assert _safe_str(["a", None, "c"]) == "a c"

    def test_empty_list_returns_empty(self):
        assert _safe_str([]) == ""

    def test_integer_coerced_to_string(self):
        assert _safe_str(42) == "42"

    def test_list_with_mixed_types(self):
        result = _safe_str([1, "two", 3])
        assert "1" in result and "two" in result


# ── _safe_float ───────────────────────────────────────────────────────────────

class TestSafeFloat:
    def test_plain_float(self):
        assert _safe_float(3.14) == pytest.approx(3.14)

    def test_integer_input(self):
        assert _safe_float(5) == 5.0

    def test_string_float(self):
        assert _safe_float("9.99") == pytest.approx(9.99)

    def test_dollar_sign_stripped(self):
        assert _safe_float("$24.99") == pytest.approx(24.99)

    def test_comma_stripped(self):
        assert _safe_float("1,299.99") == pytest.approx(1299.99)

    def test_none_returns_none(self):
        assert _safe_float(None) is None

    def test_non_numeric_string_returns_none(self):
        assert _safe_float("not a number") is None

    def test_negative_value_returns_none(self):
        assert _safe_float(-5.0) is None

    def test_zero_is_valid(self):
        assert _safe_float(0) == 0.0

    def test_whitespace_stripped(self):
        assert _safe_float("  12.5  ") == pytest.approx(12.5)


# ── _safe_int ─────────────────────────────────────────────────────────────────

class TestSafeInt:
    def test_plain_int(self):
        assert _safe_int(42) == 42

    def test_string_int(self):
        assert _safe_int("100") == 100

    def test_float_string_truncated(self):
        assert _safe_int("3.7") == 3

    def test_none_returns_none(self):
        assert _safe_int(None) is None

    def test_non_numeric_returns_none(self):
        assert _safe_int("abc") is None

    def test_zero(self):
        assert _safe_int(0) == 0

    def test_large_number(self):
        assert _safe_int("1000000") == 1_000_000


# ── _extract_image_url ────────────────────────────────────────────────────────

class TestExtractImageUrl:
    def test_large_url_preferred(self):
        images = [{"large": "http://img.com/large.jpg", "thumb": "http://img.com/thumb.jpg"}]
        assert _extract_image_url(images) == "http://img.com/large.jpg"

    def test_hi_res_fallback(self):
        images = [{"hi_res": "http://img.com/hires.jpg"}]
        assert _extract_image_url(images) == "http://img.com/hires.jpg"

    def test_thumb_last_resort(self):
        images = [{"thumb": "http://img.com/thumb.jpg"}]
        assert _extract_image_url(images) == "http://img.com/thumb.jpg"

    def test_none_returns_empty(self):
        assert _extract_image_url(None) == ""

    def test_empty_list_returns_empty(self):
        assert _extract_image_url([]) == ""

    def test_dict_without_any_url_returns_empty(self):
        assert _extract_image_url([{}]) == ""

    def test_non_list_returns_empty(self):
        assert _extract_image_url("not_a_list") == ""


# ── preprocess_record ─────────────────────────────────────────────────────────

class TestPreprocessRecord:
    def _minimal_record(self, **overrides) -> dict:
        base = {
            "parent_asin": "B001TEST01",
            "title": "Great Shampoo",
            "features": ["Sulfate-free", "Volumizing"],
            "description": ["Makes hair shiny and healthy."],
            "main_category": "All Beauty",
            "sub_category": "Hair Care",
            "store": "BeautyBrand",
            "price": "12.99",
            "average_rating": "4.5",
            "rating_number": "200",
            "images": [{"large": "http://img.com/shampoo.jpg"}],
        }
        base.update(overrides)
        return base

    def test_basic_record_processed(self):
        doc = preprocess_record(self._minimal_record())
        assert doc is not None
        assert doc["_id"] == "B001TEST01"
        assert doc["title"] == "Great Shampoo"

    def test_id_set_from_parent_asin(self):
        doc = preprocess_record(self._minimal_record(parent_asin="BASIN123"))
        assert doc["_id"] == "BASIN123"

    def test_missing_asin_returns_none(self):
        record = self._minimal_record()
        record.pop("parent_asin")
        assert preprocess_record(record) is None

    def test_empty_asin_returns_none(self):
        assert preprocess_record(self._minimal_record(parent_asin="")) is None

    def test_embedding_text_built_correctly(self):
        doc = preprocess_record(self._minimal_record())
        assert "Great Shampoo" in doc["embedding_text"]
        assert "Sulfate-free" in doc["embedding_text"]
        assert "Makes hair shiny" in doc["embedding_text"]

    def test_features_list_joined(self):
        doc = preprocess_record(self._minimal_record(features=["A", "B", "C"]))
        assert doc["features"] == "A B C"

    def test_price_coerced_to_float(self):
        doc = preprocess_record(self._minimal_record(price="$19.99"))
        assert doc["price"] == pytest.approx(19.99)

    def test_price_none_when_invalid(self):
        doc = preprocess_record(self._minimal_record(price="not_a_price"))
        assert doc["price"] is None

    def test_average_rating_coerced(self):
        doc = preprocess_record(self._minimal_record(average_rating="4.2"))
        assert doc["average_rating"] == pytest.approx(4.2)

    def test_rating_number_coerced_to_int(self):
        doc = preprocess_record(self._minimal_record(rating_number="350"))
        assert doc["rating_number"] == 350
        assert isinstance(doc["rating_number"], int)

    def test_primary_image_url_extracted(self):
        doc = preprocess_record(
            self._minimal_record(images=[{"large": "http://cdn.com/img.jpg"}])
        )
        assert doc["primary_image_url"] == "http://cdn.com/img.jpg"

    def test_missing_images_gives_empty_url(self):
        doc = preprocess_record(self._minimal_record(images=None))
        assert doc["primary_image_url"] == ""

    def test_description_uses_first_element(self):
        doc = preprocess_record(
            self._minimal_record(description=["First para.", "Second para."])
        )
        assert "First para." in doc["embedding_text"]
        assert "Second para." not in doc["embedding_text"]

    def test_all_required_keys_present(self):
        doc = preprocess_record(self._minimal_record())
        required = [
            "_id", "parent_asin", "title", "description", "features",
            "main_category", "sub_category", "store", "price",
            "average_rating", "rating_number", "primary_image_url", "embedding_text",
        ]
        for key in required:
            assert key in doc, f"Missing key: {key}"

    def test_no_embedding_vector_in_output(self):
        doc = preprocess_record(self._minimal_record())
        assert "embedding_vector" not in doc

    def test_null_description_handled(self):
        doc = preprocess_record(self._minimal_record(description=None))
        assert doc is not None
        assert isinstance(doc["description"], str)

    def test_string_description_handled(self):
        doc = preprocess_record(self._minimal_record(description="Plain string desc."))
        assert "Plain string desc." in doc["embedding_text"]


# ── iter_batches ──────────────────────────────────────────────────────────────

class TestIterBatches:
    def test_exact_multiple(self):
        items = list(range(6))
        batches = list(iter_batches(items, 3))
        assert batches == [[0, 1, 2], [3, 4, 5]]

    def test_remainder_batch(self):
        items = list(range(7))
        batches = list(iter_batches(items, 3))
        assert len(batches) == 3
        assert batches[-1] == [6]

    def test_single_item_list(self):
        batches = list(iter_batches([42], batch_size=10))
        assert batches == [[42]]

    def test_empty_list(self):
        assert list(iter_batches([], 5)) == []

    def test_batch_size_larger_than_list(self):
        items = [1, 2, 3]
        batches = list(iter_batches(items, 100))
        assert batches == [[1, 2, 3]]


# ── build_bulk_actions ────────────────────────────────────────────────────────

class TestBuildBulkActions:
    def _make_doc(self, doc_id: str = "B001") -> dict:
        return {
            "_id": doc_id,
            "title": "Test",
            "description": "Desc",
            "features": "Feat",
            "main_category": "Beauty",
            "sub_category": "",
            "store": "S",
            "price": 9.99,
            "average_rating": 4.0,
            "rating_number": 10,
            "primary_image_url": "",
            "embedding_text": "Test Feat Desc",
            "parent_asin": doc_id,
        }

    def test_yields_correct_number_of_actions(self):
        docs = [self._make_doc(f"B00{i}") for i in range(3)]
        vectors = [[0.1] * 384 for _ in range(3)]
        actions = list(build_bulk_actions(docs, vectors, "products"))
        assert len(actions) == 3

    def test_action_has_index_and_id(self):
        docs = [self._make_doc("B001")]
        vectors = [[0.0] * 384]
        actions = list(build_bulk_actions(docs, vectors, "products"))
        action = actions[0]
        assert action["_index"] == "products"
        assert action["_id"] == "B001"

    def test_source_contains_embedding_vector(self):
        docs = [self._make_doc("B001")]
        vector = [0.5] * 384
        actions = list(build_bulk_actions(docs, [vector], "products"))
        assert actions[0]["_source"]["embedding_vector"] == vector

    def test_source_does_not_contain_underscore_id(self):
        docs = [self._make_doc("B001")]
        actions = list(build_bulk_actions(docs, [[0.0] * 384], "products"))
        assert "_id" not in actions[0]["_source"]

    def test_index_name_passed_through(self):
        docs = [self._make_doc("B001")]
        actions = list(build_bulk_actions(docs, [[0.0] * 384], "my-custom-index"))
        assert actions[0]["_index"] == "my-custom-index"

    def test_empty_inputs_yield_no_actions(self):
        actions = list(build_bulk_actions([], [], "products"))
        assert actions == []
