#!/usr/bin/env python
"""
create_index.py — Create the OpenSearch products index and register the
hybrid search pipeline, without running a full ingestion.

Use this when you want to:
  - Recreate a dropped / corrupted index
  - Reset the index schema after a mapping change (add --recreate)
  - Set up the pipeline only, without touching the index

Usage:
    python scripts/create_index.py
    python scripts/create_index.py --recreate          # drop + recreate
    python scripts/create_index.py --pipeline-only     # pipeline only
    python scripts/create_index.py --host localhost --port 9200

Exit codes:
    0 — success
    1 — error
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError

from config.settings import get_settings
from ingestion.schema import (
    HYBRID_PIPELINE_ID,
    HYBRID_SEARCH_PIPELINE,
    INDEX_SETTINGS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET}  {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET}  {msg}")


def err(msg: str) -> None:
    print(f"  {RED}✗{RESET}  {msg}")


# ── Core operations ───────────────────────────────────────────────────────────

def build_client(host: str, port: int, settings) -> OpenSearch:
    http_auth = None
    if settings.opensearch_user and settings.opensearch_password:
        http_auth = (settings.opensearch_user, settings.opensearch_password)
    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=http_auth,
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=settings.opensearch_verify_certs,
        ssl_show_warn=False,
        timeout=30,
    )


def create_index(client: OpenSearch, index_name: str, recreate: bool = False) -> bool:
    """
    Create the index.  If recreate=True, delete first.
    Returns True if the index was (re)created, False if it already existed
    and recreate=False.
    """
    exists = client.indices.exists(index=index_name)

    if exists and not recreate:
        warn(f"Index '{index_name}' already exists — skipping (use --recreate to reset).")
        return False

    if exists and recreate:
        logger.info("Deleting existing index '%s'...", index_name)
        client.indices.delete(index=index_name)
        ok(f"Index '{index_name}' deleted.")

    logger.info("Creating index '%s'...", index_name)
    response = client.indices.create(index=index_name, body=INDEX_SETTINGS)
    if response.get("acknowledged"):
        ok(f"Index '{index_name}' created successfully.")
        return True
    else:
        err(f"Index creation returned unexpected response: {response}")
        return False


def register_pipeline(client: OpenSearch) -> bool:
    """Register (or update) the hybrid search pipeline. Returns True on success."""
    logger.info("Registering hybrid search pipeline '%s'...", HYBRID_PIPELINE_ID)
    try:
        client.transport.perform_request(
            "PUT",
            f"/_search/pipeline/{HYBRID_PIPELINE_ID}",
            body=HYBRID_SEARCH_PIPELINE,
        )
        ok(f"Pipeline '{HYBRID_PIPELINE_ID}' registered.")
        return True
    except Exception as exc:
        err(f"Pipeline registration failed: {exc}")
        return False


def show_index_stats(client: OpenSearch, index_name: str) -> None:
    """Print a summary of the index (doc count, size)."""
    try:
        stats = client.indices.stats(index=index_name)
        idx = stats.get("indices", {}).get(index_name, {})
        total = idx.get("total", {})
        doc_count = total.get("docs", {}).get("count", "?")
        size_bytes = total.get("store", {}).get("size_in_bytes", 0)
        size_mb = size_bytes / (1024 * 1024)
        print(f"\n  {BOLD}Index summary:{RESET}")
        print(f"    Documents : {doc_count:,}")
        print(f"    Size      : {size_mb:.1f} MB")
    except Exception as exc:
        warn(f"Could not fetch index stats: {exc}")


def show_pipeline(client: OpenSearch) -> None:
    """Print the registered pipeline configuration."""
    try:
        resp = client.transport.perform_request(
            "GET", f"/_search/pipeline/{HYBRID_PIPELINE_ID}"
        )
        print(f"\n  {BOLD}Hybrid pipeline config:{RESET}")
        print("  " + json.dumps(resp, indent=2).replace("\n", "\n  "))
    except NotFoundError:
        warn(f"Pipeline '{HYBRID_PIPELINE_ID}' not found.")
    except Exception as exc:
        warn(f"Could not fetch pipeline: {exc}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create OpenSearch index and hybrid search pipeline."
    )
    parser.add_argument("--host", default=None, help="OpenSearch host (overrides settings)")
    parser.add_argument("--port", type=int, default=None, help="OpenSearch port")
    parser.add_argument("--index", default=None, help="Index name (overrides INDEX_NAME env var)")
    parser.add_argument(
        "--recreate", action="store_true",
        help="Delete and recreate the index (WARNING: all data will be lost)"
    )
    parser.add_argument(
        "--pipeline-only", action="store_true",
        help="Register the hybrid pipeline only, skip index creation"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show index stats after creation"
    )
    args = parser.parse_args()

    settings = get_settings()
    host = args.host or settings.opensearch_host
    port = args.port or settings.opensearch_port
    index_name = args.index or settings.index_name

    print(f"\n{BOLD}Würth Search — Index Setup{RESET}")
    print(f"  OpenSearch : {host}:{port}")
    print(f"  Index      : {index_name}\n")

    client = build_client(host, port, settings)

    # Verify connectivity
    try:
        health = client.cluster.health()
        ok(f"Connected to OpenSearch (cluster status: {health.get('status', '?')})")
    except Exception as exc:
        err(f"Cannot connect to OpenSearch at {host}:{port} — {exc}")
        print("\n  Tip: run 'make up' to start the Docker stack first.\n")
        return 1

    errors = 0

    if not args.pipeline_only:
        success = create_index(client, index_name, recreate=args.recreate)
        if success is False and not client.indices.exists(index=index_name):
            errors += 1

    pipeline_ok = register_pipeline(client)
    if not pipeline_ok:
        errors += 1

    if args.stats and client.indices.exists(index=index_name):
        show_index_stats(client, index_name)

    show_pipeline(client)

    print()
    if errors == 0:
        ok("All done — index and pipeline are ready.")
        print(f"\n  Next step: run {BOLD}make ingest{RESET} to load the dataset.\n")
        return 0
    else:
        err(f"{errors} step(s) failed — check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
