#!/usr/bin/env python
"""
scripts/seed_eval_queries.py — Seed the eval_queries OpenSearch index.

Usage:
    # Seed from a JSONL file (one JSON object per line)
    python scripts/seed_eval_queries.py --queries-file eval_data/queries.jsonl

    # Generate a synthetic evaluation set from the product index
    python scripts/seed_eval_queries.py --synthetic --n 50

    # Show current eval set stats
    python scripts/seed_eval_queries.py --stats

Each query document format:
    {
      "query_id": "q001",
      "query_text": "moisturizer for oily skin",
      "relevant_asins": ["B001ABC", "B002DEF"],
      "mode": "hybrid"
    }

Exit codes: 0 = success, 1 = error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET}  {msg}")


def err(msg: str) -> None:
    print(f"  {RED}✗{RESET}  {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET}  {msg}")


def get_client():
    from config.settings import get_settings
    from opensearchpy import OpenSearch
    settings = get_settings()
    return OpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=False,
        timeout=30,
    )


def load_queries_from_file(path: str) -> list[dict]:
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def show_stats(client, evaluator) -> None:
    from mlops.evaluation.evaluator import EVAL_INDEX
    try:
        if not client.indices.exists(index=EVAL_INDEX):
            warn(f"Eval index '{EVAL_INDEX}' does not exist yet.")
            return
        resp = client.count(index=EVAL_INDEX)
        count = resp.get("count", 0)
        print(f"\n  {BOLD}Eval query set:{RESET}")
        print(f"    Documents : {count:,}")

        if count > 0:
            sample = client.search(
                index=EVAL_INDEX,
                body={"size": 3, "query": {"match_all": {}}, "_source": ["query_text"]},
            )
            print(f"    Sample queries:")
            for h in sample.get("hits", {}).get("hits", []):
                print(f"      - {h['_source'].get('query_text', '')}")
    except Exception as exc:
        err(f"Could not fetch stats: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed evaluation queries into the eval_queries OpenSearch index."
    )
    parser.add_argument(
        "--queries-file", metavar="PATH",
        help="Path to JSONL file with annotated queries"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate a synthetic eval set from the product index"
    )
    parser.add_argument(
        "--n", type=int, default=20,
        help="Number of synthetic queries to generate (default: 20)"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show eval query set statistics"
    )
    parser.add_argument(
        "--index", default=None,
        help="Product index name (overrides INDEX_NAME env var)"
    )
    args = parser.parse_args()

    if not any([args.queries_file, args.synthetic, args.stats]):
        parser.print_help()
        return 1

    from config.settings import get_settings
    from mlops.evaluation.evaluator import SearchEvaluator

    settings = get_settings()
    index_name = args.index or settings.index_name
    client = get_client()

    print(f"\n{BOLD}Würth Search — Eval Query Seeder{RESET}")
    print(f"  OpenSearch : {settings.opensearch_host}:{settings.opensearch_port}")
    print(f"  Index      : {index_name}\n")

    try:
        health = client.cluster.health()
        ok(f"Connected (cluster status: {health.get('status', '?')})")
    except Exception as exc:
        err(f"Cannot connect to OpenSearch: {exc}")
        return 1

    evaluator = SearchEvaluator(client, index_name)

    if args.stats:
        show_stats(client, evaluator)
        return 0

    queries: list[dict] = []

    if args.queries_file:
        try:
            queries = load_queries_from_file(args.queries_file)
            ok(f"Loaded {len(queries)} queries from {args.queries_file}")
        except Exception as exc:
            err(f"Failed to load queries: {exc}")
            return 1

    elif args.synthetic:
        print(f"  Generating {args.n} synthetic queries from top-rated products...")
        queries = evaluator._generate_synthetic_queries(n=args.n)
        if not queries:
            warn("Could not generate synthetic queries. Is the product index populated?")
            return 1
        ok(f"Generated {len(queries)} synthetic queries")

    if not queries:
        warn("No queries to seed.")
        return 1

    print(f"\n  Seeding {len(queries)} queries into eval_queries index...")
    indexed = evaluator.seed_eval_index(queries)
    ok(f"Seeded {indexed} queries successfully.")
    print(f"\n  Run {BOLD}make evaluate{RESET} to trigger offline evaluation.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
