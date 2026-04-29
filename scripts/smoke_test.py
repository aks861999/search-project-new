#!/usr/bin/env python
"""
Smoke test — exercises every API endpoint against a running stack.

Usage:
    python scripts/smoke_test.py [--base-url http://localhost:8000]

Exit codes:
    0 — all checks passed
    1 — one or more checks failed

Run after:
    make up
    make ingest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

try:
    import httpx
except ImportError:
    print("httpx not installed. Run: pip install httpx")
    sys.exit(1)

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

passed = 0
failed = 0


def _check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  {GREEN}✓{RESET}  {label}")
    else:
        failed += 1
        print(f"  {RED}✗{RESET}  {label}" + (f"  ({detail})" if detail else ""))


def _json(r: httpx.Response) -> dict:
    try:
        return r.json()
    except Exception:
        return {}


def run_smoke_tests(base: str) -> None:
    print(f"\n{BOLD}Würth Search API — Smoke Test{RESET}")
    print(f"Base URL: {base}\n")

    client = httpx.Client(base_url=base, timeout=30)

    # ── 1. Health check ────────────────────────────────────────────────────────
    print(f"{BOLD}[1] GET /health{RESET}")
    try:
        r = client.get("/health")
        d = _json(r)
        _check("HTTP 200", r.status_code == 200, str(r.status_code))
        _check("status field present", "status" in d)
        _check("opensearch field present", "opensearch" in d)
        _check("redis field present", "redis" in d)
        _check(f"opensearch reachable ({d.get('opensearch')})",
               d.get("opensearch") in ("green", "yellow"))
        _check(f"redis reachable ({d.get('redis')})", d.get("redis") == "ok")
    except Exception as e:
        _check("health endpoint reachable", False, str(e))

    # ── 2. Hybrid search ──────────────────────────────────────────────────────
    print(f"\n{BOLD}[2] GET /search?mode=hybrid{RESET}")
    try:
        r = client.get("/search", params={"q": "moisturizer", "mode": "hybrid", "size": 5})
        d = _json(r)
        _check("HTTP 200", r.status_code == 200, str(r.status_code))
        _check("total field present", "total" in d)
        _check("hits list present", isinstance(d.get("hits"), list))
        _check("took_ms present", "took_ms" in d)
        _check("mode == hybrid", d.get("mode") == "hybrid")
        if d.get("hits"):
            hit = d["hits"][0]
            _check("hit has title", bool(hit.get("title")))
            _check("hit has id", bool(hit.get("id")))
            _check("hit has score", isinstance(hit.get("score"), (int, float)))
    except Exception as e:
        _check("hybrid search reachable", False, str(e))

    # ── 3. Lexical search ─────────────────────────────────────────────────────
    print(f"\n{BOLD}[3] GET /search?mode=lexical{RESET}")
    try:
        r = client.get("/search", params={
            "q": "shampoo", "mode": "lexical", "size": 3,
            "max_price": "50", "min_rating": "3"
        })
        d = _json(r)
        _check("HTTP 200", r.status_code == 200, str(r.status_code))
        _check("mode == lexical", d.get("mode") == "lexical")
        _check("hits list present", isinstance(d.get("hits"), list))
    except Exception as e:
        _check("lexical search reachable", False, str(e))

    # ── 4. Semantic search ────────────────────────────────────────────────────
    print(f"\n{BOLD}[4] GET /search?mode=semantic{RESET}")
    try:
        r = client.get("/search", params={"q": "anti-aging face serum", "mode": "semantic", "size": 3})
        d = _json(r)
        _check("HTTP 200", r.status_code == 200, str(r.status_code))
        _check("mode == semantic", d.get("mode") == "semantic")
    except Exception as e:
        _check("semantic search reachable", False, str(e))

    # ── 5. NL search ──────────────────────────────────────────────────────────
    print(f"\n{BOLD}[5] POST /nl-search{RESET}")
    try:
        r = client.post("/nl-search", json={
            "query": "cheap organic shampoo for dry hair under $20",
            "size": 5
        })
        d = _json(r)
        _check("HTTP 200", r.status_code == 200, str(r.status_code))
        _check("hits list present", isinstance(d.get("hits"), list))
        _check("parsed_query present", "parsed_query" in d)
        pq = d.get("parsed_query", {})
        _check("parsed_query.semantic_query non-empty", bool(pq.get("semantic_query")))
        _check("parsed_query.filters present", "filters" in pq)
        filters = pq.get("filters", {})
        _check("price_max extracted (≤20)", filters.get("price_max") is not None
               and float(filters["price_max"]) <= 20.0)
    except Exception as e:
        _check("NL search reachable", False, str(e))

    # ── 6. Suggest ────────────────────────────────────────────────────────────
    print(f"\n{BOLD}[6] GET /suggest{RESET}")
    try:
        r = client.get("/suggest", params={"prefix": "moist", "size": 5})
        d = _json(r)
        _check("HTTP 200", r.status_code == 200, str(r.status_code))
        _check("suggestions list present", isinstance(d.get("suggestions"), list))
        _check("prefix echoed back", d.get("prefix") == "moist")
    except Exception as e:
        _check("suggest reachable", False, str(e))

    # ── 7. Cache — second identical request should be faster ──────────────────
    print(f"\n{BOLD}[7] Redis cache — repeated request{RESET}")
    try:
        params = {"q": "vitamin c serum", "mode": "hybrid", "size": 5}
        client.get("/search", params=params)  # warm cache
        t0 = time.perf_counter()
        r2 = client.get("/search", params=params)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        d2 = _json(r2)
        _check("HTTP 200 on cached request", r2.status_code == 200)
        _check("cached=true flag set", d2.get("cached") is True,
               f"cached={d2.get('cached')}")
        _check(f"cached response < 50ms ({elapsed_ms:.1f}ms)", elapsed_ms < 50)
    except Exception as e:
        _check("cache check reachable", False, str(e))

    # ── 8. Pagination ─────────────────────────────────────────────────────────
    print(f"\n{BOLD}[8] Pagination (from=5){RESET}")
    try:
        r_p1 = client.get("/search", params={"q": "moisturizer", "size": 5, "from": 0})
        r_p2 = client.get("/search", params={"q": "moisturizer", "size": 5, "from": 5})
        d1, d2 = _json(r_p1), _json(r_p2)
        ids1 = {h["id"] for h in d1.get("hits", [])}
        ids2 = {h["id"] for h in d2.get("hits", [])}
        _check("page 1 HTTP 200", r_p1.status_code == 200)
        _check("page 2 HTTP 200", r_p2.status_code == 200)
        _check("pages are disjoint", len(ids1 & ids2) == 0,
               f"overlap={ids1 & ids2}")
    except Exception as e:
        _check("pagination reachable", False, str(e))

    # ── 9. Validation errors ──────────────────────────────────────────────────
    print(f"\n{BOLD}[9] Input validation{RESET}")
    try:
        r = client.get("/search", params={"q": "", "mode": "hybrid"})
        _check("empty q returns 422", r.status_code == 422, str(r.status_code))

        r2 = client.get("/search", params={"q": "test", "mode": "invalid_mode"})
        _check("invalid mode returns 422", r2.status_code == 422, str(r2.status_code))

        r3 = client.get("/search", params={"q": "test", "min_price": "30", "max_price": "5"})
        _check("max_price < min_price returns 422", r3.status_code == 422, str(r3.status_code))
    except Exception as e:
        _check("validation errors reachable", False, str(e))

    # ── Summary ───────────────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{'─'*50}")
    print(f"{BOLD}Results: {GREEN}{passed} passed{RESET}{BOLD}, "
          f"{RED if failed else ''}{failed} failed{RESET} / {total} total")
    print(f"{'─'*50}\n")

    client.close()
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Würth Search API smoke test")
    parser.add_argument(
        "--base-url", default="http://localhost:8000",
        help="Base URL of the running API (default: http://localhost:8000)"
    )
    args = parser.parse_args()
    failures = run_smoke_tests(args.base_url)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
