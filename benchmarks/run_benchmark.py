#!/usr/bin/env python
"""DeepMem benchmark - reproducible latency measurement.

Measures end-to-end add / search latency against a running DeepMem server
and prints P50 / P95 / P99 + throughput. Self-contained: ships a synthetic
corpus so `git clone && python benchmarks/run_benchmark.py` works with no
external dataset, no API key, no model download beyond what your server is
already configured with.

What this measures (and what it does NOT):
  - infer=False (default): the vector pipeline only - embed + Qdrant upsert
    on add, embed + hybrid (vector + BM25 + time-decay) retrieval on search.
    This isolates the storage/retrieval layer Mem0 and Zep would also have
    to pay, and is the number to compare across backends.
  - infer=True (--infer): also fires the DeepSeek LLM fact-extraction path
    on add. Latency then depends on YOUR DeepSeek plan and network; we report
    it but do not claim it as a universal number.
  - We do NOT report a single "45ms P95" headline here - run it on your own
    hardware and read your own percentiles. Anything posted in the README is
    accompanied by the exact config that produced it.

Usage:
    # 1. start the server in another shell
    python server/start.py --no-mcp

    # 2. run the benchmark (default: 200 ops, concurrency 8, infer=False)
    python benchmarks/run_benchmark.py

    # heavier run, full LLM path
    python benchmarks/run_benchmark.py --ops 500 --concurrency 16 --infer

    # point at a remote server
    DEEPMEM_BASE_URL=https://my-deepmem.example.com python benchmarks/run_benchmark.py

Requires: httpx (already a project dependency).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

# Make `import adapters` work when run as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapters import DeepMemAdapter  # noqa: E402


# ── Synthetic, self-contained workload ────────────────────────────────
# Durable user-profile facts + matching queries. Good enough to exercise the
# pipeline realistically; swap in LongMemEval / LoCoMo via --dataset if you
# have them downloaded (see benchmarks/README.md).

_TOPICS = ["Python", "rust", "machine learning", "rock climbing", "jazz piano",
           "espresso", "cycle touring", "linear algebra", "sourdough", "type theory"]
_CITIES = ["Tokyo", "Lisbon", "Berlin", "Taipei", "Marseille", "Edinburgh",
           "Mexico City", "Seoul", "Reykjavik", "Cape Town"]


def build_workload(n: int):
    """Return (adds, searches) lists of (messages, user_id) / (query, user_id).

    Each user gets one add (a profile fact) and one search (a question whose
    answer is that fact), so a correct backend returns a hit on every search.
    """
    adds, searches = [], []
    for i in range(n):
        user_id = f"bench_user_{i}"
        topic = _TOPICS[i % len(_TOPICS)]
        city = _CITIES[i % len(_CITIES)]
        fact = f"My name is Pat{i}. I live in {city} and I'm into {topic}."
        adds.append(([{"role": "user", "content": fact}], user_id))
        searches.append((f"Where does Pat{i} live?", user_id))
    return adds, searches


# ── Stats ─────────────────────────────────────────────────────────────

def percentile(sorted_samples: list, p: float) -> float:
    """Linear-interpolation percentile on an already-sorted list."""
    if not sorted_samples:
        return 0.0
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    k = (len(sorted_samples) - 1) * p
    f, c = int(k), min(int(k) + 1, len(sorted_samples) - 1)
    if f == c:
        return sorted_samples[f]
    return sorted_samples[f] + (sorted_samples[c] - sorted_samples[f]) * (k - f)


def summarize(samples_ms: list) -> dict:
    s = sorted(samples_ms)
    return {
        "count": len(s),
        "p50_ms": round(percentile(s, 0.50), 2),
        "p95_ms": round(percentile(s, 0.95), 2),
        "p99_ms": round(percentile(s, 0.99), 2),
        "mean_ms": round(statistics.fmean(s), 2) if s else 0.0,
        "min_ms": round(s[0], 2) if s else 0.0,
        "max_ms": round(s[-1], 2) if s else 0.0,
    }


# ── Runner ────────────────────────────────────────────────────────────

async def run_phase(adapter, op_kind: str, items, concurrency: int,
                    infer: bool):
    """Run all items through the adapter with a bounded concurrency pool.

    Returns (latencies_ms, hit_count) for search; latencies only for add.
    """
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    hits = 0

    async def do_add(messages, user_id):
        async with sem:
            t0 = time.perf_counter()
            await adapter.add(messages, user_id, infer=infer)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    async def do_search(query, user_id):
        nonlocal hits
        async with sem:
            t0 = time.perf_counter()
            results = await adapter.search(query, user_id, top_k=5)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if results:
                hits += 1

    if op_kind == "add":
        await asyncio.gather(*[do_add(m, u) for (m, u) in items])
    else:
        await asyncio.gather(*[do_search(q, u) for (q, u) in items])
    return latencies, hits


async def main_async(args):
    base_url = os.environ.get("DEEPMEM_BASE_URL", args.base_url)
    adapter = DeepMemAdapter(base_url=base_url, timeout=args.timeout)
    print(f"DeepMem benchmark -> {base_url}  (backend={adapter.name})")
    print(f"  ops={args.ops}  concurrency={args.concurrency}  infer={args.infer}")

    # Sanity ping - fail fast with a clear message if the server is down.
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{base_url}/health")
            r.raise_for_status()
    except Exception as e:
        print(f"\nERROR: cannot reach DeepMem at {base_url} ({e}).\n"
              f"Start it first:  python server/start.py --no-mcp")
        return 1

    adds, searches = build_workload(args.ops)

    # Every user writes into a fresh partition so re-runs don't dedup-skip.
    run_id = f"{int(time.time())}"
    adds = [(m, f"{u}_{run_id}") for (m, u) in adds]
    searches = [(q, f"{u}_{run_id}") for (q, u) in searches]

    print(f"\n[1/2] add  ({len(adds)} writes, infer={args.infer})...")
    t0 = time.perf_counter()
    add_lat, _ = await run_phase(adapter, "add", adds, args.concurrency, args.infer)
    add_wall = time.perf_counter() - t0

    print(f"[2/2] search ({len(searches)} queries)...")
    t0 = time.perf_counter()
    search_lat, hits = await run_phase(adapter, "search", searches,
                                       args.concurrency, args.infer)
    search_wall = time.perf_counter() - t0

    await adapter.close()

    add_stats = summarize(add_lat)
    search_stats = summarize(search_lat)
    search_stats["hit_rate"] = round(hits / max(1, len(searches)), 3)

    # ── Report ──
    print("\n" + "=" * 56)
    print(f"{'DeepMem benchmark':^56}")
    print("=" * 56)
    print(f"  backend     : {adapter.name}")
    print(f"  base_url    : {base_url}")
    print(f"  ops         : {args.ops}  (concurrency={args.concurrency}, infer={args.infer})")
    print("-" * 56)
    print(f"  ADD    p50={add_stats['p50_ms']:>7.2f}ms  p95={add_stats['p95_ms']:>7.2f}ms  "
          f"p99={add_stats['p99_ms']:>7.2f}ms  mean={add_stats['mean_ms']:>7.2f}ms")
    print(f"         throughput={len(add_lat)/max(add_wall,1e-9):>6.1f} ops/s  "
          f"wall={add_wall:.2f}s")
    print(f"  SEARCH p50={search_stats['p50_ms']:>7.2f}ms  p95={search_stats['p95_ms']:>7.2f}ms  "
          f"p99={search_stats['p99_ms']:>7.2f}ms  mean={search_stats['mean_ms']:>7.2f}ms")
    print(f"         throughput={len(search_lat)/max(search_wall,1e-9):>6.1f} ops/s  "
          f"hit_rate={search_stats['hit_rate']:.3f}  wall={search_wall:.2f}s")
    print("=" * 56)

    # Persist machine-readable results.
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"result_{run_id}.json"
    payload = {
        "backend": adapter.name,
        "base_url": base_url,
        "ops": args.ops,
        "concurrency": args.concurrency,
        "infer": args.infer,
        "add": add_stats,
        "search": search_stats,
        "add_wall_s": round(add_wall, 3),
        "search_wall_s": round(search_wall, 3),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nresults written: {out_path}")
    return 0


def main():
    p = argparse.ArgumentParser(description="DeepMem latency benchmark")
    p.add_argument("--ops", type=int, default=200,
                   help="number of add+search pairs (default 200)")
    p.add_argument("--concurrency", type=int, default=8,
                   help="max in-flight requests (default 8)")
    p.add_argument("--infer", action="store_true",
                   help="enable LLM fact extraction on add (default off: raw pipeline)")
    p.add_argument("--base-url", default="http://localhost:8000",
                   help="DeepMem server URL (or set DEEPMEM_BASE_URL)")
    p.add_argument("--timeout", type=float, default=60.0,
                   help="per-request timeout seconds (default 60)")
    args = p.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
