# DeepMem Benchmarks

Reproducible latency measurement for the DeepMem memory layer. The goal is
simple and a little unusual for this space: **if you don't trust our numbers,
run the script yourself and produce your own.**

> Think our latency claims are inflated? Register your own API keys (or point
> the script at your own DeepMem instance) and read your own percentiles.
> Every script here ships a self-contained workload, so there's nothing to
> configure but credentials.

There are two benchmarks:

| Script | What it compares | Needs |
|--------|------------------|-------|
| [`benchmark_cloud.py`](benchmark_cloud.py) | **DeepMem cloud vs Mem0 cloud** (hosted vs hosted) | API keys from both services (register below) |
| [`run_benchmark.py`](run_benchmark.py) | **Your DeepMem instance** (self-hosted, your hardware) | A running DeepMem server, no API key |

---

## Cloud comparison — DeepMem cloud vs Mem0 cloud

The easiest way to verify DeepMem against the incumbent: run the same workload
against both hosted services and compare. **You bring the API keys** — we
don't bake ours in, so the numbers are yours.

### 1. Register and get API keys

You need an account on each service. Both have free tiers.

**DeepMem** (this project's hosted API):
1. Go to **https://deepmem.dev** → sign up.
2. In your dashboard, create an API key. It looks like `dm_live_...`.
3. `export DEEPMEM_API_KEY=dm_live_your_key_here`

**Mem0** (the incumbent, for comparison):
1. Go to **https://mem0.ai** → sign up.
2. In the dashboard, create an API key. It looks like `m0-...`.
3. `export MEM0_API_KEY=m0_your_key_here`

### 2. Run

```bash
export DEEPMEM_API_KEY=dm_live_...
export MEM0_API_KEY=m0-...

# Optional — only if you're behind a network that needs a proxy to reach the
# clouds (e.g. mainland China). httpx honors HTTPS_PROXY.
# export HTTPS_PROXY=http://127.0.0.1:7897

python benchmarks/benchmark_cloud.py
```

The script seeds both services with the same 15 conversations, waits for
extraction to settle, warms up, then runs 40 searches against each and
reports P50/P95/P99 for add and search.

### 3. Read the output

```
================================================================
metric                DeepMem cloud             Mem0 cloud
----------------------------------------------------------------
add round-trip        mean=811ms p50=792 p95=851 p99=1283  n=15   mean=713ms p50=695 p95=777 p99=799  n=15
search latency        mean=674ms p50=643 p95=811 p99=1157  n=39   mean=649ms p50=653 p95=710 p99=849  n=40
total search hits     195                       84
================================================================
Mem0 search mean / DeepMem search mean = 0.96x
```

### Honest caveats (read before quoting a number)

1. **Network dominates.** Both clouds are reached over the public internet
   (and through your proxy if you set one). RTT can be 100–400ms per call,
   which swamps the services' intrinsic processing time. A 0.9x / 1.1x
   difference from a proxied run is **noise, not a win**.
2. **For intrinsic latency, run from a low-RTT location.** Spin up a small
   VPS in the same region as the clouds (or self-host DeepMem on that VPS)
   and run the script there without a proxy. That's the only fair way to
   compare hosted latency.
3. **Add isn't apples-to-apples.** DeepMem `add(infer=False)` stores raw
   (sync); Mem0 `add` always runs LLM extraction. The `add` row is reported
   for completeness but the **search row is the fair comparison**.
4. **DeepMem returns more candidates** (`total search hits` higher) because
   it fills `top_k`; Mem0 returns fewer, more-filtered results. That's a
   recall/precision design choice, not a latency difference.

---

## Self-hosted benchmark — your DeepMem, your hardware

Measures end-to-end add / search latency against a DeepMem server you run,
and prints P50 / P95 / P99 + throughput + hit-rate. No API key, no external
service — this is the path that produces a "45ms P95"-class number, because
there's no internet RTT.

```bash
# 1. start the server (any embedder you've configured)
python server/start.py --no-mcp

# 2. run the benchmark — default 200 ops, concurrency 8, raw pipeline
python benchmarks/run_benchmark.py

# heavier run, full LLM extraction path
python benchmarks/run_benchmark.py --ops 500 --concurrency 16 --infer

# against a remote deployment
DEEPMEM_BASE_URL=https://your-deepmem.example.com python benchmarks/run_benchmark.py
```

Results print to the console and are written to `benchmarks/results/result_<ts>.json`,
stamped with the config that produced them (`ops`, `concurrency`, `infer`, `base_url`).

### What's measured

| Phase | What it exercises | What it isolates |
|-------|-------------------|------------------|
| `add` (`infer=False`, default) | embed + Qdrant upsert + dedup + history | the vector storage pipeline every backend must pay |
| `search` | embed query + hybrid scoring (vector + BM25 + entity boost + time-decay) | the retrieval path |
| `add` (`--infer`) | the above **plus** LLM fact extraction (batched distiller) | the full extraction path — depends on YOUR LLM plan |

### Caveats

1. **Latency is dominated by the embedder.** BGE-M3 on CPU, BGE-M3 on GPU, and
   Google/OpenAI embeddings give wildly different numbers. Always state which
   `EMBEDDING_PROVIDER` and model produced a result.
2. **`--infer` latency depends on your LLM plan and network.** The fair
   cross-backend comparison is `infer=False` (storage/retrieval only).
3. **Throughput is bounded by `--concurrency` and your Qdrant mode** (local
   file vs. remote server). Use a real Qdrant server for production-grade
   numbers.

---

## Adapter harness (roll your own comparison)

Both benchmarks talk to backends through a small adapter protocol
([`adapters.py`](adapters.py)):

```python
class MemoryAdapter(Protocol):
    name: str
    async def add(self, messages, user_id, infer=False) -> None: ...
    async def search(self, query, user_id, top_k=5) -> list[dict]: ...
    async def close(self) -> None: ...
```

Implemented adapters:
- `DeepMemAdapter` — self-hosted DeepMem (HTTP, open mode).
- `DeepMemCloudAdapter` — DeepMem hosted API (`dm_live_...`).
- `Mem0CloudAdapter` — Mem0 hosted API (`m0-...`).
- `ZepAdapter` — stub (install `zep-cloud` and wire it for a three-way run).

`benchmark_cloud.py` uses the two cloud adapters directly. To run the full
harness (`run_benchmark.py`) against a specific backend, instantiate the
adapter you want and pass it in — the workload, percentile math, and
reporting are backend-agnostic.

### Using a real benchmark dataset

The default workload is synthetic (user-profile facts + matching queries),
enough to exercise the pipeline realistically. To run against a published
memory benchmark:

- **LongMemEval** — multi-session long-term memory evaluation.
- **LoCoMo** — long conversation memory dataset.

Download either, convert each item to the `(messages, user_id)` /
`(query, user_id)` shape `build_workload` emits, and pass it in.

---

## Files

```
benchmarks/
├── README.md            # this file
├── benchmark_cloud.py   # DeepMem cloud vs Mem0 cloud (needs API keys)
├── run_benchmark.py     # self-hosted DeepMem latency (no key needed)
├── adapters.py          # MemoryAdapter protocol + DeepMem/Mem0 cloud impls
└── results/             # gitignored — your run outputs land here
```
