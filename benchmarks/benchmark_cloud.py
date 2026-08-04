#!/usr/bin/env python
"""Cloud benchmark: DeepMem cloud vs Mem0 cloud.

Same workload against both: seed N conversations, then run M searches.
Measures add round-trip latency and search latency (P50/P95/P99/mean).

Run:
    HTTPS_PROXY=http://127.0.0.1:7897 python scripts/benchmark_cloud.py

Env:
    DEEPMEM_API_KEY   (dm_live_...)
    MEM0_API_KEY      (m0-...)
"""
import os
import statistics
import sys
import time
import uuid

import httpx

# ── config ──────────────────────────────────────────────────────────────
DEEPMEM_BASE = "https://deepmem.dev"
MEM0_BASE = "https://api.mem0.ai"
DEEPMEM_KEY = os.environ.get("DEEPMEM_API_KEY", "")
MEM0_KEY = os.environ.get("MEM0_API_KEY", "")
PROXY = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

# ── workload ────────────────────────────────────────────────────────────
SEED_CONVERSATIONS = [
    [{"role": "user", "content": "I'm Maria, I live in Lisbon and work as a product designer."},
     {"role": "assistant", "content": "Nice to meet you, Maria!"}],
    [{"role": "user", "content": "I prefer dark mode, use Figma daily, and drink espresso with no sugar."},
     {"role": "assistant", "content": "Noted your preferences."}],
    [{"role": "user", "content": "My favorite stack is Python + FastAPI for backends and React for frontends."},
     {"role": "assistant", "content": "Got it."}],
    [{"role": "user", "content": "I have a dog named Biscuit and a cat named Pixel."},
     {"role": "assistant", "content": "Cute names!"}],
    [{"role": "user", "content": "I'm allergic to shellfish and lactose intolerant."},
     {"role": "assistant", "content": "Important to remember."}],
    [{"role": "user", "content": "I train for half-marathons on weekends, usually near the Tagus river."},
     {"role": "assistant", "content": "Great discipline!"}],
    [{"role": "user", "content": "At work we deploy on AWS, primarily ECS and RDS Postgres."},
     {"role": "assistant", "content": "Noted your infra."}],
    [{"role": "user", "content": "I'm learning Japanese, currently around N4 level."},
     {"role": "assistant", "content": "Ganbatte!"}],
    [{"role": "user", "content": "My timezone is WET (UTC+0), and I do deep work mornings 9-12."},
     {"role": "assistant", "content": "Noted."}],
    [{"role": "user", "content": "I prefer async communication over meetings, and I dislike Slack notifications."},
     {"role": "assistant", "content": "Understood."}],
    [{"role": "user", "content": "I read sci-fi; favorite authors are Liu Cixin and Ursula Le Guin."},
     {"role": "assistant", "content": "Good taste."}],
    [{"role": "user", "content": "I use Neovim with Lua config on Arch Linux."},
     {"role": "assistant", "content": "Nice setup."}],
    [{"role": "user", "content": "My birthday is March 14, and my partner's name is Tomás."},
     {"role": "assistant", "content": "Noted!"}],
    [{"role": "user", "content": "I budget with YNAB and invest in low-cost index funds."},
     {"role": "assistant", "content": "Sensible."}],
    [{"role": "user", "content": "I have a meeting every Monday 10am with the design team called 'Design Sync'."},
     {"role": "assistant", "content": "On the calendar."}],
]
QUERIES = [
    "Where does Maria live?", "What does Maria do for work?", "What editor does Maria use?",
    "What's Maria's tech stack?", "What pets does Maria have?", "Any food allergies?",
    "What's Maria's exercise routine?", "Where does Maria deploy?", "What language is Maria learning?",
    "What's Maria's timezone?", "How does Maria prefer to communicate?", "What books does Maria like?",
    "What OS does Maria use?", "When is Maria's birthday?", "How does Maria manage money?",
    "What recurring meetings does Maria have?", "Maria's favorite coffee?", "Maria's partner's name?",
    "What database does Maria use?", "What design tool does Maria use?",
] * 2  # 40 searches


def pct(nums, p):
    if not nums:
        return float("nan")
    s = sorted(nums)
    k = max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))
    return s[k]


def fmt(nums):
    if not nums:
        return "n/a"
    return (f"mean={statistics.mean(nums):.0f}ms  "
            f"p50={pct(nums,50):.0f}ms  p95={pct(nums,95):.0f}ms  "
            f"p99={pct(nums,99):.0f}ms  n={len(nums)}")


def client():
    kw = {"timeout": 60.0}
    if PROXY:
        kw["proxy"] = PROXY
    return httpx.Client(**kw)


# ── DeepMem cloud (Mem0-compatible API) ─────────────────────────────────
def deepmem_add(c, user_id, messages):
    t0 = time.perf_counter()
    r = c.post(f"{DEEPMEM_BASE}/v1/memories",
               headers={"Authorization": f"Bearer {DEEPMEM_KEY}"},
               json={"messages": messages, "user_id": user_id, "infer": False})
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000


def _result_count(payload):
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        return len(payload.get("results") or payload.get("memories") or [])
    return 0


def deepmem_search(c, user_id, query):
    t0 = time.perf_counter()
    r = c.post(f"{DEEPMEM_BASE}/v1/memories/search",
               headers={"Authorization": f"Bearer {DEEPMEM_KEY}"},
               json={"query": query, "user_id": user_id, "top_k": 5})
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000, _result_count(r.json())


# ── Mem0 cloud ──────────────────────────────────────────────────────────
def mem0_add(c, user_id, messages):
    t0 = time.perf_counter()
    r = c.post(f"{MEM0_BASE}/v1/memories/",
               headers={"Authorization": f"Token {MEM0_KEY}"},
               json={"messages": messages, "user_id": user_id})
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000


def mem0_search(c, user_id, query):
    t0 = time.perf_counter()
    r = c.post(f"{MEM0_BASE}/v1/memories/search/",
               headers={"Authorization": f"Token {MEM0_KEY}"},
               json={"query": query, "user_id": user_id})
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000, _result_count(r.json())


def run_service(name, add_fn, search_fn):
    uid = f"bench_{name}_{uuid.uuid4().hex[:6]}"
    add_lat, search_lat = [], []
    with client() as c:
        # warmup add (discard) - first call pays proxy/TLS setup.
        try:
            add_fn(c, uid, SEED_CONVERSATIONS[0])
        except Exception:
            pass

        print(f"\n[{name}] seeding {len(SEED_CONVERSATIONS)} conversations as {uid}...")
        for msgs in SEED_CONVERSATIONS:
            try:
                add_lat.append(add_fn(c, uid, msgs))
            except Exception as e:
                print(f"  add error: {e}")
            time.sleep(0.3)

        # wait for async extraction to land
        print(f"[{name}] waiting for extraction to settle...")
        settled = False
        for _ in range(40):
            try:
                _, n = search_fn(c, uid, "Where does Maria live?")
                if n > 0:
                    settled = True
                    break
            except Exception:
                pass
            time.sleep(2)
        print(f"[{name}] settled={settled} after wait; warmup + measuring...")

        # warmup (discard) so the first measured call isn't penalized by
        # proxy/TLS setup.
        for q in QUERIES[:2]:
            try:
                search_fn(c, uid, q)
            except Exception:
                pass

        print(f"[{name}] running {len(QUERIES)} searches...")
        hits, failures = 0, 0
        for q in QUERIES:
            lat = None
            for attempt in range(2):  # retry once on 5xx
                try:
                    lat, n = search_fn(c, uid, q)
                    hits += n
                    break
                except httpx.HTTPStatusError as e:
                    if e.response.status_code >= 500 and attempt == 0:
                        time.sleep(0.8)
                        continue
                    failures += 1
                    break
                except Exception:
                    failures += 1
                    break
            if lat is not None:
                search_lat.append(lat)
            time.sleep(0.3)
        print(f"[{name}] done: {len(search_lat)} ok, {failures} failed")
    return add_lat, search_lat, hits


def main():
    if not DEEPMEM_KEY or not MEM0_KEY:
        print("ERROR: set DEEPMEM_API_KEY and MEM0_API_KEY env vars.")
        sys.exit(1)
    print(f"proxy={PROXY or 'none'}")
    print(f"seed={len(SEED_CONVERSATIONS)} conversations, queries={len(QUERIES)}")

    dm_add, dm_search, dm_hits = run_service("DeepMem", deepmem_add, deepmem_search)
    m0_add, m0_search, m0_hits = run_service("Mem0", mem0_add, mem0_search)

    print("\n" + "=" * 64)
    print(f"{'metric':<22}{'DeepMem cloud':<26}{'Mem0 cloud':<26}")
    print("-" * 64)
    print(f"{'add round-trip':<22}{fmt(dm_add):<26}{fmt(m0_add):<26}")
    print(f"{'search latency':<22}{fmt(dm_search):<26}{fmt(m0_search):<26}")
    print(f"{'total search hits':<22}{dm_hits:<26}{m0_hits:<26}")
    print("=" * 64)
    if dm_search and m0_search:
        sp = statistics.mean(m0_search) / statistics.mean(dm_search)
        print(f"Mem0 search mean / DeepMem search mean = {sp:.2f}x")


if __name__ == "__main__":
    main()
