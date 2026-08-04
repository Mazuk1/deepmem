#!/usr/bin/env python
"""Functional test - calls the DeepMem engine functions DIRECTLY with real
data and prints the actual inputs/outputs. This is not a pytest assertion
suite; it's the "does it really work?" script you can read and run to see
real extraction / search / CRUD behavior end to end.

Run:
    python scripts/func_test_direct.py
    BGE_DEVICE=cpu python scripts/func_test_direct.py   # force CPU if GPU contends

Uses a throwaway temp Qdrant store and cleans up. infer=True calls go to the
real DeepSeek API (fact extraction); the rest use infer=False (raw storage)
so the script is fast and mostly offline.
"""
import asyncio
import os
import shutil
import sys
import tempfile
import time
import uuid

# Make `import deepmem` work when run as a standalone script from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Isolate from the dev ./data/qdrant (which may have a stale dim-mismatched
# collection). Fresh temp store matches whatever embedder is configured.
_QDRANT = os.path.join(tempfile.gettempdir(), f"deepmem_functest_{uuid.uuid4().hex[:8]}")
os.environ.setdefault("QDRANT_PATH", _QDRANT)


def _hr(title):
    print(f"\n{'─' * 60}\n{title}\n{'─' * 60}")


async def main():
    from deepmem.config import config
    from deepmem.vector_store import VectorStore
    from deepmem.interface import Tenant

    print(f"embedder={config.default_embedder}  qdrant={_QDRANT}  "
          f"BGE_DEVICE={os.environ.get('BGE_DEVICE', 'auto')}")
    store = VectorStore(
        qdrant_path=_QDRANT,
        embedding_dims=config.embedding_dims,
        bge_m3_path=config.bge_m3_path,
    )

    # ── 1. add(infer=True): real DeepSeek fact extraction ────────────────
    _hr("1. add(infer=True) — DeepSeek extracts facts from a conversation")
    tenant = Tenant(user_id="user_alice",
                    owner_id="agent-1", visibility_scope="team",
                    source_type="user", session_id="sess-1")
    msgs = [
        {"role": "user", "content": "I'm Zhang San, I live in Beijing, I'm a backend engineer, mainly write Go, and I like hiking on weekends."},
        {"role": "assistant", "content": "Got it Zhang San, noted."},
    ]
    t0 = time.perf_counter()
    ids = await store.add(msgs, tenant, infer=True)
    print(f"extracted {len(ids)} fact(s) in {time.perf_counter()-t0:.1f}s, ids={ids[:3]}")
    for mid in ids:
        rec = await store.get(mid, tenant)
        print(f"  • {rec.memory}")

    # ── 2. add(infer=False): raw storage, immediate ──────────────────────
    _hr("2. add(infer=False) — raw storage, no LLM")
    tenant_b = Tenant(user_id="user_bob")
    raw_ids = await store.add(
        [{"role": "user", "content": "Bob prefers dark mode and uses Neovim."}],
        tenant_b, infer=False)
    print(f"stored {len(raw_ids)} raw memory(ies): {raw_ids}")

    # ── 3. search: hybrid retrieval ──────────────────────────────────────
    _hr("3. search — hybrid (vector + BM25 + entity + time-decay)")
    for q in ["Where does Zhang San live?", "What language does Zhang San use?", "Bob likes what editor?"]:
        t0 = time.perf_counter()
        hits = await store.search(q, tenant if "Zhang San" in q else tenant_b,
                                  top_k=3, threshold=0.2, hybrid=True)
        print(f"  q='{q}'  → {len(hits)} hit(s) in {time.perf_counter()-t0:.3f}s")
        for h in hits:
            print(f"      [{h.score:.3f}] {h.memory}")

    # ── 4. groundwork fields really landed in the payload ────────────────
    _hr("4. groundwork fields (owner_id / visibility_scope / source_type / session_id)")
    pt = store._client.retrieve(collection_name=store.collection_name,
                                ids=[ids[0]], with_payload=True)[0]
    for f in ("owner_id", "visibility_scope", "source_type", "session_id"):
        print(f"  {f} = {pt.payload.get(f)!r}")

    # ── 5. update + list + get ───────────────────────────────────────────
    _hr("5. update / list / get")
    ok = await store.update(ids[0], "Zhang San lives in Beijing and is a backend engineer.", tenant)
    print(f"update({ids[0][:8]}…) → {ok}")
    listed = await store.list(tenant, limit=10)
    print(f"list(user_alice) → {len(listed)} memories (newest first)")
    got = await store.get(ids[0], tenant)
    print(f"get({ids[0][:8]}…) → '{got.memory}'")

    # ── 6. export / import round-trip ────────────────────────────────────
    _hr("6. export / import round-trip")
    exported = await store.export(tenant)
    print(f"export(user_alice) → {len(exported)} records")
    tenant_c = Tenant(user_id="user_carol")  # import under a different user
    res = await store.import_records(tenant_c, exported, skip_existing=False)
    print(f"import → inserted={res['inserted']} skipped={res['skipped']}")
    listed_c = await store.list(tenant_c, limit=10)
    print(f"list(user_carol) → {len(listed_c)} memories (round-trip OK)")

    # ── 7. delete + delete_all (soft) ────────────────────────────────────
    _hr("7. delete (soft) + delete_all")
    ok = await store.delete(ids[0], tenant)
    print(f"delete({ids[0][:8]}…) → {ok}")
    n = await store.delete_all(tenant)
    print(f"delete_all(user_alice) → {n} soft-deleted")
    listed_after = await store.list(tenant, limit=10)
    print(f"list(user_alice) after → {len(listed_after)} (expect 0)")

    # ── 8. Mem0 compat layer ─────────────────────────────────────────────
    _hr("8. Mem0 compat layer — from deepmem.compat import mem0_client")
    # The compat client talks to an HTTP server; here we just verify the
    # surface and that it's wired. (Full HTTP exercise is func_test_http.py.)
    from deepmem.compat import mem0_client
    methods = [m for m in ("add", "search", "get_all", "get", "update",
                           "delete", "delete_all", "reset") if callable(getattr(mem0_client, m, None))]
    print(f"mem0_client base_url={mem0_client.base_url}")
    print(f"mem0-shaped methods present: {methods}")

    print(f"\n✓ direct functional test done. (cleanup temp store)")
    shutil.rmtree(_QDRANT, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
