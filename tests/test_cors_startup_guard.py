"""Phase 4.4: CORS_ORIGINS='*' startup guard.

The default wildcard is fine for local development but is a foot-gun in
production: combined with allow_credentials=True browsers silently drop
Access-Control-Allow-Origin, AND wildcards across hosts trivially expose
authenticated APIs to any origin. The guard refuses to even boot in this
state unless DEEPMEMORY_DEBUG=1 is set explicitly.

Because the guard runs at module-import time, we exercise it from a
subprocess — there's no way to "un-import" server.main in-process to
re-trigger the top-level check.
"""
import os
import subprocess
import sys

import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _python_import_check(env_overrides: dict) -> subprocess.CompletedProcess:
    """Spawn a fresh python that does `import server.main`.

    Strips DEEPMEMORY_DEBUG/CORS_ORIGINS from the inherited env so the
    subprocess only sees what we explicitly hand it — this matters because
    the parent test process sets DEEPMEMORY_DEBUG=1 in conftest.py to
    bypass the guard for the rest of the suite.
    """
    env = {k: v for k, v in os.environ.items()
           if k not in {"DEEPMEMORY_DEBUG", "CORS_ORIGINS"}}
    env.update(env_overrides)
    # Make sure subprocess can resolve `server.main` and `_core.*` the same
    # way conftest.py wires it for the in-process suite.
    env.setdefault("PYTHONPATH",
                   os.pathsep.join([REPO_ROOT,
                                     os.path.join(REPO_ROOT, "deepmem", "vendor")]))
    return subprocess.run(
        [sys.executable, "-c", "import server.main"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_wildcard_without_debug_refuses_to_boot():
    result = _python_import_check({"CORS_ORIGINS": "*"})
    assert result.returncode != 0, (
        f"Expected nonzero exit; stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    combined = (result.stdout + result.stderr).lower()
    assert "cors_origins" in combined
    assert "not allowed in production" in combined


def test_wildcard_with_debug_boots_with_warning():
    result = _python_import_check({
        "CORS_ORIGINS": "*",
        "DEEPMEMORY_DEBUG": "1",
    })
    assert result.returncode == 0, (
        f"DEBUG override must allow boot; stderr={result.stderr!r}"
    )


def test_explicit_origin_list_boots_without_debug():
    result = _python_import_check({
        "CORS_ORIGINS": "https://app.example.com,https://example.com",
    })
    assert result.returncode == 0, (
        f"Explicit allow-list must boot in production; stderr={result.stderr!r}"
    )


def test_empty_cors_origins_falls_back_to_wildcard_guard():
    # The startup code coerces an empty/whitespace-only CORS_ORIGINS to ["*"],
    # which must then trigger the same guard — otherwise unsetting the var
    # accidentally would silently re-enable wildcard.
    result = _python_import_check({"CORS_ORIGINS": "   "})
    assert result.returncode != 0
    assert "cors_origins" in (result.stdout + result.stderr).lower()
