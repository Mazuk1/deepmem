"""SQLite-backed memory history log.

Records every ADD / UPDATE / DELETE event against a memory id so callers can
reconstruct what the system "remembered" at any point in time (mirrors mem0's
history table). The store is intentionally append-only — events are never
mutated, only inserted; reset() truncates a single user's rows.

Threading: Qdrant writes happen on the asyncio loop thread, so we open a
single sqlite3 connection in WAL mode and serialize writes via a Lock. WAL +
single-writer keeps schema simple while letting reads run concurrently from
the API handlers.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("deepmem.history")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id    TEXT    NOT NULL,
    user_id      TEXT    NOT NULL,
    account_id   TEXT,
    agent_id     TEXT,
    run_id       TEXT,
    event        TEXT    NOT NULL,
    prev_memory  TEXT,
    new_memory   TEXT,
    timestamp    REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_memory_id ON memory_history(memory_id);
CREATE INDEX IF NOT EXISTS idx_history_user_id   ON memory_history(user_id);
CREATE INDEX IF NOT EXISTS idx_history_timestamp ON memory_history(timestamp);
"""
# NOTE: idx_history_account is created AFTER the ALTER-TABLE add-column step
# below, not in _SCHEMA. Older DBs predate the account_id column; CREATE
# INDEX IF NOT EXISTS still fails on a missing column, so we have to add
# the column first and then build the index.


class HistoryStore:
    """Append-only event log for memory lifecycle events."""

    def __init__(self, db_path: str = "./data/history.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        # check_same_thread=False: FastAPI runs handlers on a worker thread
        # pool but we serialize writes via _lock, so the connection is safe
        # to share. Without this the second handler call raises ProgrammingError.
        self._conn = sqlite3.connect(
            db_path, check_same_thread=False, isolation_level=None,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        # Idempotent migration for DBs that pre-date the account_id column.
        # SQLite has no IF NOT EXISTS for ADD COLUMN, so we swallow the
        # "duplicate column name" OperationalError on subsequent boots.
        try:
            self._conn.execute("ALTER TABLE memory_history ADD COLUMN account_id TEXT")
            logger.info("HistoryStore: added account_id column")
        except sqlite3.OperationalError:
            pass
        # Build the account_id index unconditionally — IF NOT EXISTS makes it
        # idempotent and the column is now guaranteed to exist (either fresh
        # from CREATE TABLE or backfilled by the ALTER above).
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_account ON memory_history(account_id)"
        )
        logger.info("HistoryStore initialized at %s", db_path)

    # ── Writes ────────────────────────────────────────────────────────

    def record(self, *, memory_id: str, user_id: str, event: str,
               agent_id: Optional[str] = None,
               run_id: Optional[str] = None,
               account_id: Optional[str] = None,
               prev_memory: Optional[str] = None,
               new_memory: Optional[str] = None,
               timestamp: Optional[float] = None) -> None:
        """Append a single history event. Never raises — log failures only.

        History should never block the actual memory write; if SQLite is
        somehow unhealthy, we'd rather lose audit rows than reject the user's
        data write.
        """
        ts = timestamp if timestamp is not None else time.time()
        if event not in ("ADD", "UPDATE", "DELETE"):
            logger.warning("HistoryStore: unknown event=%s — recording anyway", event)
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO memory_history "
                    "(memory_id, user_id, account_id, agent_id, run_id, event, "
                    " prev_memory, new_memory, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (memory_id, user_id, account_id, agent_id, run_id, event,
                     prev_memory, new_memory, ts),
                )
        except Exception as e:
            logger.error("HistoryStore.record failed memory_id=%s: %s", memory_id, e)

    def record_many(self, events: List[Dict[str, Any]]) -> None:
        """Bulk insert — used by the ADD path after a batch upsert."""
        if not events:
            return
        rows = []
        for ev in events:
            rows.append((
                ev["memory_id"], ev["user_id"], ev.get("account_id"),
                ev.get("agent_id"), ev.get("run_id"),
                ev.get("event", "ADD"),
                ev.get("prev_memory"), ev.get("new_memory"),
                ev.get("timestamp", time.time()),
            ))
        try:
            with self._lock:
                self._conn.executemany(
                    "INSERT INTO memory_history "
                    "(memory_id, user_id, account_id, agent_id, run_id, event, "
                    " prev_memory, new_memory, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
        except Exception as e:
            logger.error("HistoryStore.record_many failed count=%d: %s", len(events), e)

    # ── Reads ─────────────────────────────────────────────────────────

    def history_for_memory(self, memory_id: str, user_id: str,
                           limit: int = 100,
                           account_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return events for a single memory id, oldest first.

        user_id is enforced — even if the caller knows the memory_id, they
        cannot read another tenant's audit trail. account_id, when supplied,
        adds an extra fence so two accounts that happen to use the same
        user_id string can't read each other's history.
        """
        sql = ("SELECT id, memory_id, user_id, agent_id, run_id, event, "
               "       prev_memory, new_memory, timestamp "
               "FROM memory_history "
               "WHERE memory_id=? AND user_id=?")
        params: List[Any] = [memory_id, user_id]
        if account_id is not None:
            sql += " AND account_id=?"
            params.append(account_id)
        sql += " ORDER BY timestamp ASC, id ASC LIMIT ?"
        params.append(limit)
        try:
            with self._lock:
                cur = self._conn.execute(sql, tuple(params))
                rows = cur.fetchall()
        except Exception as e:
            logger.error("HistoryStore.history_for_memory failed: %s", e)
            return []
        return [self._row_to_dict(r) for r in rows]

    def history_for_user(self, user_id: str, limit: int = 500,
                         since: Optional[float] = None,
                         account_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return all events for a user, newest first (audit / debugging)."""
        params: List[Any] = [user_id]
        sql = ("SELECT id, memory_id, user_id, agent_id, run_id, event, "
               "       prev_memory, new_memory, timestamp "
               "FROM memory_history WHERE user_id=?")
        if account_id is not None:
            sql += " AND account_id=?"
            params.append(account_id)
        if since is not None:
            sql += " AND timestamp >= ?"
            params.append(since)
        sql += " ORDER BY timestamp DESC, id DESC LIMIT ?"
        params.append(limit)
        try:
            with self._lock:
                cur = self._conn.execute(sql, tuple(params))
                rows = cur.fetchall()
        except Exception as e:
            logger.error("HistoryStore.history_for_user failed: %s", e)
            return []
        return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row) -> Dict[str, Any]:
        return {
            "id": row[0],
            "memory_id": row[1],
            "user_id": row[2],
            "agent_id": row[3],
            "run_id": row[4],
            "event": row[5],
            "prev_memory": row[6],
            "new_memory": row[7],
            "timestamp": row[8],
        }

    # ── Maintenance ───────────────────────────────────────────────────

    def reset_user(self, user_id: str,
                   account_id: Optional[str] = None) -> int:
        """Hard-delete every history row for a user. Used by /reset.

        When account_id is supplied, the wipe is scoped to that account too,
        so an account-scoped reset on user="alice" won't take out a different account's
        own user="alice" rows.
        """
        sql = "DELETE FROM memory_history WHERE user_id=?"
        params: List[Any] = [user_id]
        if account_id is not None:
            sql += " AND account_id=?"
            params.append(account_id)
        try:
            with self._lock:
                cur = self._conn.execute(sql, tuple(params))
                return cur.rowcount or 0
        except Exception as e:
            logger.error("HistoryStore.reset_user failed user=%s: %s", user_id, e)
            return 0

    def delete_for_account(self, account_id: str) -> int:
        """Hard-delete every history row belonging to an account.

        Counterpart to VectorStore.hard_delete_account — called when an
        account is being permanently removed and all of its audit rows
        need to go too.
        """
        if not account_id:
            return 0
        try:
            with self._lock:
                cur = self._conn.execute(
                    "DELETE FROM memory_history WHERE account_id=?",
                    (account_id,),
                )
                return cur.rowcount or 0
        except Exception as e:
            logger.error("HistoryStore.delete_for_account failed account=%s: %s",
                         account_id, e)
            return 0

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
