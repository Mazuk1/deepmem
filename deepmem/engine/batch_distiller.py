import asyncio
import logging
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("deepmem.engine.distiller")


class AsyncBatchDistiller:
    """Batches messages per (user, agent, run) scope, fires LLM extraction
    after silence window.

    Reduces LLM calls by ~80% by waiting for a conversation pause,
    then processing all accumulated messages in a single LLM call.

    Run-scoped batching: when callers pass run_id (and optionally agent_id),
    each session gets its own queue so concurrent agents/sessions for the
    same user don't get their messages mixed.
    """

    def __init__(self, silence_window: float = 180.0, max_batch_size: int = 50):
        self.silence_window = silence_window
        self.max_batch_size = max_batch_size
        self.queues: Dict[str, List[Dict[str, str]]] = {}
        self.metadata: Dict[str, Dict] = {}
        # BYOK adapter config per scope (api_key/base_url/model). Kept
        # in-memory only — never persisted, never logged.
        self.llm_configs: Dict[str, Dict] = {}
        # Maps composite key -> {"user_id", "agent_id", "run_id"} so
        # _fire() can recover the dimensions for the callback.
        self.scopes: Dict[str, Dict[str, Optional[str]]] = {}
        self.timers: Dict[str, asyncio.Task] = {}
        self.on_batch_ready: Optional[Callable] = None

    @staticmethod
    def _composite_key(user_id: str,
                       agent_id: Optional[str] = None,
                       run_id: Optional[str] = None) -> str:
        """Build the queue key. Plain user_id when no agent/run scope, so
        the simple single-user case matches existing call sites and tests."""
        if not agent_id and not run_id:
            return user_id
        return f"{user_id}|a={agent_id or ''}|r={run_id or ''}"

    async def enqueue(self, user_id: str,
                      messages: List[Dict[str, str]],
                      metadata: Optional[Dict] = None,
                      llm_config: Optional[Dict] = None,
                      agent_id: Optional[str] = None,
                      run_id: Optional[str] = None,
                      account_id: Optional[str] = None,
                      owner_id: Optional[str] = None,
                      visibility_scope: Optional[str] = None,
                      source_type: Optional[str] = None,
                      session_id: Optional[str] = None) -> None:
        """Add messages to the scope's queue and reset the silence timer."""
        key = self._composite_key(user_id, agent_id, run_id)
        if key not in self.queues:
            self.queues[key] = []

        self.queues[key].extend(messages)
        if metadata:
            self.metadata[key] = metadata
        if llm_config:
            # Last writer wins — fine for the BYOK use case where a user
            # uses their own key consistently.
            self.llm_configs[key] = llm_config
        # Always refresh scope so a later call without these kwargs (rare)
        # doesn't lose the original dimensions.
        self.scopes[key] = {
            "user_id": user_id,
            "agent_id": agent_id,
            "run_id": run_id,
            "account_id": account_id,
            # Multi-agent permission groundwork - forwarded to process_batch
            # so batched writes carry the same payload fields as direct adds.
            "owner_id": owner_id,
            "visibility_scope": visibility_scope,
            "source_type": source_type,
            "session_id": session_id,
        }

        queue_len = len(self.queues[key])
        logger.debug("Distiller enqueue scope=%s added=%d total=%d",
                     key, len(messages), queue_len)

        if queue_len >= self.max_batch_size:
            logger.info("Distiller max_batch_size=%d reached for scope=%s, firing immediately",
                        self.max_batch_size, key)
            await self._fire(key)
        else:
            self._reset_timer(key)

    def _reset_timer(self, key: str) -> None:
        """Reset the silence window timer for a scope."""
        if key in self.timers:
            self.timers[key].cancel()
        self.timers[key] = asyncio.create_task(self._wait_and_fire(key))
        logger.debug("Distiller timer reset for scope=%s window=%.1fs",
                     key, self.silence_window)

    async def _wait_and_fire(self, key: str) -> None:
        """Wait for silence window, then fire batch processing."""
        try:
            await asyncio.sleep(self.silence_window)
            await self._fire(key)
        except asyncio.CancelledError:
            # Timer was reset or scope cleared — leave the queue for whoever
            # owns it next; no firing.
            raise

    async def _fire(self, key: str) -> None:
        """Process all queued messages for a scope."""
        if key not in self.queues or not self.queues[key]:
            return
        batch = list(self.queues[key])
        meta = self.metadata.pop(key, None)
        llm_config = self.llm_configs.pop(key, None)
        scope = self.scopes.pop(key, None) or {"user_id": key,
                                               "agent_id": None,
                                               "run_id": None,
                                               "account_id": None,
                                               "owner_id": None,
                                               "visibility_scope": None,
                                               "source_type": None,
                                               "session_id": None}
        self.queues[key].clear()
        if key in self.timers:
            self.timers[key].cancel()
            del self.timers[key]
        msg_count = len(batch)
        logger.info(
            "Distiller FIRING batch user=%s agent=%s run=%s msg_count=%d byok=%s",
            scope["user_id"], scope.get("agent_id") or "-",
            scope.get("run_id") or "-", msg_count, bool(llm_config),
        )
        if not self.on_batch_ready:
            return

        kwargs = {
            "user_id": scope["user_id"],
            "metadata": meta,
            "llm_config": llm_config,
            "agent_id": scope.get("agent_id"),
            "run_id": scope.get("run_id"),
            "account_id": scope.get("account_id"),
            "owner_id": scope.get("owner_id"),
            "visibility_scope": scope.get("visibility_scope"),
            "source_type": scope.get("source_type"),
            "session_id": scope.get("session_id"),
        }
        # Single call. The callback (VectorStore.process_batch) accepts every
        # kwarg above; the previous progressive-degradation fallback would
        # silently drop account_id when an unrelated TypeError fired inside
        # the callback, breaking tenant isolation. If the wired callback ever
        # changes shape, that's a bug to surface, not paper over.
        await self.on_batch_ready(batch, **kwargs)

    async def flush(self, user_id: Optional[str] = None) -> None:
        """Force-fire pending batches (for tests / shutdown).

        With user_id: fire that user's queues across all run/agent scopes.
        Without: fire all pending scopes.
        """
        if user_id:
            keys_to_fire = [k for k, s in self.scopes.items()
                            if s.get("user_id") == user_id]
            # Plain user_id key (no agent/run) won't have a scope entry
            # if it was never set — include it when present.
            if user_id in self.queues and user_id not in keys_to_fire:
                keys_to_fire.append(user_id)
            for key in keys_to_fire:
                await self._fire(key)
        else:
            for key in list(self.queues.keys()):
                await self._fire(key)

    def get_queue_size(self, user_id: str,
                       agent_id: Optional[str] = None,
                       run_id: Optional[str] = None) -> int:
        """Get current queue size for a specific scope."""
        return len(self.queues.get(
            self._composite_key(user_id, agent_id, run_id), []))

    def _scope_keys_for_user(self, user_id: str) -> List[str]:
        """All composite keys belonging to a given user (any agent/run)."""
        out: List[str] = []
        for k, s in self.scopes.items():
            if s.get("user_id") == user_id:
                out.append(k)
        if user_id in self.queues and user_id not in out:
            out.append(user_id)
        return out

    async def aclear(self, user_id: Optional[str] = None) -> None:
        """Async clear — cancels timers and awaits their teardown.

        Awaiting cancellation prevents a half-fired _fire() from racing
        with the cleanup (#6 in the second review). When user_id is given,
        clears every scope (run / agent) belonging to that user.
        """
        if user_id:
            keys = self._scope_keys_for_user(user_id)
            tasks = [self.timers.pop(k, None) for k in keys]
            for k in keys:
                self.queues.pop(k, None)
                self.metadata.pop(k, None)
                self.llm_configs.pop(k, None)
                self.scopes.pop(k, None)
        else:
            tasks = list(self.timers.values())
            self.timers.clear()
            self.queues.clear()
            self.metadata.clear()
            self.llm_configs.clear()
            self.scopes.clear()
        for t in tasks:
            if t is None or t.done():
                continue
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        logger.info("Distiller cleared for user=%s", user_id or "<all>")

    def clear(self, user_id: Optional[str] = None) -> None:
        """Sync clear (legacy). Cancels timers but does not await them.

        Prefer aclear() in async contexts. Kept because some sync call sites
        (HTTP request handlers between awaits) only need fire-and-forget.
        """
        if user_id:
            keys = self._scope_keys_for_user(user_id)
            for k in keys:
                if k in self.timers:
                    self.timers[k].cancel()
                    del self.timers[k]
                self.queues.pop(k, None)
                self.metadata.pop(k, None)
                self.llm_configs.pop(k, None)
                self.scopes.pop(k, None)
            logger.info("Distiller cleared for user=%s", user_id)
        else:
            for tid in list(self.timers.keys()):
                self.timers[tid].cancel()
            self.timers.clear()
            self.queues.clear()
            self.metadata.clear()
            self.llm_configs.clear()
            self.scopes.clear()
            logger.info("Distiller cleared all users")
