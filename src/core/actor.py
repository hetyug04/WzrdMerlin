import nats
from nats.js.api import RetentionPolicy
from nats.js.errors import NotFoundError
import asyncio
import json
import logging
import os
from typing import Callable, Awaitable, Dict, List
from src.core.events import Event
from src.core.state import StateStore

logger = logging.getLogger(__name__)


class BaseActor:
    """
    Base Actor — connects to NATS with two dispatch paths:

    1. publish() / listen()  — uses CORE NATS subscriptions for reliable
       intra-process agent dispatch.  Simple, zero consumer config overhead.

    2. js_publish()          — writes to JetStream EVENTS stream for
       observability (the sniff subscriber in main.py tails this for the UI).

    v3 changes:
    - Reconnect callbacks with automatic handler re-subscription
    - publish() retries 3 times with 2/4/8s backoff; never re-raises
    """

    def __init__(self, name: str, nats_url: str = None):
        self.name = name
        self.nats_url = nats_url or os.getenv("NATS_URL", "nats://localhost:4222")
        self.nc = None
        self.js = None
        self.state_store = StateStore(nats_url=self.nats_url)
        self._handlers: Dict[str, Callable[[Event], Awaitable[None]]] = {}
        self._subs: List = []          # keep references so GC doesn't drop them

    async def connect(self):
        self.nc = await nats.connect(
            self.nats_url,
            error_cb=self._on_nats_error,
            disconnected_cb=self._on_nats_disconnect,
            reconnected_cb=self._on_nats_reconnect,
            max_reconnect_attempts=-1,   # infinite reconnect attempts
            reconnect_time_wait=2,       # 2s between attempts
        )
        self.js = self.nc.jetstream()
        await self.state_store.connect()
        await self._ensure_stream()
        logger.info(f"Actor {self.name} connected to NATS")

    async def _on_nats_error(self, e):
        logger.error(f"NATS error for {self.name}: {e}")

    async def _on_nats_disconnect(self):
        logger.warning(f"NATS disconnected: {self.name}")

    async def _on_nats_reconnect(self):
        logger.info(f"NATS reconnected: {self.name} — re-subscribing handlers")
        self._subs.clear()
        for subject in list(self._handlers.keys()):
            try:
                sub = await self.nc.subscribe(subject, cb=self._dispatch)
                self._subs.append(sub)
                logger.info(f"Actor {self.name} re-subscribed to {subject!r}")
            except Exception as e:
                logger.error(f"Actor {self.name} failed to re-subscribe to {subject!r}: {e}")

    async def _ensure_stream(self, stream_name: str = "EVENTS",
                              subjects: list = None):
        subjects = subjects or ["events.>"]
        try:
            await self.js.stream_info(stream_name)
        except NotFoundError:
            logger.info(f"Creating JetStream stream: {stream_name}")
            await self.js.add_stream(
                name=stream_name,
                subjects=subjects,
                retention=RetentionPolicy.LIMITS,
            )

    # ------------------------------------------------------------------
    # Dispatch layer: core NATS (reliable, callback-based, no consumer config)
    # ------------------------------------------------------------------

    def on(self, subject: str, handler: Callable[[Event], Awaitable[None]]):
        """Register a handler for a subject (exact match)."""
        self._handlers[subject] = handler

    async def _dispatch(self, msg):
        """Core NATS message callback — parse Event and call registered handler."""
        subject = msg.subject
        try:
            data = json.loads(msg.data.decode())
            event = Event(**data)
            handler = self._handlers.get(subject)
            if handler:
                await handler(event)
            else:
                logger.debug(f"{self.name}: no handler for subject {subject!r}")
        except Exception as e:
            logger.error(f"{self.name} dispatch error on {subject!r}: {e}", exc_info=True)

    async def listen(self, subject: str):
        """Subscribe via core NATS (not JetStream) for agent-to-agent dispatch."""
        if not self.nc:
            raise RuntimeError("Not connected. Call connect() first.")
        sub = await self.nc.subscribe(subject, cb=self._dispatch)
        self._subs.append(sub)          # hold reference
        logger.info(f"Actor {self.name} listening on {subject!r} (core NATS)")
        return sub

    async def publish(self, subject: str, event: Event, *, retries: int = 3) -> None:
        """
        Publish via core NATS with retry/backoff.

        Retries up to `retries` times with 2/4/8s backoff on failure.
        After exhausting retries, logs an error and returns — DOES NOT re-raise.

        # CRITICAL: Do NOT re-raise. Callers must not crash on publish failure.
        # In v2, an unhandled RuntimeError from publish() propagated up through
        # handle_step_requested and silently stopped agent execution.
        """
        payload = event.model_dump_json().encode()
        for attempt in range(retries):
            try:
                if not self.nc or not self.nc.is_connected:
                    raise RuntimeError("NATS not connected")
                await self.nc.publish(subject, payload)
                logger.debug(f"{self.name} published {event.type.value} → {subject}")
                return
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(
                    f"{self.name} publish to {subject!r} failed (attempt {attempt + 1}): {e}"
                )
                if attempt < retries - 1:
                    await asyncio.sleep(wait)
        logger.error(
            f"{self.name} publish to {subject!r} exhausted retries — event lost"
        )
        # CRITICAL: Do NOT re-raise.

    async def close(self):
        if self.nc:
            await self.state_store.close()
            await self.nc.drain()
            logger.info(f"Actor {self.name} closed.")
