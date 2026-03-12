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

    Keeping dispatch on core NATS avoids the JetStream push-consumer
    lifecycle issues (deliver-inbox GC, durable binding races, ack-wait
    stalls) that blocked callbacks in the previous design.
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
        self.nc = await nats.connect(self.nats_url)
        self.js = self.nc.jetstream()
        await self.state_store.connect()
        await self._ensure_stream()
        logger.info(f"Actor {self.name} connected to NATS")

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

    async def publish(self, subject: str, event: Event):
        """
        Publish via core NATS only.
        Core NATS delivers to all nc.subscribe() listeners immediately.
        The sniff in main.py also uses a core NATS subscription so it
        sees every event without an extra JetStream round-trip.
        """
        if not self.nc:
            raise RuntimeError("Not connected. Call connect() first.")
        payload = event.model_dump_json().encode()
        await self.nc.publish(subject, payload)
        logger.debug(f"{self.name} published {event.type.value} → {subject}")

    async def close(self):
        if self.nc:
            await self.state_store.close()
            await self.nc.drain()
            logger.info(f"Actor {self.name} closed.")
