import nats
from nats.js.api import KeyValueConfig
import json
import logging
from typing import Any, Dict, Optional

import os

logger = logging.getLogger(__name__)

class StateStore:
    """
    Manages persistent state across the agent OS using NATS JetStream KV store.
    This replaces Postgres from v1 and provides robust crash recovery.
    """
    def __init__(self, nats_url: str = None, bucket_name: str = "MERLIN_STATE"):
        self.nats_url = nats_url or os.getenv("NATS_URL", "nats://localhost:4222")
        self.bucket_name = bucket_name
        self.nc = None
        self.js = None
        self.kv = None

    async def connect(self):
        self.nc = await nats.connect(self.nats_url)
        self.js = self.nc.jetstream()
        
        try:
            self.kv = await self.js.key_value(self.bucket_name)
        except nats.js.errors.NotFoundError:
            logger.info(f"Creating NATS KV Bucket: {self.bucket_name}")
            self.kv = await self.js.create_key_value(
                config=KeyValueConfig(bucket=self.bucket_name)
            )
        logger.info(f"StateStore connected to NATS KV bucket: {self.bucket_name}")

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.kv:
            raise RuntimeError("Not connected. Call connect() first.")
        try:
            entry = await self.kv.get(key)
            return json.loads(entry.value.decode())
        except nats.js.errors.KeyNotFoundError:
            return None

    async def put(self, key: str, value: Dict[str, Any]):
        if not self.kv:
            raise RuntimeError("Not connected. Call connect() first.")
        payload = json.dumps(value).encode()
        await self.kv.put(key, payload)
        
    async def delete(self, key: str):
        if not self.kv:
            raise RuntimeError("Not connected. Call connect() first.")
        await self.kv.delete(key)
        
    async def close(self):
        if self.nc:
            await self.nc.close()
            logger.info("StateStore connection closed.")
