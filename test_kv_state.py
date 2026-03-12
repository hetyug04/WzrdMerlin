#!/usr/bin/env python3
"""Check what task states are persisted in NATS KV."""
import asyncio
import nats
import json
import sys

async def check_kv():
    nc = await nats.connect("nats://localhost:4222")
    try:
        js = nc.jetstream()
        kv = js.key_value("MERLIN_STATE")
        
        # List all actor_state.* keys
        cursor = await kv.keys()
        states = []
        async for key in cursor:
            if key.startswith("actor_state."):
                try:
                    entry = await kv.get(key)
                    data = json.loads(entry.value.decode())
                    states.append({
                        "key": key,
                        "status": data.get("status"),
                        "iteration": data.get("iteration"),
                        "task_id": data.get("task_id")
                    })
                except Exception as e:
                    print(f"Error reading {key}: {e}", file=sys.stderr)
        
        print(f"Persisted Tasks in NATS KV: {len(states)}")
        for s in states:
            print(f"  {s['key']}: status={s['status']}, iter={s['iteration']}")
            
    finally:
        await nc.close()

if __name__ == "__main__":
    asyncio.run(check_kv())
