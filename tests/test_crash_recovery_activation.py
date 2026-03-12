import asyncio
import json
import subprocess
import time
import uuid

import nats


NATS_URL = "nats://localhost:4222"
BUCKET = "MERLIN_STATE"
CORE_CONTAINER = "wzrdmerlinv2-core-1"
TIMEOUT_SECONDS = 25


async def wait_for_recovery_event(task_id: str, timeout: int) -> bool:
    nc = await nats.connect(NATS_URL)
    recovered = asyncio.Event()

    async def on_msg(msg):
        try:
            data = json.loads(msg.data.decode())
            payload = data.get("payload", {})
            if payload.get("task_id") == task_id and data.get("type") == "step.requested":
                recovered.set()
        except Exception:
            return

    try:
        await nc.subscribe("events.step.requested", cb=on_msg)
        try:
            await asyncio.wait_for(recovered.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
    finally:
        await nc.close()


async def seed_running_task(task_id: str):
    nc = await nats.connect(NATS_URL)
    try:
        js = nc.jetstream()
        kv = await js.key_value(BUCKET)
        state = {
            "task_id": task_id,
            "instruction": "Recovery probe task. This task exists only to verify crash recovery activation.",
            "history": [],
            "iteration": 0,
            "status": "running",
            "created_at": int(time.time()),
        }
        await kv.put(f"actor_state.{task_id}", json.dumps(state).encode())
    finally:
        await nc.close()


async def cleanup_task(task_id: str):
    nc = await nats.connect(NATS_URL)
    try:
        js = nc.jetstream()
        kv = await js.key_value(BUCKET)
        await kv.delete(f"actor_state.{task_id}")
    finally:
        await nc.close()


def restart_core_container():
    subprocess.run(["docker", "restart", CORE_CONTAINER], check=True, capture_output=True, text=True)


async def main():
    task_id = f"recovery_probe_{uuid.uuid4().hex[:8]}"
    print(f"Seeding running task: {task_id}")
    await seed_running_task(task_id)

    waiter = asyncio.create_task(wait_for_recovery_event(task_id, TIMEOUT_SECONDS))

    print(f"Restarting container: {CORE_CONTAINER}")
    restart_core_container()

    recovered = await waiter

    await cleanup_task(task_id)

    if not recovered:
        raise SystemExit(
            f"FAIL: Did not observe a recovery step.requested event for {task_id} within {TIMEOUT_SECONDS}s"
        )

    print(f"PASS: Observed crash recovery activation for {task_id}")


if __name__ == "__main__":
    asyncio.run(main())