import asyncio, json, os
from nats import connect

async def main():
    try:
        nats_url = os.getenv('NATS_URL','nats://localhost:4222')
        nc = await connect(nats_url)
        js = nc.jetstream()
        try:
            kv = await js.key_value('MERLIN_STATE')
        except Exception as e:
            print('KV bucket MERLIN_STATE not found or error:', e)
            await nc.close()
            return
        try:
            keys = await kv.keys()
        except Exception:
            keys = []
        found = False
        for k in keys:
            if k.startswith('actor_state.'):
                entry = await kv.get(k)
                val = json.loads(entry.value.decode())
                pending = val.get('pending_children')
                child_results = val.get('child_results', {}) or {}
                if pending:
                    found = True
                    print(k, 'PENDING_CHILDREN:', pending, 'CHILD_RESULTS_COUNT:', len(child_results))
                else:
                    print(k, 'no pending_children', 'CHILD_RESULTS_COUNT:', len(child_results))
        if not found:
            print('No actor_state.* with pending_children found.')
        await nc.close()
    except Exception as e:
        print('NATS connection or query failed:', e)

if __name__ == '__main__':
    asyncio.run(main())
