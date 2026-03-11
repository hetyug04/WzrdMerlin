import time, json, httpx

def probe(desc):
    url = "http://localhost:8000/api/run"
    start = time.time()
    with httpx.Client(timeout=None) as c:
        with c.stream("POST", url, params={"description": desc}) as r:
            first = None
            first_token = None
            for line in r.iter_lines():
                if not line:
                    continue
                now = time.time()
                if first is None:
                    first = now
                msg = json.loads(line)
                if msg.get("type") == "token" and first_token is None:
                    first_token = now
                    break
            print({
                "desc": desc[:40],
                "status": r.status_code,
                "first_event_s": round((first-start) if first else -1, 3),
                "first_token_s": round((first_token-start) if first_token else -1, 3),
                "total_s": round(time.time()-start, 3),
            })

probe("Hello")
probe("Hello")
probe("Write one sentence about cats.")
probe("Navigate to https://news.ycombinator.com/ and extract the top 10 titles and links.")
