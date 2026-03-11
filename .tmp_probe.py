import time
import json
import httpx

url = "http://localhost:8000/api/run"
params = {
    "description": "Navigate to https://news.ycombinator.com/ and extract the titles and links of the top 10 current stories."
}

start = time.time()
with httpx.Client(timeout=None) as client:
    with client.stream("POST", url, params=params) as response:
        print("status", response.status_code, "t", round(time.time() - start, 3))
        first = None
        for line in response.iter_lines():
            if not line:
                continue
            now = time.time()
            if first is None:
                first = now
                print("first_line_latency", round(first - start, 3))
            msg = json.loads(line)
            print("event", msg.get("type"), "dt", round(now - start, 3))
            if msg.get("type") in ("tool_start", "error", "done"):
                break
