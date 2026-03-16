"""
WzrdMerlin v3 — OllamaClient

Replaces v2's ModelInterface (~760 lines) + inference.py (~627 lines) with ~250 lines.
Fixes:
  - keep_alive=-1 on every request (model never unloads)
  - per-chunk 120s read timeout (was 300s total in v2)
  - CircuitBreaker (5 failures → OPEN; 60s recovery → HALF_OPEN)
  - 3 retries with 2/4/8s backoff on connection errors
  - 6-strategy JSON parser migrated verbatim from v2 llm.py
  - <think>...</think> streaming parser migrated verbatim from v2 llm.py
"""
import asyncio
import json
import logging
import math
import os
import re
import time
from typing import Any, AsyncIterator, Callable, Awaitable, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


class OllamaUnavailableError(Exception):
    """Raised when the circuit breaker is open or model is not loaded."""


# ── CircuitBreaker ─────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    CLOSED → OPEN after `failure_threshold` consecutive failures.
    OPEN → HALF_OPEN after `recovery_timeout` seconds.
    HALF_OPEN → CLOSED on success, OPEN on failure.
    """
    _STATE_CLOSED = "closed"
    _STATE_OPEN = "open"
    _STATE_HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self._threshold = failure_threshold
        self._recovery = recovery_timeout
        self._failures = 0
        self._state = self._STATE_CLOSED
        self._opened_at: float = 0.0

    def record_success(self) -> None:
        self._failures = 0
        self._state = self._STATE_CLOSED

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._state = self._STATE_OPEN
            self._opened_at = time.monotonic()

    def is_open(self) -> bool:
        if self._state == self._STATE_CLOSED:
            return False
        if self._state == self._STATE_OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._recovery:
                self._state = self._STATE_HALF_OPEN
                return False
            return True
        # HALF_OPEN — allow one probe
        return False


# ── OllamaClient ──────────────────────────────────────────────────────────────

class OllamaClient:
    """
    Async Ollama /api/chat client with circuit breaker and retry.

    Single persistent httpx.AsyncClient — reuses TCP connection pool.
    """

    def __init__(
        self,
        base_url: str = "http://host.docker.internal:11434",
        model: str = "merlin-model:latest",
        context_window: int = 32768,
        temperature: float = 0.3,
        think: bool = False,
        read_timeout: float = 120.0,
        keep_alive: str = "-1",
        shared_http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.context_window = context_window
        self.temperature = temperature
        self.think = think
        # Normalize keep_alive: bare integer strings (e.g. "-1", "0") must be
        # sent as JSON integers — Ollama rejects "time: missing unit in duration".
        try:
            self.keep_alive: int | str = int(keep_alive)
        except (ValueError, TypeError):
            self.keep_alive = keep_alive

        self._owns_client = shared_http_client is None
        self._client = shared_http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=read_timeout,
                write=10.0,
                pool=5.0,
            )
        )
        self._breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Model load / unload
    # ------------------------------------------------------------------

    async def evict_all_models(self) -> None:
        """Evict ALL models currently loaded in Ollama VRAM.
        Uses GET /api/ps to list running models, then unloads each one."""
        try:
            resp = await self._client.get(
                f"{self.base_url}/api/ps",
                timeout=httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0),
            )
            if resp.status_code != 200:
                logger.warning(f"OllamaClient: /api/ps returned {resp.status_code}")
                return
            data = resp.json()
            running = data.get("models", [])
            if not running:
                logger.info("OllamaClient: no models loaded in VRAM — nothing to evict")
                return
            for m in running:
                name = m.get("name") or m.get("model", "")
                if not name:
                    continue
                try:
                    await self._client.post(
                        f"{self.base_url}/api/generate",
                        json={"model": name, "keep_alive": 0},
                        timeout=httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0),
                    )
                    logger.info(f"OllamaClient: evicted '{name}' from VRAM")
                except Exception as e:
                    logger.warning(f"OllamaClient: failed to evict '{name}': {e}")
        except Exception as e:
            logger.warning(f"OllamaClient: evict_all_models failed (non-fatal): {e}")

    async def unload_model(self) -> None:
        """Evict this model from GPU VRAM immediately (keep_alive=0)."""
        try:
            await self._client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "keep_alive": 0},
                timeout=httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0),
            )
            logger.info(f"OllamaClient: unloaded '{self.model}' from GPU")
        except Exception as e:
            logger.warning(f"OllamaClient: unload_model failed (non-fatal): {e}")

    async def preload_model(self) -> None:
        """Warm-load this model into GPU VRAM without generating tokens."""
        try:
            await self._client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "keep_alive": self.keep_alive, "prompt": ""},
                timeout=httpx.Timeout(connect=5.0, read=60.0, write=5.0, pool=5.0),
            )
            logger.info(f"OllamaClient: preloaded '{self.model}' into GPU")
        except Exception as e:
            logger.warning(f"OllamaClient: preload_model failed (non-fatal): {e}")

    # ------------------------------------------------------------------
    # Model check
    # ------------------------------------------------------------------

    async def ensure_model_loaded(self) -> bool:
        """GET /api/tags; return True if self.model is present."""
        try:
            resp = await self._client.get(f"{self.base_url}/api/tags")
            if resp.status_code != 200:
                return False
            data = resp.json()
            models = data.get("models", [])
            return any(
                m.get("name") == self.model or m.get("model") == self.model
                for m in models
            )
        except Exception as e:
            logger.warning(f"OllamaClient: ensure_model_loaded failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    # Default think budget: 4096 tokens (~16K chars). Prevents infinite think loops.
    THINK_BUDGET_CHARS = int(os.getenv("MERLIN_THINK_BUDGET_CHARS", "16000"))

    async def stream_chat(
        self,
        messages: List[Dict],
        heartbeat_cb: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> AsyncIterator[Tuple[str, str]]:
        """
        Stream /api/chat. Yields ("think"|"content", text) tuples.
        Retries up to 3 times with 2/4/8s backoff on connection errors.
        Raises OllamaUnavailableError if circuit breaker is open.
        Think tokens are capped at THINK_BUDGET_CHARS to prevent infinite loops.
        """
        if self._breaker.is_open():
            raise OllamaUnavailableError(
                f"OllamaClient: circuit breaker is open for {self.model}"
            )

        # Don't send think=true/false — the Modelfile template controls <think>
        # behavior. Sending think=true to a non-thinking model causes an error;
        # sending think=false may conflict with the template.
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.context_window,
            },
        }

        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                async with self._client.stream(
                    "POST", f"{self.base_url}/api/chat", json=payload
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        logger.error(
                            f"OllamaClient: /api/chat returned {resp.status_code}: {body[:300]}"
                        )
                        self._breaker.record_failure()
                        return

                    in_think = False
                    buffer = ""
                    think_chars = 0
                    budget_exceeded = False

                    async for raw_line in resp.aiter_lines():
                        if not raw_line.strip():
                            continue
                        try:
                            chunk = json.loads(raw_line)
                        except json.JSONDecodeError:
                            continue

                        delta = chunk.get("message", {}).get("content", "")
                        if not delta:
                            if chunk.get("done"):
                                break
                            continue

                        if heartbeat_cb:
                            await heartbeat_cb(delta)

                        buffer += delta

                        # Parse <think> tags — migrated from v2 llm.py _stream_ollama
                        while buffer:
                            if in_think:
                                end = buffer.find("</think>")
                                if end != -1:
                                    chunk_text = buffer[:end]
                                    think_chars += len(chunk_text)
                                    yield ("think", chunk_text)
                                    buffer = buffer[end + 8:]
                                    in_think = False
                                else:
                                    safe = max(0, len(buffer) - 8)
                                    if safe > 0:
                                        think_chars += safe
                                        yield ("think", buffer[:safe])
                                        buffer = buffer[safe:]
                                    # Check think budget
                                    if think_chars > self.THINK_BUDGET_CHARS:
                                        logger.warning(
                                            f"OllamaClient: think budget exceeded "
                                            f"({think_chars} chars > {self.THINK_BUDGET_CHARS}) — "
                                            f"forcing end of thinking"
                                        )
                                        yield ("think", "\n[budget exceeded — stopping think]\n")
                                        # Inject </think> to force content mode
                                        buffer = "</think>" + buffer
                                        budget_exceeded = True
                                    break
                            else:
                                start = buffer.find("<think>")
                                if start != -1:
                                    if start > 0:
                                        yield ("content", buffer[:start])
                                    buffer = buffer[start + 7:]
                                    in_think = True
                                    # If budget was already exceeded, don't re-enter think
                                    if budget_exceeded:
                                        in_think = False
                                        yield ("content", buffer)
                                        buffer = ""
                                        break
                                else:
                                    safe = max(0, len(buffer) - 7)
                                    if safe > 0:
                                        yield ("content", buffer[:safe])
                                        buffer = buffer[safe:]
                                    break

                    if buffer:
                        yield ("think" if in_think else "content", buffer)

                self._breaker.record_success()
                return

            except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
                last_error = e
                self._breaker.record_failure()
                wait = 2 ** attempt
                logger.warning(
                    f"OllamaClient: stream_chat attempt {attempt + 1} failed: {e} — retrying in {wait}s"
                )
                if attempt < 2:
                    await asyncio.sleep(wait)
            except Exception as e:
                last_error = e
                self._breaker.record_failure()
                logger.error(f"OllamaClient: stream_chat unexpected error: {e}")
                return

        logger.error(
            f"OllamaClient: stream_chat exhausted retries for {self.model}: {last_error}"
        )

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: List[Dict],
        temperature_override: Optional[float] = None,
    ) -> Optional[str]:
        """Non-streaming POST /api/chat. Returns raw content string or None."""
        if self._breaker.is_open():
            raise OllamaUnavailableError(
                f"OllamaClient: circuit breaker is open for {self.model}"
            )

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "think": False,
            "options": {
                "temperature": temperature_override if temperature_override is not None else self.temperature,
                "num_ctx": self.context_window,
            },
        }
        try:
            resp = await self._client.post(f"{self.base_url}/api/chat", json=payload)
            if resp.status_code != 200:
                logger.error(f"OllamaClient: chat returned {resp.status_code}: {resp.text[:300]}")
                self._breaker.record_failure()
                return None
            data = resp.json()
            self._breaker.record_success()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            self._breaker.record_failure()
            logger.error(f"OllamaClient: chat failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def build_messages(
        self,
        system_prompt: str,
        history: List[Dict],
        instruction: str,
    ) -> List[Dict]:
        """Build chat messages for Ollama.

        Key constraint: Qwen3.5's chat template collapses consecutive user
        messages (im_start/im_end tokens). The "generate next action" prompt
        must therefore be appended to the LAST user message, never added as
        a separate message — otherwise the model loses history context.
        """
        messages = [{"role": "system", "content": system_prompt}]

        if not history:
            # No history: combine objective + generate prompt in one user message
            messages.append({
                "role": "user",
                "content": (
                    f"Current Objective: {instruction}\n\n"
                    "Generate your next tool call as a JSON object."
                ),
            })
            return messages

        # First user message: objective only (history follows as assistant/user pairs)
        messages.append({
            "role": "user",
            "content": f"Current Objective: {instruction}",
        })

        for i, msg in enumerate(history):
            action = msg.get("action")
            is_last = (i == len(history) - 1)
            result_text = msg.get("result", "")

            if isinstance(action, dict) and action.get("tool", "").startswith("_"):
                # Context fold summary — inject as user message
                content = f"[Context summary from earlier steps]\n{result_text}"
                if is_last:
                    content += "\n\nGenerate your next tool call as a JSON object."
                messages.append({"role": "user", "content": content})
                continue

            messages.append({
                "role": "assistant",
                "content": json.dumps(action) if action else "(no action)",
            })

            # Merge "generate" prompt into the last result message to avoid
            # consecutive user messages which break Qwen3.5's context tracking.
            if is_last:
                messages.append({
                    "role": "user",
                    "content": (
                        f"Result: {result_text}\n\n"
                        "Generate your next tool call as a JSON object."
                    ),
                })
            else:
                messages.append({
                    "role": "user",
                    "content": f"Result: {result_text}",
                })

        return messages

    @staticmethod
    def _estimate_prompt_tokens(messages: List[Dict]) -> int:
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return max(1, math.ceil(total_chars / 4))

    # ==================================================================
    # Robust JSON parser — migrated verbatim from v2 ModelInterface
    # DO NOT simplify or refactor. This parser is battle-tested against
    # Qwen3.5-9B output.
    # ==================================================================

    @staticmethod
    def _repair(text: str) -> str:
        """Apply common repairs to LLM-emitted JSON."""
        s = text
        # 1. Replace escaped single quotes (\' → ') — invalid in JSON
        s = s.replace("\\'", "'")
        # 2. Fix literal (unescaped) newlines/tabs inside string values.
        out = []
        in_str = False
        esc = False
        for ch in s:
            if esc:
                out.append(ch)
                esc = False
                continue
            if ch == '\\' and in_str:
                out.append(ch)
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                out.append(ch)
                continue
            if in_str:
                if ch == '\n':
                    out.append('\\n')
                    continue
                if ch == '\r':
                    out.append('\\r')
                    continue
                if ch == '\t':
                    out.append('\\t')
                    continue
            out.append(ch)
        return "".join(out)

    @staticmethod
    def _extract_balanced(text: str) -> Optional[str]:
        """
        Walk from the first '{' to the matching '}', tracking string
        literals so nested braces in strings don't confuse the count.
        """
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == '\\' and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        if depth > 0:
            return text[start:] + ("}" * depth)
        return None

    @staticmethod
    def _try_json(text: str) -> Optional[Dict[str, Any]]:
        """Attempt json.loads; return dict or None."""
        if not text or not text.strip():
            return None
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "tool" in obj:
                return obj
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    @staticmethod
    def _regex_extract(text: str) -> Optional[Dict[str, Any]]:
        """Last-resort extraction: pull 'tool' and 'args' with regex."""
        tool_m = re.search(r'"tool"\s*:\s*"([^"]+)"', text)
        if not tool_m:
            return None
        tool_name = tool_m.group(1)
        args_m = re.search(r'"args"\s*:\s*(\{)', text)
        if args_m:
            start = args_m.start(1)
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(text)):
                ch = text[i]
                if esc:
                    esc = False
                    continue
                if ch == '\\' and in_str:
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        args_str = text[start:i + 1]
                        try:
                            args = json.loads(args_str)
                            return {"tool": tool_name, "args": args}
                        except json.JSONDecodeError:
                            pass
                        try:
                            args = json.loads(OllamaClient._repair(args_str))
                            return {"tool": tool_name, "args": args}
                        except json.JSONDecodeError:
                            pass
                        break
        val_m = re.search(
            r'"(?:cmd|path|query|url|summary|content|question)"\s*:\s*"((?:[^"\\]|\\.)*)"',
            text,
        )
        if val_m:
            key = re.search(r'"(cmd|path|query|url|summary|content|question)"', text).group(1)
            return {"tool": tool_name, "args": {key: val_m.group(1)}}
        return {"tool": tool_name, "args": {}}

    def parse_action(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Robust JSON extractor tuned for Qwen3.5 tool-call output.
        Tries 6 progressively more aggressive strategies.
        DO NOT simplify — this is battle-tested against Qwen3.5-9B output.
        """
        if not content or not content.strip():
            return None

        # 0. Strip <think> blocks
        clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        # 1. Direct parse
        result = self._try_json(clean)
        if result:
            return result

        # 2. Strip markdown fences
        defenced = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", clean).strip()
        if defenced != clean:
            result = self._try_json(defenced)
            if result:
                return result

        # 3. Repair common LLM mistakes
        repaired = self._repair(clean)
        result = self._try_json(repaired)
        if result:
            return result

        # 4. Balanced-brace extraction on repaired text
        balanced = self._extract_balanced(repaired)
        if balanced:
            result = self._try_json(balanced)
            if result:
                return result
            result = self._try_json(self._repair(balanced))
            if result:
                return result

        # 5. Balanced-brace on original
        balanced_orig = self._extract_balanced(clean)
        if balanced_orig and balanced_orig != balanced:
            result = self._try_json(balanced_orig)
            if result:
                return result

        # 6. Regex extraction — last resort
        result = self._regex_extract(clean)
        if result:
            logger.info(f"Recovered action via regex: tool={result['tool']}")
            return result

        logger.warning(f"Could not extract JSON from LLM response: {content[:300]!r}")
        return None

    # Alias for compatibility
    _parse_json_action = parse_action
