"""
WzrdMerlin v2 — Model Interface

Routes LLM calls through llama-server (OpenAI-compatible /v1/chat/completions)
managed by LlamaCppManager.  Falls back to Ollama if INFERENCE_BACKEND=ollama.

Robust JSON parser handles common Qwen3.5 output mistakes.
"""
import os
import logging
import json
import math
import re
import time
from typing import List, Dict, Any, Optional, AsyncIterator, Tuple, TYPE_CHECKING

import httpx

from src.core.config import get_config

if TYPE_CHECKING:
    from src.core.inference import LlamaCppManager

logger = logging.getLogger(__name__)


class ModelInterface:
    """
    Unified LLM interface supporting two backends:
      - llama-server (default): OpenAI-compatible /v1/chat/completions
      - ollama (fallback): Ollama /api/chat

    When backed by LlamaCppManager, auto-starts the server and uses
    dual-backend routing (ROCm for prefill, Vulkan for decode).
    """

    def __init__(
        self,
        model_name: str = None,
        inference_manager: "LlamaCppManager" = None,
    ):
        cfg = get_config()
        active = cfg.get_active_model()

        # INFERENCE_BACKEND env var overrides config (backward compatibility)
        self.backend_type = os.getenv(
            "INFERENCE_BACKEND", cfg.inference.backend
        ).lower()
        self.inference_manager = inference_manager

        # Model name: constructor arg > config profile > env fallback
        self._model_name_override = model_name

        if self.backend_type == "ollama":
            self.model_name = model_name or active.model_name
            self.api_base = os.getenv("OLLAMA_BASE_URL", cfg.ollama.base_url)
        else:
            self.model_name = model_name or active.model_name
            self.api_base = os.getenv(
                "LLAMA_API_BASE",
                f"http://127.0.0.1:{cfg.inference.port}",
            )

        self.context_window = active.context_window
        self.think = active.think
        self.think_budget = active.think_budget
        self.temperature = active.temperature
        # Hard cap for agent tool-call generation to avoid endless rambling.
        self.action_max_tokens = int(os.getenv("MERLIN_ACTION_MAX_TOKENS", "4096"))

    def refresh_from_config(self) -> None:
        """
        Re-read model settings from the current active profile.
        Call this after switching models (e.g. via inference_mgr.switch_model()).
        """
        cfg = get_config()
        active = cfg.get_active_model()
        self.backend_type = os.getenv("INFERENCE_BACKEND", cfg.inference.backend).lower()
        if not self._model_name_override:
            self.model_name = active.model_name
        if self.backend_type == "ollama":
            self.api_base = os.getenv("OLLAMA_BASE_URL", cfg.ollama.base_url)
        else:
            self.api_base = os.getenv(
                "LLAMA_API_BASE",
                f"http://127.0.0.1:{cfg.inference.port}",
            )
        self.context_window = active.context_window
        self.think = active.think
        self.think_budget = active.think_budget
        self.temperature = active.temperature
        self.action_max_tokens = int(os.getenv("MERLIN_ACTION_MAX_TOKENS", "4096"))
        logger.info(f"ModelInterface refreshed: model={self.model_name} ctx={self.context_window}")

    # ------------------------------------------------------------------
    # API base resolution
    # ------------------------------------------------------------------

    def _resolve_api_base(self, prompt_tokens: int = 0) -> str:
        """
        Get the correct API base URL.
        If we have an inference manager, use it for dual-backend routing.
        """
        if self.inference_manager:
            backend = self.inference_manager.select_backend(prompt_tokens)
            base = self.inference_manager.get_api_base(backend=backend)
            if base:
                return base
        return self.api_base

    async def _ensure_server(self, prompt_tokens: int = 0) -> str:
        """Ensure the inference server is running, return api_base."""
        if self.inference_manager:
            backend = self.inference_manager.select_backend(prompt_tokens)
            slot = await self.inference_manager.ensure_model(backend=backend)
            return slot.api_base
        return self.api_base

    # ------------------------------------------------------------------
    # Message building (shared between backends)
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        system_prompt: str,
        history: List[Dict[str, Any]],
        instruction: str,
    ) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({
            "role": "user",
            "content": f"Current Objective: {instruction}",
        })
        for msg in history:
            action = msg.get("action")
            if isinstance(action, dict) and action.get("tool", "").startswith("_"):
                messages.append({
                    "role": "user",
                    "content": f"[Context summary from earlier steps]\n{msg.get('result', '')}",
                })
                continue
            messages.append({
                "role": "assistant",
                "content": json.dumps(action) if action else "(no action)",
            })
            messages.append({
                "role": "user",
                "content": f"Result: {msg.get('result', '')}",
            })
        messages.append({
            "role": "user",
            "content": "Generate your next tool call as a JSON object.",
        })
        return messages

    @staticmethod
    def _estimate_prompt_tokens(messages: List[Dict[str, str]]) -> int:
        """Rough token estimate for dual-backend routing."""
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return max(1, math.ceil(total_chars / 4))

    # ------------------------------------------------------------------
    # Payload builders (backend-specific)
    # ------------------------------------------------------------------

    def _build_llama_payload(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
    ) -> dict:
        """Build payload for llama-server /v1/chat/completions."""
        payload: dict = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "temperature": self.temperature,
        }
        if stream:
            payload["max_tokens"] = max(64, self.action_max_tokens)
        elif self.think and self.think_budget > 0:
            payload["max_tokens"] = self.think_budget + 512
        elif self.context_window > 0:
            payload["max_tokens"] = self.context_window
        return payload

    def _build_ollama_payload(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
    ) -> dict:
        """Build payload for Ollama /api/chat."""
        payload: dict = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "keep_alive": -1,
            "think": self.think if stream else False,
            "format": "json",
        }
        options: dict = {"temperature": self.temperature}
        if self.context_window > 0:
            options["num_ctx"] = self.context_window
        if stream:
            options["num_predict"] = max(64, self.action_max_tokens)
        elif self.think and self.think_budget > 0:
            options["num_predict"] = self.think_budget + 512
        payload["options"] = options
        return payload

    # ------------------------------------------------------------------
    # Streaming — llama-server (/v1/chat/completions SSE)
    # ------------------------------------------------------------------

    async def _stream_llama(
        self,
        messages: List[Dict[str, str]],
    ) -> AsyncIterator[Tuple[str, str]]:
        """Stream from llama-server OpenAI-compatible API."""
        prompt_tokens = self._estimate_prompt_tokens(messages)
        api_base = await self._ensure_server(prompt_tokens)
        payload = self._build_llama_payload(messages, stream=True)
        url = f"{api_base}/v1/chat/completions"

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
            ) as client:
                async with client.stream("POST", url, json=payload) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        logger.error(f"llama-server returned {resp.status_code}: {body[:300]}")
                        return

                    # Record activity for watchdog
                    if self.inference_manager:
                        self.inference_manager.record_activity()

                    in_think = False
                    buffer = ""

                    async for raw_line in resp.aiter_lines():
                        line = raw_line.strip()
                        if not line:
                            continue
                        if line == "data: [DONE]":
                            break
                        if not line.startswith("data: "):
                            continue

                        try:
                            chunk = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if not content:
                            finish = choices[0].get("finish_reason")
                            if finish:
                                break
                            continue

                        buffer += content

                        # Parse <think> tags from content stream
                        while buffer:
                            if in_think:
                                end = buffer.find("</think>")
                                if end != -1:
                                    yield ("think", buffer[:end])
                                    buffer = buffer[end + 8:]
                                    in_think = False
                                else:
                                    safe = max(0, len(buffer) - 8)
                                    if safe > 0:
                                        yield ("think", buffer[:safe])
                                        buffer = buffer[safe:]
                                    break
                            else:
                                start = buffer.find("<think>")
                                if start != -1:
                                    if start > 0:
                                        yield ("content", buffer[:start])
                                    buffer = buffer[start + 7:]
                                    in_think = True
                                else:
                                    safe = max(0, len(buffer) - 7)
                                    if safe > 0:
                                        yield ("content", buffer[:safe])
                                        buffer = buffer[safe:]
                                    break

                    if buffer:
                        yield ("think" if in_think else "content", buffer)

                    # Record activity at end
                    if self.inference_manager:
                        self.inference_manager.record_activity()

        except httpx.ConnectError:
            logger.error(f"Cannot connect to llama-server at {api_base}")
        except Exception as e:
            logger.error(f"llama-server streaming failed: {e}")

    # ------------------------------------------------------------------
    # Streaming — Ollama (/api/chat)
    # ------------------------------------------------------------------

    async def _stream_ollama(
        self,
        messages: List[Dict[str, str]],
    ) -> AsyncIterator[Tuple[str, str]]:
        """Stream from Ollama API (legacy fallback)."""
        payload = self._build_ollama_payload(messages, stream=True)

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
            ) as client:
                async with client.stream(
                    "POST",
                    f"{self.api_base}/api/chat",
                    json=payload,
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        logger.error(f"Ollama returned {resp.status_code}: {body[:300]}")
                        return

                    in_think = False
                    buffer = ""

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

                        buffer += delta

                        while buffer:
                            if in_think:
                                end = buffer.find("</think>")
                                if end != -1:
                                    yield ("think", buffer[:end])
                                    buffer = buffer[end + 8:]
                                    in_think = False
                                else:
                                    safe = max(0, len(buffer) - 8)
                                    if safe > 0:
                                        yield ("think", buffer[:safe])
                                        buffer = buffer[safe:]
                                    break
                            else:
                                start = buffer.find("<think>")
                                if start != -1:
                                    if start > 0:
                                        yield ("content", buffer[:start])
                                    buffer = buffer[start + 7:]
                                    in_think = True
                                else:
                                    safe = max(0, len(buffer) - 7)
                                    if safe > 0:
                                        yield ("content", buffer[:safe])
                                        buffer = buffer[safe:]
                                    break

                    if buffer:
                        yield ("think" if in_think else "content", buffer)

        except httpx.ConnectError:
            logger.error(f"Cannot connect to Ollama at {self.api_base}")
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")

    # ------------------------------------------------------------------
    # Public streaming API
    # ------------------------------------------------------------------

    async def generate_action_streaming(
        self,
        system_prompt: str,
        history: List[Dict[str, Any]],
        instruction: str,
    ) -> AsyncIterator[Tuple[str, str]]:
        """
        Streams LLM output as (chunk_type, text) tuples:
          "think"   — content inside <think>...</think>
          "content" — everything outside think blocks

        Routes to llama-server or Ollama based on INFERENCE_BACKEND.
        """
        messages = self._build_messages(system_prompt, history, instruction)

        if self.backend_type == "ollama":
            async for chunk in self._stream_ollama(messages):
                yield chunk
        else:
            async for chunk in self._stream_llama(messages):
                yield chunk

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    async def generate_action(
        self,
        system_prompt: str,
        history: List[Dict[str, Any]],
        instruction: str,
    ) -> Optional[Dict[str, Any]]:
        messages = self._build_messages(system_prompt, history, instruction)

        if self.backend_type == "ollama":
            return await self._generate_action_ollama(messages)
        else:
            return await self._generate_action_llama(messages)

    async def _generate_action_llama(
        self, messages: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        """Non-streaming call to llama-server."""
        prompt_tokens = self._estimate_prompt_tokens(messages)
        api_base = await self._ensure_server(prompt_tokens)
        payload = self._build_llama_payload(messages, stream=False)
        url = f"{api_base}/v1/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code != 200:
                    logger.error(f"llama-server returned {resp.status_code}: {resp.text[:300]}")
                    return None
                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    return None
                content = choices[0].get("message", {}).get("content", "")
                if self.inference_manager:
                    self.inference_manager.record_activity()
                return self.parse_action(content)
        except Exception as e:
            logger.error(f"llama-server generation failed: {e}")
            return None

    async def _generate_action_ollama(
        self, messages: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        """Non-streaming call to Ollama."""
        payload = self._build_ollama_payload(messages, stream=False)

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{self.api_base}/api/chat",
                    json=payload,
                )
                if resp.status_code != 200:
                    logger.error(f"Ollama returned {resp.status_code}: {resp.text[:300]}")
                    return None
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                return self.parse_action(content)
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return None

    async def generate_text(self, prompt: str, max_tokens: int = 300) -> str:
        """Simple text completion — used for context folding summarisation."""
        messages = [{"role": "user", "content": prompt}]

        if self.backend_type == "ollama":
            return await self._generate_text_ollama(messages, max_tokens)
        else:
            return await self._generate_text_llama(messages, max_tokens)

    async def _generate_text_llama(
        self, messages: List[Dict[str, str]], max_tokens: int
    ) -> str:
        api_base = await self._ensure_server()
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{api_base}/v1/chat/completions", json=payload
                )
                if resp.status_code != 200:
                    logger.error(f"generate_text: llama-server returned {resp.status_code}")
                    return "(summarisation failed)"
                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    return "(summarisation failed)"
                return choices[0].get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.error(f"generate_text failed: {e}")
            return "(summarisation failed)"

    async def _generate_text_ollama(
        self, messages: List[Dict[str, str]], max_tokens: int
    ) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "keep_alive": -1,
            "think": False,
            "options": {"temperature": 0.2, "num_predict": max_tokens},
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(f"{self.api_base}/api/chat", json=payload)
                if resp.status_code != 200:
                    logger.error(f"generate_text: Ollama returned {resp.status_code}")
                    return "(summarisation failed)"
                data = resp.json()
                return data.get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.error(f"generate_text failed: {e}")
            return "(summarisation failed)"

    # ==================================================================
    #  Robust JSON parser for LLM output
    #
    #  Handles the common mistakes Qwen3.5 makes:
    #    - \' (escaped single quote, invalid JSON)
    #    - Extra closing braces  }}}
    #    - Literal newlines / tabs inside strings
    #    - Double-escaped backslashes \\\\" from nested code
    #    - Markdown fences around JSON
    #    - Prose before / after the JSON object
    #    - <think> blocks
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
        s = "".join(out)

        return s

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
        """
        Last-resort extraction: pull 'tool' and 'args' with regex.
        """
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
                            args = json.loads(ModelInterface._repair(args_str))
                            return {"tool": tool_name, "args": args}
                        except json.JSONDecodeError:
                            pass
                        break

        val_m = re.search(r'"(?:cmd|path|query|url|summary|content|question)"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if val_m:
            key = re.search(r'"(cmd|path|query|url|summary|content|question)"', text).group(1)
            return {"tool": tool_name, "args": {key: val_m.group(1)}}

        return {"tool": tool_name, "args": {}}

    def parse_action(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Robust JSON extractor tuned for Qwen3.5 tool-call output.
        Tries 6 progressively more aggressive strategies.
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

    _parse_json_action = parse_action
