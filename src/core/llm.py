import os
import logging
from typing import List, Dict, Any, Optional, AsyncIterator, Tuple
import httpx
import json
import re

logger = logging.getLogger(__name__)


class ModelInterface:
    """
    Calls Ollama's /api/chat directly via httpx.
    No LiteLLM — eliminates param-dropping, model-info lookups, and adapter overhead.
    """
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
        self.api_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.context_window = int(os.getenv("OLLAMA_NUM_CTX", "0"))
        self.think = os.getenv("OLLAMA_THINK", "false").lower() == "true"

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
    ) -> dict:
        payload: dict = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "think": self.think,
        }
        options: dict = {"temperature": 0.6}
        if self.context_window > 0:
            options["num_ctx"] = self.context_window
        payload["options"] = options
        return payload

    def _build_messages(
        self,
        system_prompt: str,
        history: List[Dict[str, Any]],
        instruction: str,
    ) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages.append({
                "role": "user",
                "content": f"Past Action: {msg.get('action')}\nResult: {msg.get('result')}",
            })
        messages.append({
            "role": "user",
            "content": f"Current Objective: {instruction}\nGenerate your next tool call as a JSON object.",
        })
        return messages

    # ------------------------------------------------------------------
    # Streaming
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
        """
        messages = self._build_messages(system_prompt, history, instruction)
        payload = self._build_payload(messages, stream=True)

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
    # Non-streaming
    # ------------------------------------------------------------------

    async def generate_action(
        self,
        system_prompt: str,
        history: List[Dict[str, Any]],
        instruction: str,
    ) -> Optional[Dict[str, Any]]:
        messages = self._build_messages(system_prompt, history, instruction)
        payload = self._build_payload(messages, stream=False)

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
        #    Walk char-by-char respecting proper escapes.
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
        # If we ran out of text with depth>0, return what we have and let
        # the caller try to fix it by appending missing braces.
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
        Works even when the JSON is badly broken, as long as the tool
        name and a rough args structure are present.
        """
        tool_m = re.search(r'"tool"\s*:\s*"([^"]+)"', text)
        if not tool_m:
            return None

        tool_name = tool_m.group(1)

        # Try to grab the args value — could be a dict or a string
        args_m = re.search(r'"args"\s*:\s*(\{)', text)
        if args_m:
            # Find balanced braces starting from the args open brace
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
                        # Try to parse the args dict directly
                        try:
                            args = json.loads(args_str)
                            return {"tool": tool_name, "args": args}
                        except json.JSONDecodeError:
                            pass
                        # Try with repairs
                        try:
                            args = json.loads(ModelInterface._repair(args_str))
                            return {"tool": tool_name, "args": args}
                        except json.JSONDecodeError:
                            pass
                        break

        # If we can at least get the tool name, extract a single string arg
        # Pattern: "cmd": "..." or "summary": "..." etc.
        val_m = re.search(r'"(?:cmd|path|query|url|summary|content|question)"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if val_m:
            key = re.search(r'"(cmd|path|query|url|summary|content|question)"', text).group(1)
            return {"tool": tool_name, "args": {key: val_m.group(1)}}

        # Bare minimum: just the tool name with empty args
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

        # 1. Direct parse (fastest path — works when the model is well-behaved)
        result = self._try_json(clean)
        if result:
            return result

        # 2. Strip markdown fences and retry
        defenced = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", clean).strip()
        if defenced != clean:
            result = self._try_json(defenced)
            if result:
                return result

        # 3. Repair common LLM mistakes and retry
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
            # Try repairing the extracted portion too
            result = self._try_json(self._repair(balanced))
            if result:
                return result

        # 5. Balanced-brace on original (before repair, in case repair broke it)
        balanced_orig = self._extract_balanced(clean)
        if balanced_orig and balanced_orig != balanced:
            result = self._try_json(balanced_orig)
            if result:
                return result

        # 6. Regex extraction — last resort, pulls tool name and args separately
        result = self._regex_extract(clean)
        if result:
            logger.info(f"Recovered action via regex: tool={result['tool']}")
            return result

        logger.warning(f"Could not extract JSON from LLM response: {content[:300]!r}")
        return None

    _parse_json_action = parse_action
