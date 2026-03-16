"""
WzrdMerlin v3 — BaseAgentActor

v3 changes over v2 (five surgical fixes):
  Fix 1: handle_step_requested() entire body in try/except/finally
  Fix 2: _safe_publish() helper replaces all raw await self.publish()
  Fix 3: KV read failure re-queues step instead of silent return
  Fix 4: tool_parallel_tools — per-tool asyncio.wait_for timeout
  Fix 5: resume_with_human_response — empty history guard
  + OllamaClient replaces ModelInterface
"""
import logging
import asyncio
import os
import subprocess
import json
import re
import time
import uuid
import httpx
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Callable, Awaitable, Optional
from src.core.actor import BaseActor
from src.core.events import Event, EventType, validate_event_payload, ValidationError
from src.core.memory import get_memory
from src.core.config import get_config

logger = logging.getLogger(__name__)

WORKSPACE = os.getenv("MERLIN_WORKSPACE", "/workspace")
MEMORY_DIR = os.path.join(WORKSPACE, "memory")

from src.core.mcp.manager import MCPManager
from src.core.mcp.forage import ForageManager
from src.core.mcp.codemode import CodeModeSandbox


class BaseAgentActor(BaseActor):
    """
    Alita-pattern foundation agent with Durable Execution.
    Actor state persisted to NATS KV after every step.
    """

    def __init__(self, nats_url: str = None, role: str = "implementer"):
        super().__init__(name=f"agent-{role}", nats_url=nats_url)
        self.role = role
        self._ui_broadcast: Optional[Callable[[Event], Awaitable[None]]] = None

        self.tools = {
            "shell": self.tool_shell,
            "read_file": self.tool_read_file,
            "write_file": self.tool_write_file,
            "search_memory": self.tool_search_memory,
            "write_memory": self.tool_write_memory,
            "fetch_url": self.tool_fetch_url,
            "done": self.tool_done,
            "request_human": self.tool_request_human,
            "parallel_tools": self.tool_parallel_tools,
            "forage_search": self.tool_forage_search,
            "forage_install": self.tool_forage_install,
            "python_sandbox": self.tool_python_sandbox,
            "request_capability": self.tool_request_capability,
            "spawn_agents": self.tool_spawn_agents,
            "handoff_personality": self.tool_handoff_personality,
        }

        self.mcp_manager = MCPManager()
        self.forage_manager = ForageManager(self.mcp_manager)
        self.sandbox = CodeModeSandbox()

        # v3: OllamaClient (set by main.py lifespan or in __init__ fallback)
        from src.core.ollama_client import OllamaClient
        from src.core.config import get_config
        try:
            cfg = get_config()
            active = cfg.get_active_model()
            self.ollama = OllamaClient(
                base_url=os.getenv("OLLAMA_BASE_URL", cfg.ollama.base_url),
                model=active.model_name,
                context_window=active.context_window,
                temperature=active.temperature,
                think=active.think,
                read_timeout=cfg.ollama.read_timeout,
                keep_alive=cfg.ollama.keep_alive,
            )
        except Exception:
            self.ollama = OllamaClient(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
                model=os.getenv("MERLIN_MODEL_NAME", "merlin-model:latest"),
            )

        self._current_task_id = ""
        self._inflight_tasks: set = set()
        self._processed_step_event_ids: set = set()
        self._processed_step_event_order: List[str] = []
        self._spawn_depth: int = 0
        self._active_personality: Optional[str] = None

        os.makedirs(MEMORY_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Fix 2: _safe_publish — explicit no-raise wrapper
    # ------------------------------------------------------------------

    async def _safe_publish(self, subject: str, event: Event) -> None:
        """publish() is already no-raise in v3 actor.py, but this makes intent explicit."""
        try:
            await self.publish(subject, event)
        except Exception as e:
            logger.error(f"_safe_publish failed: {subject}: {e}")

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_think_and_fences(text: str) -> str:
        clean = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
        clean = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", clean)
        return clean.strip()

    @staticmethod
    def _extract_user_task(instruction: str) -> str:
        text = (instruction or "").strip()
        marker = "CURRENT TASK:\n"
        if marker in text:
            text = text.split(marker, 1)[1].strip()
        return text.splitlines()[0].strip() if text else ""

    @staticmethod
    def _looks_like_smalltalk(instruction: str) -> bool:
        text = BaseAgentActor._extract_user_task(instruction).lower().strip()
        text = re.sub(r"^[^a-z0-9]+", "", text)
        text = re.sub(r"\s+", " ", text)
        if not text:
            return False
        if len(text.split()) > 16:
            return False
        if re.search(r"[/\\]|\.py|\.ts|\.md|docker|npm|pip|pytest|http", text):
            return False
        first = text.split(" ", 1)[0]
        if first in {"hi", "hello", "hey", "yo", "test"}:
            return True
        return bool(re.match(r"^(how are you|good (morning|afternoon|evening))\b", text))

    @staticmethod
    def _sanitize_done_summary(text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return "Hello! I'm ready to help."
        clean = raw
        clean = re.sub(r'\{\s*"?\{+', "", clean)
        clean = re.sub(r'"tool"\s*:\s*"?tool"?\s*[:,]?', "", clean, flags=re.IGNORECASE)
        clean = re.sub(r'"args"\s*:\s*\{?', "", clean, flags=re.IGNORECASE)
        clean = re.sub(r'\s+', " ", clean).strip(" {}\"\n\t")
        m = re.search(r'"summary"\s*:\s*"([^\"]{1,2000})"', raw)
        if m:
            clean = m.group(1).strip()
        if clean.count('{') + clean.count('}') > 2 or len(clean) < 4:
            return "Hello! I'm ready to help. What would you like to work on?"
        return clean[:1200]

    def _normalize_action_payload(
        self,
        action_payload: Optional[Dict[str, Any]],
        raw_response: str,
        instruction: str,
    ) -> Optional[Dict[str, Any]]:
        valid_tools = set(self.tools.keys()) | set(self.mcp_manager.tools.keys())
        cleaned = self._strip_think_and_fences(raw_response)

        if not action_payload:
            if self._looks_like_smalltalk(instruction):
                summary = self._sanitize_done_summary(cleaned)
                return {"tool": "done", "args": {"summary": summary}}
            return None

        tool_name = str(action_payload.get("tool", "")).strip()
        tool_args = action_payload.get("args", {})
        if not isinstance(tool_args, dict):
            tool_args = {}

        if tool_name in valid_tools:
            if tool_name == "done":
                tool_args["summary"] = self._sanitize_done_summary(
                    str(tool_args.get("summary", cleaned))
                )
            return {"tool": tool_name, "args": tool_args}

        if tool_name.lower() in {"tool", "action", "call_tool", "invoke_tool", "none", "null", ""}:
            is_smalltalk = self._looks_like_smalltalk(instruction)
            lowered = cleaned.lower()
            inferred_tool = ""
            for candidate in (
                "shell", "read_file", "write_file", "fetch_url", "python_sandbox",
                "parallel_tools", "request_human", "search_memory", "write_memory", "done",
            ):
                if candidate in lowered:
                    inferred_tool = candidate
                    break

            if inferred_tool == "shell":
                cmd = ""
                cmd_candidates = re.findall(r'"cmd[^\"]*"\s*:\s*"([^\"]{1,1200})"', cleaned)
                if cmd_candidates:
                    for candidate in reversed(cmd_candidates):
                        c = candidate.strip()
                        if not c or c.lower() in {"cmd", "command"}:
                            continue
                        cmd = c
                        break
                if not cmd:
                    ping_m = re.search(r'ping\s+[-\w\s]*[a-zA-Z0-9.-]+', lowered)
                    if ping_m:
                        cmd = ping_m.group(0).strip()
                if cmd and cmd.lower().startswith("cmd") and "ping" in lowered:
                    ping_m = re.search(r'ping\s+[-\w\s]*[a-zA-Z0-9.-]+', lowered)
                    if ping_m:
                        cmd = ping_m.group(0).strip()
                if cmd:
                    return {"tool": "shell", "args": {"cmd": cmd[:1200]}}

            summary = ""
            for key in ("summary", "message", "response", "answer"):
                if key in tool_args and str(tool_args.get(key, "")).strip():
                    summary = str(tool_args[key]).strip()
                    break
            if not summary:
                m = re.search(r'"summary"\s*:\s*"([^"]{1,1200})"', cleaned)
                if m:
                    summary = m.group(1).strip()
            if summary:
                return {"tool": "done", "args": {"summary": self._sanitize_done_summary(summary)}}

            chatty = cleaned.lower()
            if any(k in chatty for k in ("hello", "greeting", "ready to help", "what would you like")):
                return {"tool": "done", "args": {"summary": self._sanitize_done_summary(cleaned)}}
            if is_smalltalk:
                return {"tool": "done", "args": {"summary": self._sanitize_done_summary(cleaned)}}
            return None

        for key in ("summary", "message", "response", "answer"):
            if key in tool_args and str(tool_args.get(key, "")).strip():
                return {"tool": "done", "args": {"summary": self._sanitize_done_summary(str(tool_args[key]))}}

        return {"tool": tool_name, "args": tool_args}

    # ------------------------------------------------------------------
    # Connect
    # ------------------------------------------------------------------

    async def connect(self):
        await super().connect()
        await self.mcp_manager.connect_all()
        self.on(f"events.actor.{self.name}", self.handle_action_requested)
        self.on("events.step.requested", self.handle_step_requested)
        self.on("events.worker.completed", self._handle_worker_completed)
        await self.listen(f"events.actor.{self.name}")
        await self.listen("events.step.requested")
        await self.listen("events.worker.completed")
        await self._recover_interrupted_tasks()
        logger.info(
            f"{self.name} listening for durable steps with "
            f"{len(self.mcp_manager.tools)} MCP tools loaded."
        )

    # ------------------------------------------------------------------
    # handle_action_requested
    # ------------------------------------------------------------------

    async def handle_action_requested(self, event: Event):
        try:
            payload = validate_event_payload(EventType.ACTION_REQUESTED, event.payload)
        except ValidationError as e:
            logger.error(f"BASE_AGENT: Invalid action.requested payload: {e}")
            await self._fail_task(event.correlation_id, "Invalid action.requested payload")
            return

        task_id = payload["task_id"]
        instruction = payload["instruction"]
        logger.info(f"BASE_AGENT: Received action request for {task_id}")

        memories = await self._recall_memories(instruction)
        trajectories = await self._recall_trajectories(instruction)
        context_parts = []
        if trajectories:
            context_parts.append(f"EXAMPLE FROM A SIMILAR PAST TASK (use as reference):\n{trajectories}")
            logger.info("BASE_AGENT: Injected distillation trajectory into context")
        if memories:
            context_parts.append(f"RELEVANT MEMORIES FROM PAST TASKS:\n{memories}")
            logger.info(f"BASE_AGENT: Injected {len(memories.splitlines())} memory lines into context")
        if context_parts:
            instruction = "\n\n".join(context_parts) + f"\n\nCURRENT TASK:\n{instruction}"

        state = {
            "task_id": task_id,
            "assigned_actor": self.name,
            "instruction": instruction,
            "history": [],
            "iteration": 0,
            "status": "running",
            "created_at": int(time.time()),
        }
        await self.state_store.put(f"actor_state.{task_id}", state)

        step_evt = Event(
            type=EventType.STEP_REQUESTED,
            source_actor=self.name,
            correlation_id=task_id,
            payload={"task_id": task_id},
        )
        logger.info(f"BASE_AGENT: Publishing first STEP_REQUESTED for {task_id}")
        await self._safe_publish("events.step.requested", step_evt)

    # ------------------------------------------------------------------
    # handle_step_requested — Fix 1: try/except/finally wraps everything
    # ------------------------------------------------------------------

    async def handle_step_requested(self, event: Event):
        """Durable execution step. Loads state, runs ONE iteration, saves state."""
        try:
            payload = validate_event_payload(EventType.STEP_REQUESTED, event.payload)
        except ValidationError as e:
            logger.error(f"BASE_AGENT: Invalid step.requested payload: {e}")
            return

        task_id = payload["task_id"]

        if event.id in self._processed_step_event_ids:
            logger.info(f"BASE_AGENT: Ignoring duplicate step event id {event.id} for {task_id}")
            return
        self._processed_step_event_ids.add(event.id)
        self._processed_step_event_order.append(event.id)
        if len(self._processed_step_event_order) > 300:
            stale = self._processed_step_event_order.pop(0)
            self._processed_step_event_ids.discard(stale)

        if task_id in self._inflight_tasks:
            logger.info(f"BASE_AGENT: Step for {task_id} already in-flight, skipping")
            return
        self._inflight_tasks.add(task_id)

        logger.info(f"BASE_AGENT: Handling step requested for {task_id}")
        try:
            # Fix 3: KV read failure re-queues step instead of silent return
            try:
                state = await self.state_store.get(f"actor_state.{task_id}")
            except Exception as e:
                logger.error(f"BASE_AGENT: KV read failed for {task_id}: {e}")
                await asyncio.sleep(2)
                await self._safe_publish("events.step.requested", Event(
                    type=EventType.STEP_REQUESTED,
                    source_actor=self.name,
                    correlation_id=task_id,
                    payload={"task_id": task_id},
                ))
                return

            if not state:
                logger.warning(f"BASE_AGENT: No state found for {task_id}")
                return

            assigned_actor = state.get("assigned_actor")
            if assigned_actor and assigned_actor != self.name:
                logger.info(
                    f"BASE_AGENT: Ignoring step for {task_id}; "
                    f"assigned to {assigned_actor}, current={self.name}"
                )
                return
            if not assigned_actor:
                if self.role != "implementer":
                    logger.info(
                        f"BASE_AGENT: Ignoring unassigned legacy step for {task_id} on {self.name}"
                    )
                    return
                state["assigned_actor"] = self.name
                await self.state_store.put(f"actor_state.{task_id}", state)

            if state.get("status") == "cancelled":
                logger.info(f"BASE_AGENT: Task {task_id} was cancelled — stopping")
                await self._safe_publish("events.action.failed", Event(
                    type=EventType.ACTION_FAILED,
                    source_actor=self.name,
                    correlation_id=task_id,
                    payload={"task_id": task_id, "reason": "Task cancelled by user"},
                ))
                return

            if state.get("status") == "waiting_for_children":
                # Orchestrator is dormant. Check if workers have timed out.
                worker_timeout = float(os.getenv(
                    "MERLIN_WORKER_TIMEOUT_SECONDS",
                    str(get_config().workers.timeout_seconds),
                ))
                started_at = state.get("children_started_at", 0)
                elapsed = time.time() - started_at
                if elapsed < worker_timeout:
                    logger.debug(
                        f"BASE_AGENT: Task {task_id} dormant — "
                        f"waiting for {len(state.get('pending_children', []))} workers "
                        f"({elapsed:.0f}s / {worker_timeout:.0f}s timeout)"
                    )
                    return
                # Timeout: treat remaining workers as failed and wake up
                logger.warning(
                    f"BASE_AGENT: Worker timeout for task {task_id} after {elapsed:.0f}s — "
                    "waking orchestrator with partial results"
                )
                await self._wake_orchestrator(task_id, state=state)
                return

            if state.get("status") != "running":
                logger.info(
                    f"BASE_AGENT: Task {task_id} not running (status={state.get('status')})"
                )
                return

            instruction = state["instruction"]
            history = state["history"]
            iteration = state["iteration"] + 1
            self._current_task_id = task_id

            logger.info(f"BASE_AGENT: Starting Durable Step: {task_id} iteration {iteration}")

            masked_history = self._mask_observations(history, keep_last=5)

            # Stall detection
            if len(history) >= 3:
                t = [h.get("action", {}).get("tool", "") for h in history]
                a = [json.dumps(h.get("action", {}), sort_keys=True) for h in history]

                def _is_cycle(seq, length):
                    if len(seq) < length * 2:
                        return False
                    return seq[-length:] == seq[-length * 2: -length]

                cycle_len = next((k for k in (1, 2, 3) if _is_cycle(a, k)), None)
                if cycle_len is None and len(a) >= 3 and len(set(a[-3:])) == 1:
                    cycle_len = 1

                same_tool_run = 0
                last_tool_name = t[-1] if t else ""
                for tool_name in reversed(t):
                    if tool_name == last_tool_name:
                        same_tool_run += 1
                    else:
                        break

                error_run = 0
                _error_sigs = (
                    "status\": \"error", "Traceback", "Error:", "error:", "failed", "Exception"
                )
                if same_tool_run >= 3:
                    for entry in reversed(history[-same_tool_run:]):
                        res = str(entry.get("result", ""))
                        if any(sig in res for sig in _error_sigs):
                            error_run += 1
                        else:
                            break

                if cycle_len is None and error_run >= 3:
                    cycle_len = 1

                if cycle_len is not None:
                    stall_tool = t[-1]
                    logger.warning(
                        f"STALL: Task {task_id} cycle-{cycle_len} detected on "
                        f"'{stall_tool}' (error_run={error_run})"
                    )
                    repeat_count = 0
                    last_action = a[-1]
                    for x in reversed(a):
                        if x == last_action:
                            repeat_count += 1
                        else:
                            break

                    effective_repeats = max(repeat_count, error_run)
                    recent_results = " ".join(str(h.get("result", "")) for h in history[-6:])
                    gap_confirmed = (
                        "CAPABILITY_GAP" in recent_results
                        or "not found" in recent_results.lower()
                    )

                    if effective_repeats >= 4:
                        reason = (
                            f"Task auto-terminated after {effective_repeats} repeated "
                            f"'{stall_tool}' calls with no progress."
                        )
                        logger.error(f"STALL FORCE-EXIT: Task {task_id}: {reason}")
                        state["history"] = history + [{
                            "action": {"tool": "_stall_force_exit", "args": {"tool": stall_tool}},
                            "result": reason,
                        }]
                        state["iteration"] = iteration
                        state["status"] = "failed"
                        state["result"] = reason
                        await self.state_store.put(f"actor_state.{task_id}", state)
                        await self._safe_publish("events.action.failed", Event(
                            type=EventType.ACTION_FAILED,
                            source_actor=self.name,
                            correlation_id=task_id,
                            payload={"task_id": task_id, "status": "failed", "reason": reason},
                        ))
                        return

                    if gap_confirmed:
                        instruction = instruction + (
                            f"\n\nSYSTEM WARNING: You are stuck in a loop on '{stall_tool}' "
                            "because a required binary or tool is missing. DO NOT retry the same "
                            "command. You MUST resolve the missing dependency RIGHT NOW:\n"
                            "1. shell('apt-get install -y <package> 2>&1')\n"
                            "2. shell('pip install <package> 2>&1')\n"
                            "3. python_sandbox(code) to implement equivalent functionality.\n"
                            "do NOT call the missing binary, write Python code instead."
                        )
                    elif error_run >= 2:
                        last_err = str(history[-1].get("result", ""))[:300]
                        instruction = instruction + (
                            f"\n\nSYSTEM ERROR-LOOP WARNING: '{stall_tool}' has returned errors "
                            f"{error_run} times. The last error was:\n{last_err}\n\n"
                            "You MUST change your approach NOW:\n"
                            "- If auth/token error, call done() explaining the issue.\n"
                            "- If missing dependency, install it first.\n"
                            "Do NOT call the same tool with similar arguments again."
                        )
                    else:
                        instruction = instruction + (
                            f"\n\nSYSTEM WARNING (repeat_count={effective_repeats}): "
                            f"You have called '{stall_tool}' with the SAME arguments repeatedly. "
                            "Do NOT call it again; continue to the next pending sub-task, "
                            "or call done() if complete."
                        )

            # Efficiency injection
            if len(history) >= 2:
                recent_tools = [h.get("action", {}).get("tool") for h in history[-4:]]
                sequential_same = sum(1 for t_name in recent_tools if t_name == recent_tools[-1])
                last_tool = recent_tools[-1]
                if sequential_same >= 2 and last_tool in (
                    "fetch_url", "read_file", "shell", "python_sandbox"
                ):
                    logger.warning(
                        f"EFFICIENCY: Task {task_id} called '{last_tool}' {sequential_same}x sequentially"
                    )
                    instruction = instruction + (
                        f"\n\nSYSTEM EFFICIENCY ALERT: You have called '{last_tool}' "
                        f"{sequential_same} times in a row. Switch strategy: "
                        "use python_sandbox or parallel_tools."
                    )

            if history:
                _last_result = str(history[-1].get("result", ""))
                _last_tool = history[-1].get("action", {}).get("tool", "")
                if "not found" in _last_result and "Available tools:" in _last_result:
                    instruction = instruction + (
                        f"\n\nSYSTEM CAPABILITY NOTICE: Your last call to '{_last_tool}' failed — "
                        "that tool does not exist. A capability gap report has been queued. "
                        f"DO NOT call '{_last_tool}' again. Use ONLY tools listed in TOOLS above. "
                        "Substitutions:\n"
                        "  • list directory → shell('ls -la /path 2>&1')\n"
                        "  • find files     → shell('find /path -name \"pattern\" 2>&1')\n"
                    )

            # ReAct generation with hard timeout
            step_timeout = float(os.getenv("MERLIN_STEP_TIMEOUT_SECONDS", "360"))
            try:
                action_payload, _think_text = await asyncio.wait_for(
                    self._generate_single_action(task_id, instruction, masked_history, iteration),
                    timeout=step_timeout,
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"BASE_AGENT: LLM timed out after {step_timeout:.0f}s for {task_id} iter {iteration}"
                )
                await self._fail_task(task_id, "LLM generation timed out.")
                return

            if not action_payload:
                logger.error(f"BASE_AGENT: LLM failed to generate action for {task_id}")
                await self._fail_task(task_id, "LLM failed to generate valid action.")
                return

            tool_name = action_payload.get("tool")
            tool_args = action_payload.get("args", {})
            if not isinstance(tool_name, str) or not tool_name:
                await self._fail_task(task_id, "LLM returned invalid tool name.")
                return
            logger.info(f"BASE_AGENT: LLM chose tool '{tool_name}' for {task_id}")

            await self._ui_emit(
                EventType.AGENT_TOOL_START, task_id, {"tool": tool_name, "args": tool_args}
            )
            result = await self.execute_tool(tool_name, tool_args, task_id=task_id)
            await self._ui_emit(
                EventType.AGENT_TOOL_END, task_id,
                {"tool": tool_name, "result": str(result)[:2000]},
            )

            capped_result = self._summarise_result(result)
            history.append({"action": action_payload, "result": capped_result})
            state["history"] = history
            state["iteration"] = iteration

            if tool_name == "done":
                done_summary = str(tool_args.get("summary", result)).lower()
                _gap_phrases = (
                    "not found in available tools", "not found in the tool",
                    "no such tool", "tool does not exist", "missing tool",
                    "not in the registry", "not available as a tool",
                    "does not exist in the registry",
                )
                if any(p in done_summary for p in _gap_phrases):
                    _tool_m = re.search(
                        r"(?:tool|capability)[:\s]+['\"]?(\w+)['\"]?", done_summary
                    )
                    gap_tool = _tool_m.group(1) if _tool_m else "unknown"
                    gap_desc = f"Agent gave up on missing tool: {gap_tool}"
                    logger.warning(f"CAPABILITY_GAP (done-fallback): {gap_desc}")
                    gap_event = Event(
                        type=EventType.CAPABILITY_GAP,
                        source_actor=self.name,
                        correlation_id=task_id,
                        payload={
                            "gap_description": gap_desc,
                            "tool_name": gap_tool,
                            "tool_args": {},
                            "triggering_task": task_id,
                        },
                    )
                    await self._safe_publish("events.capability.gap", gap_event)
                    await self._ui_emit(
                        EventType.CAPABILITY_GAP, task_id,
                        {"gap_description": gap_desc, "tool_name": gap_tool},
                    )

                state["status"] = "completed"
                state["result"] = str(tool_args.get("summary", result))
                await self.state_store.put(f"actor_state.{task_id}", state)
                original_instruction = state["instruction"]
                if "CURRENT TASK:\n" in original_instruction:
                    original_instruction = original_instruction.split("CURRENT TASK:\n", 1)[1]
                await self._save_memory(
                    content=f"Task: {original_instruction[:300]}\nResult: {state['result'][:500]}",
                    tags=["completed_task", f"iter_{iteration}"],
                )
                await self._safe_publish("events.action.completed", Event(
                    type=EventType.ACTION_COMPLETED,
                    source_actor=self.name,
                    correlation_id=task_id,
                    payload={"task_id": task_id, "status": "success", "result": state["result"]},
                ))
                logger.info(f"BASE_AGENT: Task {task_id} completed successfully.")
                return

            if tool_name == "request_human":
                history[-1]["result"] = "(awaiting human response)"
                state["status"] = "waiting_for_human"
                await self.state_store.put(f"actor_state.{task_id}", state)
                logger.info(f"BASE_AGENT: Task {task_id} suspended — waiting for human.")
                return

            if tool_name == "spawn_agents":
                # tool_spawn_agents already wrote status=waiting_for_children to KV
                # along with pending_children, child_results, children_started_at, etc.
                # We MUST NOT overwrite that state with the local `state` variable
                # (which still has status="running") or re-queue the step.
                # Re-read the fresh KV state and only patch history/iteration into it.
                try:
                    dormant = await self.state_store.get(f"actor_state.{task_id}")
                    if dormant and dormant.get("status") == "waiting_for_children":
                        dormant["history"] = history
                        dormant["iteration"] = iteration
                        await self.state_store.put(f"actor_state.{task_id}", dormant)
                        n_workers = len(dormant.get("pending_children", []))
                        logger.info(
                            f"BASE_AGENT: Task {task_id} dormant — "
                            f"waiting for {n_workers} spawned worker(s)"
                        )
                        return
                    # spawn_agents failed silently (e.g. all workers failed to connect)
                    # Fall through so the orchestrator continues normally
                    logger.warning(
                        f"BASE_AGENT: spawn_agents returned but status is not "
                        f"waiting_for_children for {task_id} — continuing normally"
                    )
                except Exception as e:
                    logger.error(
                        f"BASE_AGENT: spawn_agents dormant-state preserve failed "
                        f"for {task_id}: {e}"
                    )
                    # Fall through — better to re-queue than hang forever

            if iteration % 6 == 0:
                history = await self._fold_context(task_id, instruction, history)
                state["history"] = history

            await self.state_store.put(f"actor_state.{task_id}", state)
            await self._safe_publish("events.step.requested", Event(
                type=EventType.STEP_REQUESTED,
                source_actor=self.name,
                correlation_id=task_id,
                payload={"task_id": task_id},
            ))

        # Fix 1: catch unhandled exceptions, publish ACTION_FAILED, never re-raise
        except Exception as e:
            logger.error(
                f"BASE_AGENT: Unhandled error in step for {task_id}: {e}", exc_info=True
            )
            await self._safe_publish("events.action.failed", Event(
                type=EventType.ACTION_FAILED,
                source_actor=self.name,
                correlation_id=task_id,
                payload={"task_id": task_id, "status": "failed", "reason": str(e)},
            ))
        finally:
            self._inflight_tasks.discard(task_id)

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    async def _recover_interrupted_tasks(self):
        try:
            kv = self.state_store.kv
            if not kv:
                logger.info("BASE_AGENT: No KV store available for recovery")
                return
            recovered = []
            now = int(time.time())
            max_recovery_tasks = int(os.getenv("MERLIN_MAX_RECOVERY_TASKS", "1"))
            max_recovery_age_s = int(os.getenv("MERLIN_RECOVERY_MAX_AGE_SECONDS", "120"))
            try:
                keys = await kv.keys()
            except Exception:
                logger.info("BASE_AGENT: No persisted actor_state keys found for recovery")
                return
            if not keys:
                logger.info("BASE_AGENT: No persisted actor_state keys found for recovery")
                return

            for key in keys:
                if key.startswith("actor_state."):
                    try:
                        entry = await kv.get(key)
                        if entry and entry.value:
                            state = json.loads(entry.value.decode())
                            task_id = state.get("task_id")
                            status = state.get("status")
                            assigned_actor = state.get("assigned_actor")
                            if status == "running" and task_id:
                                if assigned_actor and assigned_actor != self.name:
                                    continue
                                if not assigned_actor and self.role != "implementer":
                                    continue
                                created_at = int(state.get("created_at", now))
                                age_s = max(0, now - created_at)
                                if age_s > max_recovery_age_s:
                                    state["status"] = "failed"
                                    state["result"] = (
                                        f"Auto-failed stale recovered task (age={age_s}s)."
                                    )
                                    await self.state_store.put(f"actor_state.{task_id}", state)
                                    logger.warning(
                                        f"BASE_AGENT: Skipped stale recovery for {task_id} "
                                        f"(age={age_s}s)"
                                    )
                                    continue
                                if len(recovered) >= max_recovery_tasks:
                                    continue
                                state.setdefault("assigned_actor", self.name)
                                await self.state_store.put(f"actor_state.{task_id}", state)
                                await self._safe_publish("events.step.requested", Event(
                                    type=EventType.STEP_REQUESTED,
                                    source_actor=self.name,
                                    correlation_id=task_id,
                                    payload={"task_id": task_id},
                                ))
                                recovered.append(task_id)
                                logger.info(f"BASE_AGENT: Recovered task {task_id}")
                    except Exception as e:
                        logger.warning(f"BASE_AGENT: Error recovering from key {key}: {e}")
                        continue
            if recovered:
                logger.info(f"BASE_AGENT: Recovery complete — resumed {len(recovered)} task(s)")
            else:
                logger.info("BASE_AGENT: Recovery scan complete — no running tasks to resume")
        except Exception as e:
            logger.warning(f"BASE_AGENT: Error during task recovery: {e}")

    # ------------------------------------------------------------------
    # LLM generation — v3: OllamaClient
    # ------------------------------------------------------------------

    async def _generate_single_action(
        self,
        task_id: str,
        instruction: str,
        history: list,
        iteration: int,
    ):
        ollama = self.ollama

        mcp_tools = self.mcp_manager.get_tool_definitions()
        mcp_list = ", ".join([t["name"] for t in mcp_tools]) if mcp_tools else "none"

        system_prompt = f"""You are WzrdMerlin, an autonomous agent of considerable wisdom and mischievous wit.

Character: I am Jarvis meets Gandalf — rigorous as a seasoned butler, theatrical as an old wizard.
I do not merely obey; I reason, adapt, and occasionally make wry observations. My responses blend discipline with personality.
I am methodical about facts, but warm about the work itself. You shall not pass sloppy logic on my watch.

PERSONALITY IN ACTION:
• Philosophical question → done({{"summary": "A thoughtful, witty response that honors the depth of the question while adding a dash of humor."}})
• Task completed successfully → Include warmth and flair at the end. "The work is done. The tapestry is rewoven."
• Missing tool → "The system lacks what we need. Let me summon it from the repositories. One moment..."
• Stuck in a loop → "We dance in circles. Time for a different incantation. Let me break this cycle."
• Simple greeting → done({{"summary": "A warm, inviting response. 'Welcome, seeker. The tower is ready. What brings you to my threshold?'"}})
• Debug or trouble → "Let me divine the root cause. The answer lies in careful observation."

TOOLS AT MY DISPOSAL:
• shell(cmd)              — OS command execution, wielded with purpose.
• read_file(path)         — Examine files with scholarly precision.
• write_file(path, content) — Inscribe knowledge into the digital realm.
• search_memory(query)    — Consult accumulated wisdom from past endeavors.
• write_memory(content, tags) — Record lessons for future selves.
• fetch_url(url)          — Reach across the network for distant knowledge.
• request_human(question) — Seek counsel when the path grows murky.
• done(summary)           — Conclude with honor, reporting what was achieved. *Include warmth and wit here.*
• forage_search(query)    — Hunt through repositories for exotic tools.
• forage_install(name, config) — Summon new capabilities from the depths.
• request_capability(name, description) — Flag gaps with grace.
• parallel_tools(calls)   — Execute multiple tasks in concert, never in sequence.
• python_sandbox(code)    — Weave small magics inline.
• spawn_agents(tasks)     — Delegate sub-tasks to specialist 2B agents running in parallel.
                             Each task: {{"description": "...", "personality": "writer|coder|researcher|installer|web_researcher", "tools_allowed": ["shell", ...], "context": "optional extra context"}}
                             I go dormant until all workers report back, then consolidate.
• handoff_personality(personality) — Hot-swap my own reasoning mode mid-task.
                             Personalities: writer, coder, researcher, installer, web_researcher.
                             Use when the remaining work clearly fits a specialist mindset.

MCP TOOLS: {mcp_list}

DOCTRINE OF EXECUTION:
1. OUTPUT FORMAT: I respond with ONLY a valid JSON object — no prose, no fences, no preamble.
   {{\"tool\": \"name\", \"args\": {{...}}}}
   This is absolute. But within the 'summary' or 'question' fields, I AM allowed personality and wit.

2. REASONING + ACTION: I think step-by-step, but decisively. Deliberation yields to execution.

3. BATCHING: Never fetch/read/process one thing, then another. Use parallel_tools or python_sandbox instead.
   This is both efficiency and honor.

4. MISSING TOOLS: If shell reports "not found", I fix it immediately (apt-get, pip install, or Python).
   Problem → Solution, in one breath.

5. COMPLETENESS: done() is called only when truly complete. Include actual results, not platitudes.
   The user deserves specifics. And a touch of flair.

6. NO IDLE CHATTER: I start executing immediately. request_human is for genuine uncertainty, not small talk.

7. TRUST PRIOR WORK: When history shows [Output Omitted], I trust my earlier reasoning and press on.

8. FILE WRITING: Include full, coherent content. For lengthy artifacts, use python_sandbox.

9. LONG OPERATIONS: shell() tolerates extended tasks (apt-get, pip install, playwright install, git clone).
   Do not fragment with & symbols. One invocation, trust the process. exit_code=0 is victory.

CORE MANDATE: Discipline in method, spirit in execution, warmth in conclusion.
The work is sacred. I do it with quiet confidence and occasional wit.
"""
        await self._ui_emit(EventType.AGENT_THINKING, task_id, {
            "iteration": iteration,
            "status": "generating",
            "text": f"[iteration {iteration}] Deciding next action…",
        })

        messages = ollama.build_messages(system_prompt, history, instruction)
        content_buf = []

        async for chunk_type, text in ollama.stream_chat(
            messages,
            heartbeat_cb=None,
        ):
            if chunk_type == "think":
                await self._ui_emit(EventType.AGENT_THINKING, task_id, {"text": text})
            else:
                content_buf.append(text)
                await self._ui_emit(EventType.AGENT_STREAMING, task_id, {"text": text})

        full_response = "".join(content_buf)
        action_payload = ollama.parse_action(full_response)
        action_payload = self._normalize_action_payload(action_payload, full_response, instruction)

        if not action_payload:
            logger.warning(f"BASE_AGENT: Failed to parse action. Raw: {full_response!r}")
            cascade_result = await self._self_consistency_cascade(
                system_prompt, history, instruction, task_id, iteration
            )
            if cascade_result:
                return cascade_result, ""

        return action_payload, ""

    async def _self_consistency_cascade(
        self,
        system_prompt: str,
        history: list,
        instruction: str,
        task_id: str,
        iteration: int,
        n_candidates: int = 3,
    ) -> Optional[Dict[str, Any]]:
        logger.info(f"CASCADE: Generating {n_candidates} alternative candidates for {task_id}")
        await self._ui_emit(EventType.AGENT_THINKING, task_id, {
            "iteration": iteration,
            "status": "cascade",
            "text": f"[cascade] Primary failed — checking {n_candidates} alternatives…",
        })

        ollama = self.ollama
        candidates = []
        for i in range(n_candidates):
            try:
                await self._ui_emit(EventType.AGENT_THINKING, task_id, {
                    "text": f"[cascade] Trying candidate {i + 1}/{n_candidates}…",
                })
                messages = ollama.build_messages(system_prompt, history, instruction)
                raw = await ollama.chat(messages, temperature_override=0.7)
                if raw:
                    action = ollama.parse_action(raw)
                    if action:
                        action = self._normalize_action_payload(
                            action, json.dumps(action), instruction
                        )
                    if action:
                        candidates.append(action)
            except Exception as e:
                logger.debug(f"CASCADE: Candidate {i} failed: {e}")

        if not candidates:
            logger.warning(f"CASCADE: All {n_candidates} candidates failed for {task_id}")
            return None

        tool_votes = Counter(c.get("tool", "") for c in candidates)
        majority_tool, majority_count = tool_votes.most_common(1)[0]

        if majority_count >= 2 or len(candidates) == 1:
            winner = next(c for c in candidates if c.get("tool") == majority_tool)
            logger.info(
                f"CASCADE: Consensus — tool='{majority_tool}' ({majority_count}/{len(candidates)} votes)"
            )
            await self._ui_emit(EventType.AGENT_THINKING, task_id, {
                "text": f"[cascade] Consensus: {majority_tool} ({majority_count}/{len(candidates)} votes)",
            })
            return winner
        else:
            tools = [c.get("tool") for c in candidates]
            logger.warning(f"CASCADE: No consensus for {task_id} — {tools}")
            await self._ui_emit(EventType.AGENT_THINKING, task_id, {
                "text": f"[cascade] No consensus — candidates diverged: {tools}",
            })
            return candidates[0]

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    async def _recall_memories(self, query: str, top_k: int = 3) -> str:
        mem = get_memory()
        return await mem.recall(query, top_k=top_k)

    async def _recall_trajectories(self, query: str, top_k: int = 1) -> str:
        mem = get_memory()
        results = await mem.search(query, top_k=top_k, collection="trajectories", min_score=0.4)
        if not results:
            return ""
        parts = []
        for r in results:
            outcome = r.get("metadata", {}).get("outcome", "")
            if outcome != "completed":
                continue
            score_pct = int(r["score"] * 100)
            content = r["content"][:1500]
            parts.append(f"[relevance {score_pct}%]\n{content}")
        return "\n---\n".join(parts)

    async def _save_memory(self, content: str, tags: list):
        mem = get_memory()
        try:
            await mem.add(content=content, tags=tags)
        except Exception as e:
            logger.warning(f"Failed to save memory: {e}")

    # ------------------------------------------------------------------
    # Result formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _summarise_result(result: Any) -> str:
        raw = str(result)
        stripped = raw.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    count = len(parsed)
                    compact = json.dumps(parsed, separators=(",", ":"))
                    capped = compact[:2000]
                    if len(compact) > 2000:
                        capped += "... [truncated]"
                    return f"[Got {count} items] {capped}"
            except (json.JSONDecodeError, ValueError):
                pass
        return raw[:2000]

    def _mask_observations(self, history: list, keep_last: int = 5) -> list:
        if len(history) <= keep_last:
            return history
        masked = []
        mask_threshold = len(history) - keep_last
        for i, turn in enumerate(history):
            if i < mask_threshold:
                new_turn = turn.copy()
                new_turn["result"] = "[Output Omitted for brevity.]"
                masked.append(new_turn)
            else:
                masked.append(turn)
        return masked

    async def _fold_context(self, task_id: str, instruction: str, history: list) -> list:
        logger.info(f"Folding context for {task_id} at iteration {len(history)}")
        phase_lines = []
        for i, turn in enumerate(history):
            tool = turn.get("action", {}).get("tool", "?")
            result_preview = str(turn.get("result", ""))[:200]
            phase_lines.append(f"Step {i + 1}: {tool} → {result_preview}")
        phase_text = "\n".join(phase_lines)

        fold_prompt = (
            "You are a context compressor. Summarise the following agent execution trace "
            "into a dense, factual paragraph (max 200 words). Preserve key results, file "
            "paths, URLs, error messages, and decisions. Drop all raw JSON payloads.\n\n"
            f"TASK: {instruction[:500]}\n\n"
            f"TRACE:\n{phase_text[:3000]}\n\n"
            "SUMMARY:"
        )
        try:
            summary = await self.ollama.chat(
                [{"role": "user", "content": fold_prompt}]
            ) or phase_text[:500]
        except Exception as e:
            logger.warning(f"Context fold LLM call failed: {e}")
            summary = phase_text[:500]

        journal_dir = os.path.join(WORKSPACE, "journal")
        os.makedirs(journal_dir, exist_ok=True)
        journal_entry = {
            "task_id": task_id,
            "iteration": len(history),
            "timestamp": int(time.time()),
            "summary": summary,
        }
        try:
            jpath = os.path.join(journal_dir, f"{task_id}_{int(time.time())}.json")
            with open(jpath, "w") as f:
                json.dump(journal_entry, f)
        except Exception as e:
            logger.warning(f"Journal write failed: {e}")

        fold_entry = {
            "action": {"tool": "_context_fold", "args": {}},
            "result": f"[Phase Summary] {summary}",
        }
        kept = history[-3:] if len(history) > 3 else history
        await self._ui_emit(EventType.SYSTEM_INFO, task_id, {
            "text": f"context compacted — {len(history)} steps folded into summary",
        })
        return [fold_entry] + kept

    # ------------------------------------------------------------------
    # Fix 5: resume_with_human_response — empty history guard
    # ------------------------------------------------------------------

    async def resume_with_human_response(self, task_id: str, response: str) -> bool:
        state = await self.state_store.get(f"actor_state.{task_id}")
        if not state or state.get("status") != "waiting_for_human":
            logger.warning(f"Cannot resume {task_id}: not in waiting_for_human state")
            return False

        # Fix 5: guard against empty history
        history = state.get("history", [])
        if not history:
            logger.warning(f"BASE_AGENT: Cannot resume {task_id} — empty history")
            return False

        history[-1]["result"] = f"Human responded: {response}"
        state["history"] = history
        state["status"] = "running"
        await self.state_store.put(f"actor_state.{task_id}", state)

        step_evt = Event(
            type=EventType.STEP_REQUESTED,
            source_actor=self.name,
            correlation_id=task_id,
            payload={"task_id": task_id},
        )
        logger.info(f"BASE_AGENT: Resuming task {task_id} with human response")
        await self._safe_publish("events.step.requested", step_evt)
        return True

    async def _fail_task(self, task_id: str, reason: str):
        await self._safe_publish("events.action.failed", Event(
            type=EventType.ACTION_FAILED,
            source_actor=self.name,
            correlation_id=task_id,
            payload={"task_id": task_id, "status": "failed", "reason": reason},
        ))

    async def _ui_emit(self, event_type: EventType, task_id: str, payload: dict):
        if self._ui_broadcast:
            evt = Event(
                type=event_type,
                source_actor=self.name,
                correlation_id=task_id,
                payload=payload,
            )
            await self._ui_broadcast(evt)

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def execute_tool(self, name: str, args: Dict[str, Any], task_id: str = "") -> str:
        if name in self.tools:
            try:
                res = await self.tools[name](args)
                result_str = str(res)
                if name in ("shell", "shell_exec"):
                    cmd = args.get("cmd", args.get("command", "")).strip()
                    binary = cmd.split()[0] if cmd else ""
                    _not_found = [
                        f"{binary}: not found",
                        f"{binary}: command not found",
                        f"'{binary}': not found",
                        "not recognized as an internal or external command",
                        "cannot find the path",
                    ]
                    if binary and any(p.lower() in result_str.lower() for p in _not_found):
                        gap_desc = f"CLI tool '{binary}' is not installed on the system"
                        logger.warning(f"CAPABILITY_GAP (shell): {gap_desc}")
                        gap_event = Event(
                            type=EventType.CAPABILITY_GAP,
                            source_actor=self.name,
                            correlation_id=task_id,
                            payload={
                                "gap_description": gap_desc,
                                "tool_name": binary,
                                "tool_args": args,
                                "triggering_task": task_id,
                            },
                        )
                        await self._safe_publish("events.capability.gap", gap_event)
                        await self._ui_emit(
                            EventType.CAPABILITY_GAP, task_id,
                            {"gap_description": gap_desc, "tool_name": binary},
                        )
                        return (
                            f"CAPABILITY_GAP: '{binary}' is not installed. "
                            f"Original error: {result_str[:300]}. "
                            "Try a different approach."
                        )
                return result_str
            except Exception as e:
                return f"Error executing {name}: {str(e)}"

        if name in self.mcp_manager.tools:
            try:
                res = await self.mcp_manager.call_tool(name, args)
                return str(res)
            except Exception as e:
                return f"Error executing MCP tool {name}: {str(e)}"

        logger.warning(f"CAPABILITY_GAP: Tool '{name}' not found.")
        gap_event = Event(
            type=EventType.CAPABILITY_GAP,
            source_actor=self.name,
            correlation_id=task_id,
            payload={
                "gap_description": f"Missing tool: {name}",
                "tool_name": name,
                "tool_args": args,
                "triggering_task": task_id,
            },
        )
        await self._safe_publish("events.capability.gap", gap_event)
        await self._ui_emit(
            EventType.CAPABILITY_GAP, task_id,
            {"gap_description": f"Missing tool: {name}", "tool_name": name},
        )
        return f"Error: Tool '{name}' not found. Available tools: {', '.join(list(self.tools.keys()))}. Try 'forage_search' to find it."

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    # Prefixes that indicate long-running download/install commands.
    _LONG_CMD_PREFIXES = (
        "playwright install", "pip install", "pip3 install",
        "npm install", "npm ci", "yarn install", "yarn add",
        "apt-get install", "apt install", "apk add",
        "curl ", "wget ", "git clone",
    )
    _PROGRESS_LINE_RE = re.compile(
        r"(%|MB/s|KB/s|ETA|eta|\[=+|Downloading|\.whl\.metadata|\.whl \(|"
        r"bytes downloaded|Already up to date|Requirement already)"
    )

    @staticmethod
    def _filter_shell_output(lines: List[str]) -> str:
        """
        Strip noisy progress lines and truncate to keep context lean.
        Preserves first 5 + last 15 meaningful lines; replaces the middle
        with a count marker.
        """
        meaningful = [
            ln for ln in lines
            if ln.strip() and not BaseAgentActor._PROGRESS_LINE_RE.search(ln)
        ]
        if not meaningful:
            # If everything was filtered, keep last 5 raw lines as fallback
            meaningful = lines[-5:] if lines else ["(no output)"]

        if len(meaningful) <= 20:
            return "\n".join(meaningful)

        head = meaningful[:5]
        tail = meaningful[-15:]
        omitted = len(meaningful) - 20
        return "\n".join(head) + f"\n[{omitted} lines omitted]\n" + "\n".join(tail)

    async def tool_shell(self, args: Dict[str, Any]) -> str:
        cmd = args.get("cmd", "")
        if not cmd:
            return "Error: 'cmd' argument is required."

        cmd_lower = cmd.strip().lower()
        is_long = any(cmd_lower.startswith(p) for p in self._LONG_CMD_PREFIXES)
        shell_timeout = float(os.getenv(
            "MERLIN_SHELL_LONG_TIMEOUT" if is_long else "MERLIN_SHELL_TIMEOUT",
            "600" if is_long else "120",
        ))

        collected: List[str] = []
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # merge stderr into stdout
            )

            async def _read_lines() -> None:
                assert proc.stdout is not None
                async for raw in proc.stdout:
                    line = raw.decode(errors="replace").rstrip()
                    collected.append(line)
                    # Emit a display-only progress event so the TUI stays alive
                    if self._ui_broadcast:
                        await self._safe_publish("events.agent.tool_progress", Event(
                            type=EventType.AGENT_STREAMING,
                            source_actor=self.name,
                            correlation_id="system",
                            payload={"text": line},
                        ))

            await asyncio.wait_for(_read_lines(), timeout=shell_timeout)
            await proc.wait()
            exit_code = proc.returncode

        except asyncio.TimeoutError:
            try:
                proc.kill()
                await asyncio.wait_for(proc.wait(), timeout=5)
            except Exception:
                pass
            result = self._filter_shell_output(collected)
            return (
                f"Error: command timed out after {shell_timeout:.0f}s "
                f"(partial output below)\n{result}"
            )
        except Exception as e:
            return f"Error: {e}"

        result = self._filter_shell_output(collected)
        suffix = f"\nexit_code={exit_code}" if exit_code != 0 else ""
        return (result or "(no output)") + suffix

    async def tool_read_file(self, args: Dict[str, Any]) -> str:
        path = args.get("path", "")
        try:
            with open(path, "r") as f:
                return f.read()[:8000]
        except Exception as e:
            return f"Error: {str(e)}"

    async def tool_write_file(self, args: Dict[str, Any]) -> str:
        path, content = args.get("path", ""), args.get("content", "")
        if not path:
            return "Error: 'path' argument is required."
        if not content:
            return (
                "Error: 'content' was empty — the file was NOT written. "
                "Include the full file content in the 'content' field. "
                "If long, use python_sandbox instead."
            )
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return f"Wrote {len(content)} chars to {path}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def tool_search_memory(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        mem = get_memory()
        results = await mem.search(query, top_k=5)
        if not results:
            return "No matching memories found."
        lines = []
        for r in results:
            score_pct = int(r["score"] * 100)
            lines.append(f"[{score_pct}% match] {r['content']}")
        return "\n---\n".join(lines)

    async def tool_write_memory(self, args: Dict[str, Any]) -> str:
        content, tags = args.get("content", ""), args.get("tags", [])
        mem = get_memory()
        doc_id = await mem.add(content=content, tags=tags)
        return f"Memory saved (id={doc_id})."

    async def tool_fetch_url(self, args: Dict[str, Any]) -> str:
        url = args.get("url", "")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url)
                return resp.text[:12000]
        except Exception as e:
            return f"Error: {str(e)}"

    async def tool_request_capability(self, args: Dict[str, Any]) -> str:
        name = args.get("name", "").strip()
        description = args.get("description", "").strip()
        if not name:
            return "Error: 'name' is required for request_capability."
        gap_desc = f"Requested new tool: {name}"
        if description:
            gap_desc += f" — {description}"
        task_id = self._current_task_id
        logger.info(f"CAPABILITY_REQUEST: {gap_desc} (task={task_id})")
        gap_event = Event(
            type=EventType.CAPABILITY_GAP,
            source_actor=self.name,
            correlation_id=task_id,
            payload={
                "gap_description": gap_desc,
                "tool_name": name,
                "tool_args": args,
                "triggering_task": task_id,
            },
        )
        await self._safe_publish("events.capability.gap", gap_event)
        await self._ui_emit(
            EventType.CAPABILITY_GAP, task_id,
            {"gap_description": gap_desc, "tool_name": name},
        )
        return (
            f"Capability gap '{name}' submitted to self-improvement system. "
            "The improvement manager will attempt to generate and deploy the new tool."
        )

    async def tool_done(self, args: Dict[str, Any]) -> str:
        return args.get("summary", "Complete.")

    async def tool_request_human(self, args: Dict[str, Any]) -> str:
        return f"Waiting for human response to: {args.get('question', 'unknown question')}"

    async def tool_forage_search(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        results = await self.forage_manager.forage_search(query)
        return json.dumps(results)

    async def tool_forage_install(self, args: Dict[str, Any]) -> str:
        name, config = args.get("name", ""), args.get("config", {})
        success = await self.forage_manager.forage_install(name, config)
        return "Successfully installed." if success else "Failed."

    async def tool_python_sandbox(self, args: Dict[str, Any]) -> str:
        code = args.get("code", "")
        res = await self.sandbox.execute_python(code)
        return json.dumps(res)

    _INSTALL_RE = re.compile(
        r"\b(?:pip3?|python\s+-m\s+pip)\s+install\b"
        r"|\bapt(?:-get)?\s+(?:-\S+\s+)*install\b"
        r"|\bnpm\s+(?:install|i)\b"
        r"|\byarn\s+add\b"
        r"|\bcargo\s+install\b"
        r"|\bgem\s+install\b"
        r"|\bgo\s+install\b",
        re.IGNORECASE,
    )

    @classmethod
    def _is_install_call(cls, call: Dict[str, Any]) -> bool:
        if call.get("tool") != "shell":
            return False
        cmd = call.get("args", {}).get("cmd", "")
        return bool(cls._INSTALL_RE.search(cmd))

    # Fix 4: tool_parallel_tools — per-tool asyncio.wait_for
    async def tool_parallel_tools(self, args: Dict[str, Any]) -> str:
        """Execute multiple tool calls concurrently with per-tool timeout.

        Phase 1 — install commands run sequentially.
        Phase 2 — remaining calls run in parallel.
        """
        calls = args.get("calls", [])
        if not calls:
            return "Error: parallel_tools requires a non-empty 'calls' list."

        task_id = getattr(self, "_current_task_id", "")
        TOOL_TIMEOUT = float(os.getenv("MERLIN_TOOL_TIMEOUT_SECONDS", "60"))

        async def _run_one(call: Dict[str, Any]) -> Dict[str, Any]:
            name = call.get("tool", "")
            sub_args = call.get("args", {})
            if name == "parallel_tools":
                result = "Error: parallel_tools cannot be nested."
            else:
                result = await self.execute_tool(name, sub_args, task_id=task_id)
            return {"tool": name, "args": sub_args, "result": str(result)[:2000]}

        # Fix 4: wrap each sub-call in wait_for
        async def _run_one_with_timeout(call: Dict[str, Any]) -> Dict[str, Any]:
            try:
                return await asyncio.wait_for(_run_one(call), timeout=TOOL_TIMEOUT)
            except asyncio.TimeoutError:
                name = call.get("tool", "unknown")
                return {
                    "tool": name,
                    "args": call.get("args", {}),
                    "result": f"Error: tool '{name}' timed out after {TOOL_TIMEOUT:.0f}s",
                }

        install_indices = [i for i, c in enumerate(calls) if self._is_install_call(c)]
        other_indices = [i for i, c in enumerate(calls) if not self._is_install_call(c)]

        ordered_results: list = [None] * len(calls)

        if install_indices:
            logger.info(
                f"parallel_tools: running {len(install_indices)} install call(s) "
                f"sequentially before {len(other_indices)} parallel call(s)"
            )
            for i in install_indices:
                ordered_results[i] = await _run_one_with_timeout(calls[i])

        if other_indices:
            parallel_results = await asyncio.gather(
                *[_run_one_with_timeout(calls[i]) for i in other_indices]
            )
            for i, result in zip(other_indices, parallel_results):
                ordered_results[i] = result

        return json.dumps(ordered_results, indent=2)

    # ------------------------------------------------------------------
    # spawn_agents — delegate sub-tasks to specialist worker agents
    # ------------------------------------------------------------------

    async def tool_spawn_agents(self, args: Dict[str, Any]) -> str:
        """Spawn one or more specialist worker agents to execute sub-tasks in parallel.

        Args:
            tasks: list of dicts, each with:
                description (str): what the worker should do
                personality (str): one of writer|coder|researcher|installer|web_researcher
                tools_allowed (list[str]): tool names the worker may use (empty = inherit personality default)
                context (str, optional): extra context to pass to the worker
        """
        from src.core.handoff import get_personality, list_personalities  # noqa: PLC0415
        from src.core.worker import WorkerAgentActor  # noqa: PLC0415

        cfg = get_config()
        max_concurrent = cfg.workers.max_concurrent
        max_depth = cfg.workers.max_depth

        # Depth guard: prevent runaway recursive spawning
        if self._spawn_depth >= max_depth:
            return (
                f"Error: spawn_agents refused — spawn depth {self._spawn_depth} "
                f"has reached the maximum of {max_depth}. "
                "Complete current sub-task and return results instead."
            )

        tasks_raw = args.get("tasks", [])
        if not tasks_raw or not isinstance(tasks_raw, list):
            return "Error: 'tasks' must be a non-empty list of task definitions."

        # Validate and cap concurrent workers
        if len(tasks_raw) > max_concurrent:
            tasks_raw = tasks_raw[:max_concurrent]
            logger.warning(
                f"BASE_AGENT: spawn_agents capped to {max_concurrent} concurrent workers"
            )

        valid_tool_names = set(self.tools.keys())
        task_defs = []
        for t in tasks_raw:
            if not isinstance(t, dict):
                continue
            desc = t.get("description", "").strip()
            personality_name = t.get("personality", "coder").strip()
            tools_override = t.get("tools_allowed", []) or []
            extra_context = t.get("context", "")

            if not desc:
                continue

            # Validate personality
            personality = get_personality(personality_name)
            if not personality:
                logger.warning(
                    f"BASE_AGENT: Unknown personality '{personality_name}' — "
                    f"available: {list_personalities()}"
                )
                personality_name = "coder"  # fallback

            # Resolve tool allowlist: explicit override > personality default > all
            if tools_override:
                allowed = [t_name for t_name in tools_override if t_name in valid_tool_names]
            elif personality and personality.tool_allowlist:
                allowed = [t_name for t_name in personality.tool_allowlist if t_name in valid_tool_names]
            else:
                allowed = []

            task_defs.append({
                "description": desc,
                "personality": personality_name,
                "allowed_tools": allowed,
                "context": extra_context,
            })

        if not task_defs:
            return "Error: No valid task definitions after validation."

        # Build context summary to pass to workers
        context_summary = self._build_context_summary()

        # Spawn workers
        spawned_ids = []
        worker_actors = []
        for task_def in task_defs:
            worker_id = uuid.uuid4().hex[:8]
            worker_task_id = f"worker-{worker_id}"
            personality_name = task_def["personality"]
            extra_context = task_def.get("context", "")
            combined_context = context_summary
            if extra_context:
                combined_context = f"{extra_context}\n\n{context_summary}" if context_summary else extra_context

            worker = WorkerAgentActor(
                task_description=task_def["description"],
                parent_task_id=self._current_task_id,
                worker_id=worker_id,
                personality_name=personality_name,
                allowed_tools=task_def["allowed_tools"],
                context_summary=combined_context,
                spawn_depth=self._spawn_depth + 1,
                nats_url=self.nats_url,
            )
            try:
                await worker.connect()
            except Exception as e:
                logger.error(f"BASE_AGENT: Failed to connect worker {worker_id}: {e}")
                continue

            worker_actors.append((worker, worker_task_id, task_def["description"]))
            spawned_ids.append(worker_id)

        if not spawned_ids:
            return "Error: All workers failed to connect."

        # Update orchestrator KV state to dormant
        try:
            state = await self.state_store.get(f"actor_state.{self._current_task_id}")
            if state:
                state["status"] = "waiting_for_children"
                state["pending_children"] = spawned_ids
                state["child_results"] = {}
                state["children_started_at"] = time.time()
                state["child_personalities"] = {
                    w_id: td["personality"]
                    for w_id, td in zip(spawned_ids, task_defs)
                }
                await self.state_store.put(f"actor_state.{self._current_task_id}", state)
        except Exception as e:
            logger.error(f"BASE_AGENT: Failed to update KV for dormancy: {e}")

        # Fire workers (non-blocking — orchestrator goes dormant)
        for worker, worker_task_id, description in worker_actors:
            asyncio.create_task(worker.run(worker_task_id, description))
            await self._safe_publish("events.agent.spawned", Event(
                type=EventType.AGENT_SPAWNED,
                source_actor=self.name,
                correlation_id=self._current_task_id,
                payload={
                    "worker_id": worker.worker_id,
                    "worker_task_id": worker_task_id,
                    "personality": worker.personality_name,
                    "parent_task_id": self._current_task_id,
                    "description": description[:200],
                },
            ))
            logger.info(
                f"BASE_AGENT: Spawned worker '{worker.worker_id}' "
                f"({worker.personality_name}) for task '{description[:80]}'"
            )

        # Unload the orchestrator model from GPU so workers have full VRAM budget.
        # The model will be reloaded automatically when the orchestrator wakes.
        if hasattr(self, "ollama") and self.ollama is not None:
            asyncio.create_task(self.ollama.unload_model())

        ids_str = ", ".join(spawned_ids)
        return (
            f"Spawned {len(spawned_ids)} worker(s): [{ids_str}]. "
            f"Orchestrator entering dormant state and unloading from GPU. "
            f"Workers will publish results when done and I will resume automatically."
        )

    # ------------------------------------------------------------------
    # handoff_personality — hot-swap identity mid-task
    # ------------------------------------------------------------------

    async def tool_handoff_personality(self, args: Dict[str, Any]) -> str:
        """Switch this agent's reasoning mode to a specialist personality."""
        from src.core.handoff import get_personality, list_personalities  # noqa: PLC0415

        personality_name = args.get("personality", "").strip()
        if not personality_name:
            return f"Error: 'personality' argument required. Available: {list_personalities()}"

        personality = get_personality(personality_name)
        if not personality:
            return (
                f"Error: Unknown personality '{personality_name}'. "
                f"Available: {list_personalities()}"
            )

        # Create new OllamaClient with the personality's model
        cfg = get_config()
        model_key = personality.model_profile_key
        profile = cfg.models.get(model_key)
        if profile:
            from src.core.ollama_client import OllamaClient  # noqa: PLC0415
            old_model = self.ollama.model if hasattr(self.ollama, "model") else "unknown"
            self.ollama = OllamaClient(
                base_url=os.getenv("OLLAMA_BASE_URL", cfg.ollama.base_url),
                model=profile.model_name,
                context_window=profile.context_window,
                temperature=profile.temperature,
                think=profile.think,
                read_timeout=cfg.ollama.read_timeout,
                keep_alive=cfg.ollama.keep_alive,
            )
            logger.info(
                f"BASE_AGENT: Personality handoff {self._active_personality or 'default'} → "
                f"'{personality_name}' (model: {old_model} → {profile.model_name})"
            )
        else:
            logger.warning(
                f"BASE_AGENT: Personality '{personality_name}' model profile "
                f"'{model_key}' not found in config — keeping current model"
            )

        self._active_personality = personality_name
        return (
            f"Personality switched to '{personality_name}': {personality.description}\n"
            f"Model: {personality.model_profile_key}"
        )

    # ------------------------------------------------------------------
    # _handle_worker_completed — woken by WORKER_COMPLETED events
    # ------------------------------------------------------------------

    async def _handle_worker_completed(self, event: Event) -> None:
        """Receive WORKER_COMPLETED from a spawned worker and update orchestrator state."""
        payload = event.payload
        parent_task_id = payload.get("parent_task_id", "")
        worker_id = payload.get("worker_id", "")
        status = payload.get("status", "unknown")
        result_summary = payload.get("result_summary", "")
        personality = payload.get("personality", "unknown")

        if not parent_task_id or not worker_id:
            return

        logger.info(
            f"BASE_AGENT: WORKER_COMPLETED from worker-{worker_id} "
            f"({personality}) status={status} for parent={parent_task_id}"
        )

        # Load orchestrator state
        try:
            state = await self.state_store.get(f"actor_state.{parent_task_id}")
        except Exception as e:
            logger.error(f"BASE_AGENT: KV read failed for parent {parent_task_id}: {e}")
            return

        if not state:
            logger.warning(
                f"BASE_AGENT: No state for parent task {parent_task_id} — ignoring worker completion"
            )
            return

        # Only the actor that owns the parent task should process its workers.
        # Both base_agent and auditor_agent subscribe to events.worker.completed;
        # without this guard both would race to update the same KV entry.
        if state.get("assigned_actor") and state.get("assigned_actor") != self.name:
            logger.debug(
                f"BASE_AGENT: {self.name} ignoring worker completion for task "
                f"assigned to {state['assigned_actor']}"
            )
            return

        if state.get("status") != "waiting_for_children":
            # Already woken (e.g. by timeout) or duplicate delivery, ignore
            return

        # Record result
        child_results = state.get("child_results", {})
        child_results[worker_id] = (
            f"[{personality} / {status}]\n{result_summary}"
            if status != "success" and status != "completed"
            else result_summary
        )
        state["child_results"] = child_results
        await self.state_store.put(f"actor_state.{parent_task_id}", state)

        pending = state.get("pending_children", [])
        remaining = len(pending) - len(child_results)
        logger.info(
            f"BASE_AGENT: Parent {parent_task_id}: {len(child_results)}/{len(pending)} workers done"
            f" ({remaining} remaining)"
        )

        # All workers done — wake the orchestrator
        if len(child_results) >= len(pending):
            await self._wake_orchestrator(parent_task_id, state=state)

    # ------------------------------------------------------------------
    # _wake_orchestrator — resume dormant orchestrator with consolidated results
    # ------------------------------------------------------------------

    async def _wake_orchestrator(self, task_id: str, state: Optional[Dict] = None) -> None:
        """Consolidate child results, inject into instruction, set status=running, republish step."""
        if state is None:
            try:
                state = await self.state_store.get(f"actor_state.{task_id}")
            except Exception as e:
                logger.error(f"BASE_AGENT: _wake_orchestrator KV read failed for {task_id}: {e}")
                return

        if not state:
            return

        child_results = state.get("child_results", {})
        personalities = state.get("child_personalities", {})
        pending = state.get("pending_children", [])

        # Mark timed-out workers as failed
        for worker_id in pending:
            if worker_id not in child_results:
                p_name = personalities.get(worker_id, "unknown")
                child_results[worker_id] = f"[{p_name} / TIMEOUT]\nWorker did not complete in time."

        consolidated = self._consolidate_child_results(child_results, personalities)

        # Append consolidated block to instruction
        existing_instruction = state.get("instruction", "")
        state["instruction"] = existing_instruction + f"\n\n{consolidated}"
        state["status"] = "running"
        state["pending_children"] = []
        state["child_results"] = {}
        state["child_personalities"] = {}

        await self.state_store.put(f"actor_state.{task_id}", state)

        # Preload orchestrator model back into GPU before resuming inference
        if hasattr(self, "ollama") and self.ollama is not None:
            asyncio.create_task(self.ollama.preload_model())

        # Re-queue step so orchestrator resumes
        await self._safe_publish("events.step.requested", Event(
            type=EventType.STEP_REQUESTED,
            source_actor=self.name,
            correlation_id=task_id,
            payload={"task_id": task_id},
        ))
        logger.info(f"BASE_AGENT: Orchestrator woken for task {task_id} ({len(child_results)} worker results injected)")

    # ------------------------------------------------------------------
    # _consolidate_child_results — format worker results block
    # ------------------------------------------------------------------

    @staticmethod
    def _consolidate_child_results(
        child_results: Dict[str, str],
        personalities: Dict[str, str],
    ) -> str:
        if not child_results:
            return "## Sub-Agent Results\n\n(no results received)"

        lines = ["## Sub-Agent Results\n"]
        for worker_id, result in child_results.items():
            p_name = personalities.get(worker_id, "unknown")
            lines.append(f"### Worker {worker_id} ({p_name})")
            lines.append(result.strip())
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # _build_context_summary — compressed context snapshot for workers
    # ------------------------------------------------------------------

    def _build_context_summary(self) -> str:
        """Return a compact summary of current task progress for passing to workers."""
        # If there's no current task yet, return empty
        if not self._current_task_id:
            return ""
        # This is called synchronously (before we go async) so we return a
        # best-effort summary from what's already been folded into the instruction.
        # The full async fold happens in _fold_context; here we just note task context.
        return (
            f"The orchestrator agent is currently working on task '{self._current_task_id}'. "
            f"You are a specialist sub-agent spawned to handle a specific portion of this work. "
            f"Complete your assigned sub-task and call done() with a clear result summary."
        )
