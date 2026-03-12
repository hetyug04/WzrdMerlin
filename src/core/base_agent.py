import logging
import asyncio
import os
import subprocess
import json
import re
import time
import httpx
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Callable, Awaitable, Optional
from src.core.actor import BaseActor
from src.core.events import Event, EventType, validate_event_payload, ValidationError

logger = logging.getLogger(__name__)

# Persistent workspace — mounted as a Docker volume so it survives restarts
WORKSPACE = os.getenv("MERLIN_WORKSPACE", "/workspace")
MEMORY_DIR = os.path.join(WORKSPACE, "memory")
AGENT_REQUIREMENTS = os.path.join(WORKSPACE, "requirements-agent.txt")


from src.core.mcp.manager import MCPManager
from src.core.mcp.forage import ForageManager
from src.core.mcp.codemode import CodeModeSandbox

class BaseAgentActor(BaseActor):
    """
    The Alita-pattern foundation agent. 
    Implements Phase 3 Durable Execution:
    - Actor state is persisted to NATS KV after every step.
    - Agent "sleeps" (terminates) between steps to save VRAM/CPU.
    - Observation Masking and Context Folding preserve token budget.
    """
    def __init__(self, nats_url: str = None, role: str = "implementer"):
        super().__init__(name=f"agent-{role}", nats_url=nats_url)
        self.role = role
        self._ui_broadcast: Optional[Callable[[Event], Awaitable[None]]] = None
        
        # Core Toolset
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

            # Phase 2: Autonomous Extensions
            "forage_search": self.tool_forage_search,
            "forage_install": self.tool_forage_install,
            "python_sandbox": self.tool_python_sandbox,
        }
        
        # Phase 2: MCP & Sandbox Managers
        self.mcp_manager = MCPManager()
        self.forage_manager = ForageManager(self.mcp_manager)
        self.sandbox = CodeModeSandbox()
        
        # Cached LLM interface — reuse across iterations to avoid per-call httpx overhead
        from src.core.llm import ModelInterface
        self.llm = ModelInterface()
        
        # Current executing task id — set at start of each step, used by parallel_tools
        self._current_task_id = ""
        # Reliability guards
        self._inflight_tasks: set[str] = set()
        self._processed_step_event_ids: set[str] = set()
        self._processed_step_event_order: List[str] = []
        
        # Ensure memory directory exists
        os.makedirs(MEMORY_DIR, exist_ok=True)

    async def connect(self):
        await super().connect()
        # Initialize MCP connections
        await self.mcp_manager.connect_all()
        
        # Phase 3: Listen for single-step requests
        # We listen on events.actor.agent-implementer (our name)
        # and global events.step.requested
        self.on(f"events.actor.{self.name}", self.handle_action_requested)
        self.on("events.step.requested", self.handle_step_requested)
        
        await self.listen(f"events.actor.{self.name}")
        await self.listen("events.step.requested")
        
        # Recovery: Resume any interrupted tasks from last restart
        await self._recover_interrupted_tasks()
        
        logger.info(f"{self.name} listening for durable steps with {len(self.mcp_manager.tools)} MCP tools loaded.")

    async def handle_action_requested(self, event: Event):
        """Initial task entry point. Initializes state and triggers first step."""
        try:
            payload = validate_event_payload(EventType.ACTION_REQUESTED, event.payload)
        except ValidationError as e:
            logger.error(f"BASE_AGENT: Invalid action.requested payload: {e}")
            await self._fail_task(event.correlation_id, "Invalid action.requested payload")
            return

        task_id = payload["task_id"]
        instruction = payload["instruction"]
        
        logger.info(f"BASE_AGENT: Received action request for {task_id}")

        # Auto-retrieve relevant memories and prepend to instruction
        memories = await self._recall_memories(instruction)
        if memories:
            instruction = (
                f"RELEVANT MEMORIES FROM PAST TASKS:\n{memories}\n\n"
                f"CURRENT TASK:\n{instruction}"
            )
            logger.info(f"BASE_AGENT: Injected {len(memories.splitlines())} memory lines into context")
        
        # Initialize actor state in KV
        state = {
            "task_id": task_id,
            "instruction": instruction,
            "history": [],
            "iteration": 0,
            "status": "running",
            "created_at": int(time.time())
        }
        await self.state_store.put(f"actor_state.{task_id}", state)
        
        # Trigger first step
        step_evt = Event(
            type=EventType.STEP_REQUESTED,
            source_actor=self.name,
            correlation_id=task_id,
            payload={"task_id": task_id}
        )
        logger.info(f"BASE_AGENT: Publishing first STEP_REQUESTED for {task_id}")
        await self.publish("events.step.requested", step_evt)

    async def handle_step_requested(self, event: Event):
        """Durable execution step. Loads state, runs ONE iteration, saves state, and sleeps."""
        try:
            payload = validate_event_payload(EventType.STEP_REQUESTED, event.payload)
        except ValidationError as e:
            logger.error(f"BASE_AGENT: Invalid step.requested payload: {e}")
            return

        task_id = payload["task_id"]

        # Idempotency guard: ignore duplicate event ids (at-least-once delivery safe)
        if event.id in self._processed_step_event_ids:
            logger.info(f"BASE_AGENT: Ignoring duplicate step event id {event.id} for {task_id}")
            return
        self._processed_step_event_ids.add(event.id)
        self._processed_step_event_order.append(event.id)
        if len(self._processed_step_event_order) > 300:
            stale = self._processed_step_event_order.pop(0)
            self._processed_step_event_ids.discard(stale)

        # In-flight guard: if a step is already executing for this task, skip re-entry.
        if task_id in self._inflight_tasks:
            logger.info(f"BASE_AGENT: Step for {task_id} already in-flight, skipping duplicate request")
            return
        self._inflight_tasks.add(task_id)

        logger.info(f"BASE_AGENT: Handling step requested for {task_id}")
        try:
            state = await self.state_store.get(f"actor_state.{task_id}")
            if not state:
                logger.warning(f"BASE_AGENT: No state found for {task_id}")
                return

            if state.get("status") != "running":
                logger.info(f"BASE_AGENT: Task {task_id} is not running (status={state.get('status')})")
                return

            instruction = state["instruction"]
            history = state["history"]
            iteration = state["iteration"] + 1
            self._current_task_id = task_id

            logger.info(f"BASE_AGENT: Starting Durable Step: {task_id} iteration {iteration}")

            # 1. Observation Masking
            masked_history = self._mask_observations(history, keep_last=10)

            # Stall detection
            if len(history) >= 4:
                t = [h.get("action", {}).get("tool", "") for h in history]
                a = [json.dumps(h.get("action", {}), sort_keys=True) for h in history]

                def _is_cycle(seq, length):
                    if len(seq) < length * 2:
                        return False
                    return seq[-length:] == seq[-length * 2 : -length]

                cycle_len = next((k for k in (1, 2, 3) if _is_cycle(t, k)), None)
                if cycle_len is None and len(a) >= 3 and len(set(a[-3:])) == 1:
                    cycle_len = 1

                if cycle_len is not None:
                    stall_tool = t[-1]
                    logger.warning(f"STALL: Task {task_id} cycle-{cycle_len} detected on '{stall_tool}'")

                    repeat_count = 0
                    last_action = a[-1]
                    for x in reversed(a):
                        if x == last_action:
                            repeat_count += 1
                        else:
                            break

                    recent_results = " ".join(str(h.get("result", "")) for h in history[-6:])
                    gap_confirmed = "CAPABILITY_GAP" in recent_results or "not found" in recent_results.lower()

                    if repeat_count >= 4:
                        reason = (
                            f"Task auto-terminated after {repeat_count} identical '{stall_tool}' calls with no progress."
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
                        await self.publish("events.action.failed", Event(
                            type=EventType.ACTION_FAILED,
                            source_actor=self.name,
                            correlation_id=task_id,
                            payload={"task_id": task_id, "status": "failed", "reason": reason},
                        ))
                        return

                    if gap_confirmed:
                        instruction = instruction + (
                            f"\n\nSYSTEM WARNING: You are stuck in a loop on '{stall_tool}' because a required "
                            "binary or tool is missing. DO NOT retry the same command. "
                            "You MUST resolve the missing dependency RIGHT NOW using one of these approaches:\n"
                            "1. shell('apt-get install -y <package> 2>&1') to install it via apt\n"
                            "2. shell('pip install <package> 2>&1') if it is a Python package\n"
                            "3. python_sandbox(code) to implement equivalent functionality in pure Python — "
                            "do NOT call the missing binary, write Python code that does the same thing."
                        )
                    else:
                        instruction = instruction + (
                            f"\n\nSYSTEM WARNING (repeat_count={repeat_count}): You have called '{stall_tool}' "
                            "with the SAME arguments and result repeatedly. Do NOT call it again; continue to "
                            "the next pending sub-task, or call done() if complete."
                        )

            # Efficiency injection
            if len(history) >= 2:
                recent_tools = [h.get("action", {}).get("tool") for h in history[-4:]]
                sequential_same = sum(1 for t in recent_tools if t == recent_tools[-1])
                last_tool = recent_tools[-1]
                if sequential_same >= 2 and last_tool in ("fetch_url", "read_file", "shell"):
                    logger.warning(f"EFFICIENCY: Task {task_id} called '{last_tool}' {sequential_same}x sequentially")
                    instruction = instruction + (
                        f"\n\nSYSTEM EFFICIENCY ALERT: You have called '{last_tool}' {sequential_same} "
                        "times in a row. Switch strategy now: use python_sandbox or parallel_tools."
                    )

            # 2. ReAct generation with hard timeout
            try:
                action_payload, _think_text = await asyncio.wait_for(
                    self._generate_single_action(task_id, instruction, masked_history, iteration),
                    timeout=90.0,
                )
            except asyncio.TimeoutError:
                logger.error(f"BASE_AGENT: LLM timed out after 90s for {task_id} iteration {iteration}")
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

            # 3. Execute
            await self._ui_emit(EventType.AGENT_TOOL_START, task_id, {"tool": tool_name, "args": tool_args})
            result = await self.execute_tool(tool_name, tool_args, task_id=task_id)
            await self._ui_emit(EventType.AGENT_TOOL_END, task_id, {"tool": tool_name, "result": str(result)[:500]})

            # 4. Persist
            capped_result = self._summarise_result(result)
            history.append({"action": action_payload, "result": capped_result})
            state["history"] = history
            state["iteration"] = iteration

            if tool_name == "done":
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
                await self.publish("events.action.completed", Event(
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
                logger.info(f"BASE_AGENT: Task {task_id} suspended — waiting for human response.")
                return

            if iteration % 10 == 0:
                history = await self._fold_context(task_id, instruction, history)
                state["history"] = history

            await self.state_store.put(f"actor_state.{task_id}", state)

            next_step = Event(
                type=EventType.STEP_REQUESTED,
                source_actor=self.name,
                correlation_id=task_id,
                payload={"task_id": task_id},
            )
            logger.info(f"BASE_AGENT: Requesting NEXT step for {task_id}")
            await self.publish("events.step.requested", next_step)
        finally:
            self._inflight_tasks.discard(task_id)

    async def _recover_interrupted_tasks(self):
        """On startup, resume any tasks left in 'running' state from the last session."""
        try:
            # Query all actor state keys
            kv = self.state_store.kv
            if not kv:
                logger.info("BASE_AGENT: No KV store available for recovery")
                return
            
            # Scan for task states
            recovered = []
            keys = await kv.keys()
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
                            
                            # Only resume tasks that were actively running
                            if status == "running" and task_id:
                                await self.publish("events.step.requested", Event(
                                    type=EventType.STEP_REQUESTED,
                                    source_actor=self.name,
                                    correlation_id=task_id,
                                    payload={"task_id": task_id}
                                ))
                                recovered.append(task_id)
                                logger.info(f"BASE_AGENT: Recovered task {task_id} from persistent state")
                    except Exception as e:
                        logger.warning(f"BASE_AGENT: Error recovering from key {key}: {e}")
                        continue
            
            if recovered:
                logger.info(f"BASE_AGENT: Recovery complete — resumed {len(recovered)} task(s)")
            else:
                logger.info("BASE_AGENT: Recovery scan complete — no running tasks to resume")
        except Exception as e:
            logger.warning(f"BASE_AGENT: Error during task recovery: {e}")
            # Don't fail startup if recovery fails

    async def _generate_single_action(self, task_id: str, instruction: str, history: list, iteration: int):
        llm = self.llm
        
        mcp_tools = self.mcp_manager.get_tool_definitions()
        mcp_list = ", ".join([t["name"] for t in mcp_tools]) if mcp_tools else "none"

        system_prompt = f"""You are WzrdMerlin, an autonomous agent. Think step by step, then output ONE JSON tool call.

TOOLS:
- shell(cmd)
- read_file(path)
- write_file(path, content)
- search_memory(query) — search episodic memory for relevant past tasks, learned patterns, or facts.
  Use this at the START of complex tasks to check if you've done something similar before.
- write_memory(content, tags) — persist important facts, API patterns, or reusable solutions.
  Use this when you discover something non-obvious (e.g. the correct GitHub API endpoint format).
- fetch_url(url)
- request_human(question)
- done(summary)
- forage_search(query)
- forage_install(name, config)

- parallel_tools(calls) — fire N tool calls and get all results back in one step.
  ORDERING GUARANTEE: any shell install commands (pip install, apt install, etc.) in
  the calls list are automatically run FIRST and SEQUENTIALLY before the rest execute
  in parallel. This means you CAN safely put an install and a use of that binary in the
  SAME parallel_tools call — the install will always complete before the use runs.
  Use this any time you need to fetch, read, or process more than ONE item of the same kind.
  calls = list of {{"tool": "...", "args": {{...}}}} objects.
  Example — install a tool then immediately use it alongside other fetches:
  {{"tool": "parallel_tools", "args": {{"calls": [
    {{"tool": "shell", "args": {{"cmd": "pip install gron 2>&1"}}}},
    {{"tool": "shell", "args": {{"cmd": "gron /workspace/data.json"}}}},
    {{"tool": "fetch_url", "args": {{"url": "https://api.example.com/data"}}}}
  ]}}}}

- python_sandbox(code) — execute a Python script and get its output.
  Use this for ANY bulk data operation: fetching N items, parsing, transforming, formatting.
  The sandbox has httpx, requests, json, re, csv, datetime available.
  Example — fetch 15 GitHub PRs in ONE step:
  {{"tool": "python_sandbox", "args": {{"code": "import httpx\\nresp = httpx.get('https://api.github.com/repos/owner/repo/pulls?state=closed&per_page=15')\\nprint(resp.text)"}}}}

MCP TOOLS: {mcp_list}

RULES:
1. Reply with ONLY a valid JSON object — no prose, no markdown fences.
2. Format: {{"tool": "name", "args": {{...}}}}
3. BATCHING MANDATE: If a task involves fetching/reading/processing MORE THAN ONE item, you MUST use parallel_tools or python_sandbox to handle ALL of them in a single step. Never call fetch_url or read_file in a loop one-at-a-time.
4. PLAN BEFORE ACTING: For multi-step tasks, think about the most efficient path. Prefer one python_sandbox call that does everything over many sequential fetch_url calls.
5. MISSING TOOLS: If a shell command returns 'not found', that binary is not installed. You MUST immediately try one of these in order:
   a. Install it: shell('apt-get install -y <package> 2>&1')
   b. Install via pip: shell('pip install <package> 2>&1')
   c. Implement it in Python: python_sandbox(code that does the same thing without the binary)
   Never give up on a sub-task just because a binary is missing. Always resolve it.
6. COMPLETENESS: Only call done() when every sub-task has been completed or genuinely attempted with all alternatives exhausted. Include actual results.
7. If [Output Omitted] appears in history, trust your earlier reasoning for that step.
8. NEVER use request_human to greet, introduce yourself, or ask what the user wants. Start executing immediately. Only use request_human for information that is truly impossible to infer (e.g. a secret key, a login password).
9. PROGRESS: After each tool result, check your task list. Move to the next pending sub-task immediately — do not re-execute completed sub-tasks.
"""
        await self._ui_emit(EventType.AGENT_THINKING, task_id, {
            "iteration": iteration,
            "status": "generating",
            "text": f"[iteration {iteration}] Deciding next action…",
        })

        content_buf = []
        async for chunk_type, text in llm.generate_action_streaming(system_prompt, history, instruction):
            if chunk_type == "think":
                await self._ui_emit(EventType.AGENT_THINKING, task_id, {"text": text})
            else:
                content_buf.append(text)
                await self._ui_emit(EventType.AGENT_STREAMING, task_id, {"text": text})

        full_response = "".join(content_buf)
        action_payload = llm.parse_action(full_response)
        
        if not action_payload:
            logger.warning(f"BASE_AGENT: Failed to parse action. Raw LLM response: {full_response!r}")
            
        return action_payload, ""

    async def _recall_memories(self, query: str, top_k: int = 3) -> str:
        """Search episodic memory and return a formatted block of relevant past results."""
        memory_path = Path(MEMORY_DIR)
        if not memory_path.exists():
            return ""
        query_lower = query.lower()
        # Score each memory file by keyword overlap
        scored = []
        for fp in memory_path.glob("*.json"):
            try:
                data = json.loads(fp.read_text())
                content = data.get("content", "")
                tags = " ".join(data.get("tags", []))
                combined = (content + " " + tags).lower()
                # Count overlapping words
                words = set(w for w in query_lower.split() if len(w) > 3)
                score = sum(1 for w in words if w in combined)
                if score > 0:
                    scored.append((score, content))
            except Exception:
                continue
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [c for _, c in scored[:top_k]]
        return "\n---\n".join(top) if top else ""

    async def _save_memory(self, content: str, tags: list):
        """Persist a memory entry to the episodic memory store."""
        mid = str(int(time.time()))
        try:
            os.makedirs(MEMORY_DIR, exist_ok=True)
            with open(os.path.join(MEMORY_DIR, f"{mid}.json"), "w") as f:
                json.dump({"content": content, "tags": tags, "timestamp": mid}, f)
        except Exception as e:
            logger.warning(f"Failed to save memory: {e}")

    @staticmethod
    def _summarise_result(result: Any) -> str:
        """
        Convert a tool result into a history-safe string that tells the agent
        EXACTLY what it got, without silently truncating mid-JSON.

        - JSON arrays  → prepend "[Got N items]" so the agent knows how many
        - JSON objects  → kept verbatim up to 8000 chars
        - Plain strings → kept up to 8000 chars
        """
        raw = str(result)
        # Try to detect and annotate JSON arrays
        stripped = raw.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    count = len(parsed)
                    # Serialize compactly and cap
                    compact = json.dumps(parsed, separators=(",", ":"))
                    capped = compact[:8000]
                    if len(compact) > 8000:
                        capped += "... [truncated]"
                    return f"[Got {count} items] {capped}"
            except (json.JSONDecodeError, ValueError):
                pass
        # For dicts and plain text, just cap at 8000
        return raw[:8000]

    def _mask_observations(self, history: list, keep_last: int = 5) -> list:
        """Observation Masking: Replaces older verbose tool outputs with placeholders."""
        if len(history) <= keep_last:
            return history
        
        masked = []
        mask_threshold = len(history) - keep_last
        
        for i, turn in enumerate(history):
            if i < mask_threshold:
                new_turn = turn.copy()
                new_turn["result"] = "[Output Omitted for brevity. Refer to earlier reasoning if needed.]"
                masked.append(new_turn)
            else:
                masked.append(turn)
        return masked

    async def _fold_context(self, task_id: str, instruction: str, history: list) -> list:
        """Context Folding: Summarise completed phase into a dense artifact,
        evict transient tool outputs, and persist a journal entry."""
        logger.info(f"Folding context for {task_id} at iteration {len(history)}")

        # Build a compact digest of everything that happened in this phase
        phase_lines = []
        for i, turn in enumerate(history):
            tool = turn.get("action", {}).get("tool", "?")
            result_preview = str(turn.get("result", ""))[:200]
            phase_lines.append(f"Step {i+1}: {tool} → {result_preview}")
        phase_text = "\n".join(phase_lines)

        # Use the LLM to produce a compressed summary (~200 tokens)
        fold_prompt = (
            "You are a context compressor. Summarise the following agent execution trace "
            "into a dense, factual paragraph (max 200 words). Preserve key results, file "
            "paths, URLs, error messages, and decisions. Drop all raw JSON payloads.\n\n"
            f"TASK: {instruction[:500]}\n\n"
            f"TRACE:\n{phase_text[:3000]}\n\n"
            "SUMMARY:"
        )
        try:
            summary = await self.llm.generate_text(fold_prompt, max_tokens=300)
        except Exception as e:
            logger.warning(f"Context fold LLM call failed: {e}")
            summary = phase_text[:500]

        # Journal: persist the fold to disk so it survives across restarts
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

        # Replace history with the fold artifact + keep last 3 turns verbatim
        fold_entry = {
            "action": {"tool": "_context_fold", "args": {}},
            "result": f"[Phase Summary] {summary}",
        }
        kept = history[-3:] if len(history) > 3 else history
        return [fold_entry] + kept

    async def resume_with_human_response(self, task_id: str, response: str) -> bool:
        """Resume a suspended task with the human's response."""
        state = await self.state_store.get(f"actor_state.{task_id}")
        if not state or state.get("status") != "waiting_for_human":
            logger.warning(f"Cannot resume {task_id}: not in waiting_for_human state")
            return False

        if state["history"]:
            state["history"][-1]["result"] = f"Human responded: {response}"

        state["status"] = "running"
        await self.state_store.put(f"actor_state.{task_id}", state)

        step_evt = Event(
            type=EventType.STEP_REQUESTED,
            source_actor=self.name,
            correlation_id=task_id,
            payload={"task_id": task_id}
        )
        logger.info(f"BASE_AGENT: Resuming task {task_id} with human response")
        await self.publish("events.step.requested", step_evt)
        return True

    async def _fail_task(self, task_id: str, reason: str):
        # Publish on the failed channel with the failed event type.
        await self.publish("events.action.failed", Event(
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

    async def execute_tool(self, name: str, args: Dict[str, Any], task_id: str = "") -> str:
        if name in self.tools:
            try:
                res = await self.tools[name](args)
                result_str = str(res)

                # Detect "command not found" from shell as a capability gap.
                # This catches cases where the tool *exists* but the underlying
                # CLI binary isn't installed, so the existing gap-detection
                # (which only fires for unknown tool names) never triggers.
                if name in ("shell", "shell_exec"):
                    cmd = args.get("cmd", args.get("command", "")).strip()
                    binary = cmd.split()[0] if cmd else ""
                    # Use binary-specific patterns to avoid false positives from programs
                    # that print "No such file or directory" about their *arguments* (e.g.
                    # `gron missing_file.txt` succeeds as a binary but fails on the arg).
                    # Only shell-level "binary not found" errors include the binary name.
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
                        await self.publish("events.capability.gap", gap_event)
                        await self._ui_emit(EventType.CAPABILITY_GAP, task_id, {
                            "gap_description": gap_desc,
                            "tool_name": binary,
                        })
                        return (
                            f"CAPABILITY_GAP: '{binary}' is not installed. "
                            f"Original error: {result_str[:300]}. "
                            f"Try a different approach to accomplish the same goal."
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

        # Tool not found — emit capability gap so Alita can try to synthesize it
        logger.warning(f"CAPABILITY_GAP: Tool '{name}' not found. Emitting gap event.")
        gap_event = Event(
            type=EventType.CAPABILITY_GAP,
            source_actor=self.name,
            correlation_id=task_id,
            payload={
                "gap_description": f"Missing tool: {name}",
                "tool_name": name,
                "tool_args": args,
                "triggering_task": task_id,
            }
        )
        await self.publish("events.capability.gap", gap_event)
        # Also broadcast to the UI so the dashboard shows it
        await self._ui_emit(EventType.CAPABILITY_GAP, task_id, {
            "gap_description": f"Missing tool: {name}",
            "tool_name": name,
        })

        return f"Error: Tool '{name}' not found. Available tools: {', '.join(list(self.tools.keys()))}. Try 'forage_search' to find it."

    # --- Tool Implementations (Shell, File, Memory, Forage, Sandbox) ---

    async def tool_shell(self, args: Dict[str, Any]) -> str:
        cmd = args.get("cmd", "")
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE),
                timeout=120
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            output = ((stdout or b"") + (stderr or b"")).decode(errors="replace").strip()
            return output if output else "(no output)"
        except Exception as e:
            return f"Error: {str(e)}"

    async def tool_read_file(self, args: Dict[str, Any]) -> str:
        path = args.get("path", "")
        try:
            with open(path, "r") as f:
                return f.read()[:8000]
        except Exception as e:
            return f"Error: {str(e)}"

    async def tool_write_file(self, args: Dict[str, Any]) -> str:
        path, content = args.get("path", ""), args.get("content", "")
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return f"Wrote {len(content)} chars to {path}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def tool_search_memory(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "").lower()
        result = await self._recall_memories(query, top_k=5)
        return result if result else "No matching memories found."

    async def tool_write_memory(self, args: Dict[str, Any]) -> str:
        content, tags = args.get("content", ""), args.get("tags", [])
        await self._save_memory(content, tags)
        return "Memory saved."

    async def tool_fetch_url(self, args: Dict[str, Any]) -> str:
        url = args.get("url", "")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url)
                return resp.text[:12000]
        except Exception as e:
            return f"Error: {str(e)}"

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

    # Matches any shell command that installs packages/binaries and mutates the
    # environment.  These are "producers" — they must fully complete before any
    # "consumer" call that relies on the newly-installed binary/package runs.
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
        """True if this call is a shell command that mutates the environment."""
        if call.get("tool") != "shell":
            return False
        cmd = call.get("args", {}).get("cmd", "")
        return bool(cls._INSTALL_RE.search(cmd))

    async def tool_parallel_tools(self, args: Dict[str, Any]) -> str:
        """Execute multiple tool calls concurrently, with phased ordering.

        Phase 1 — Mutating (install) commands run SEQUENTIALLY in declaration
                   order.  This ensures a `pip install foo` always completes
                   before any sibling call that shells out to `foo`.
        Phase 2 — All remaining (read / fetch / query) calls run in parallel
                   via asyncio.gather once Phase 1 is done.

        This is the read/write tier split used by production coding agents to
        prevent producer-consumer races without sacrificing read parallelism.
        """
        calls = args.get("calls", [])
        if not calls:
            return "Error: parallel_tools requires a non-empty 'calls' list."

        task_id = getattr(self, "_current_task_id", "")

        async def _run_one(call: Dict[str, Any]) -> Dict[str, Any]:
            name = call.get("tool", "")
            sub_args = call.get("args", {})
            if name == "parallel_tools":
                result = "Error: parallel_tools cannot be nested."
            else:
                result = await self.execute_tool(name, sub_args, task_id=task_id)
            return {"tool": name, "args": sub_args, "result": str(result)[:2000]}

        # Partition into install (mutating) vs everything else, preserving
        # original indices so the result list stays in declaration order.
        install_indices = [i for i, c in enumerate(calls) if self._is_install_call(c)]
        other_indices   = [i for i, c in enumerate(calls) if not self._is_install_call(c)]

        ordered_results: list = [None] * len(calls)

        # Phase 1: installs run one-by-one so each binary/package is available
        # before the next install (or any consumer call) runs.
        if install_indices:
            logger.info(
                f"parallel_tools: running {len(install_indices)} install call(s) "
                f"sequentially before {len(other_indices)} parallel call(s)"
            )
            for i in install_indices:
                ordered_results[i] = await _run_one(calls[i])

        # Phase 2: remaining calls in parallel — installs are fully settled.
        if other_indices:
            parallel_results = await asyncio.gather(
                *[_run_one(calls[i]) for i in other_indices]
            )
            for i, result in zip(other_indices, parallel_results):
                ordered_results[i] = result

        return json.dumps(ordered_results, indent=2)
