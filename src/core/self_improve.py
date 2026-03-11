import os
import subprocess
import shutil
import logging
import uuid
import json
import time
from typing import Dict, Any, Tuple, Optional
from src.core.actor import BaseActor
from src.core.events import Event, EventType

logger = logging.getLogger(__name__)

WORKSPACE = os.getenv("MERLIN_WORKSPACE", "/workspace")
ROLLBACK_LOG = os.path.join(WORKSPACE, "rollback.json")


class ImprovementManager(BaseActor):
    """
    Listens for CAPABILITY_GAP events. When an agent flags a missing capability,
    this component:
      1. Creates an isolated git worktree
      2. Uses the LLM to generate a real code patch
      3. Validates with compileall + pytest
      4. Records a rollback point before merging
      5. Promotes on success, records failure on rejection

    All state persists in /workspace/ (Docker volume).
    """
    def __init__(self, nats_url: str = None, repo_workspace: str = None):
        super().__init__(name="improvement-manager", nats_url=nats_url)
        self.repo_workspace = repo_workspace or WORKSPACE

    async def connect(self):
        await super().connect()
        self.on("events.system.improvement", self.handle_improvement_queued)
        await self.listen("events.system.improvement")
        self.on("events.capability.gap", self.handle_improvement_queued)
        await self.listen("events.capability.gap")
        logger.info(f"{self.name} listening for capability gaps...")

    async def handle_improvement_queued(self, event: Event):
        gap_desc = event.payload.get("gap_description")
        task_src = event.payload.get("triggering_task")

        logger.info(f"Processing improvement for gap: {gap_desc} (task: {task_src})")

        # Check if we've already failed this exact gap recently (prevent loops)
        if self._is_recently_failed(gap_desc):
            logger.warning(f"Skipping gap '{gap_desc}' — failed recently, avoiding loop.")
            return

        # 1. Setup isolated candidate worktree
        candidate_dir, branch_name = self._setup_worktree()
        if not candidate_dir:
            logger.error("Failed to setup improvement worktree.")
            return

        try:
            # 2. LLM-driven patch generation
            patch_ok = await self._apply_patch(candidate_dir, gap_desc, task_src)

            if not patch_ok:
                logger.error("LLM patching failed.")
                self._record_failure(gap_desc, "patch_generation_failed")
                return

            # 3. Validation pipeline
            is_valid, validation_log = self._validate_candidate(candidate_dir)

            if is_valid:
                # 4. Record rollback point BEFORE merging
                rollback_hash = self._record_rollback_point(branch_name)
                logger.info(f"Rollback point saved: {rollback_hash}")

                # 5. Promote
                self._promote_candidate(candidate_dir, branch_name)

                # Emit success event
                deployed_evt = Event(
                    type=EventType.IMPROVEMENT_DEPLOYED,
                    source_actor=self.name,
                    correlation_id=event.correlation_id,
                    payload={
                        "gap_description": gap_desc,
                        "status": "deployed",
                        "branch": branch_name,
                        "rollback_hash": rollback_hash,
                    },
                )
                await self.publish("events.system.improvement", deployed_evt)
                logger.info(f"Improvement deployed for: {gap_desc}")
            else:
                logger.warning(f"Candidate failed validation:\n{validation_log}")
                self._record_failure(gap_desc, validation_log)
        finally:
            # Always clean up the worktree
            self._cleanup_worktree(candidate_dir, branch_name)

    def _setup_worktree(self) -> Tuple[Optional[str], Optional[str]]:
        branch_name = f"improve-{uuid.uuid4().hex[:8]}"
        worktree_path = os.path.join(self.repo_workspace, "candidates", branch_name)

        # Ensure base repo is a git repo
        if not os.path.exists(".git"):
            subprocess.run(["git", "init"], check=True, capture_output=True)
            subprocess.run(["git", "add", "."], check=True, capture_output=True)
            try:
                subprocess.run(
                    ["git", "commit", "-m", "Initial commit"],
                    check=True, capture_output=True,
                )
            except subprocess.CalledProcessError:
                pass

        # Determine current branch name for worktree base
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True,
        )
        base_branch = result.stdout.strip() or "main"

        try:
            os.makedirs(os.path.dirname(worktree_path), exist_ok=True)
            subprocess.run(
                ["git", "worktree", "add", "-b", branch_name, worktree_path, base_branch],
                check=True, capture_output=True, text=True,
            )
            return worktree_path, branch_name
        except subprocess.CalledProcessError as e:
            logger.error(f"Worktree setup failed: {e.stderr}")
            return None, None

    async def _apply_patch(self, candidate_dir: str, gap_desc: str, task_src: str) -> bool:
        """
        Use the LLM to generate an actual code patch for the capability gap.
        The LLM sees the current codebase structure and writes new/modified files.
        """
        from src.core.llm import ModelInterface

        llm = ModelInterface()

        # Gather codebase context: list existing tools and agent structure
        try:
            agent_code = ""
            agent_path = os.path.join(candidate_dir, "src", "core", "base_agent.py")
            if os.path.exists(agent_path):
                with open(agent_path, "r") as f:
                    agent_code = f.read()
        except Exception:
            agent_code = "(could not read base_agent.py)"

        patch_prompt = f"""You are a code-generation assistant for WzrdMerlin, an autonomous agent OS.

A capability gap was detected: "{gap_desc}"
Triggering task: "{task_src}"

The agent's tool registry is in base_agent.py. Here is the current code:

```python
{agent_code[:8000]}
```

Generate a patch to address this capability gap. Your response MUST be a JSON object:
{{
  "files": [
    {{
      "path": "relative/path/to/file.py",
      "content": "full file content as a string"
    }}
  ],
  "test_file": {{
    "path": "tests/test_new_capability.py",
    "content": "pytest test code"
  }},
  "description": "what this patch does"
}}

Rules:
- Only modify or create files under src/core/ or tests/
- If adding a new tool, add it to the tools dict in BaseAgentActor.__init__
- Include a working pytest test
- Keep changes minimal and focused
"""

        try:
            action = await llm.generate_action(
                system_prompt="You are a code generation assistant. Output only valid JSON.",
                history=[],
                instruction=patch_prompt,
            )

            if not action or "files" not in action:
                logger.error(f"LLM patch response missing 'files': {action}")
                return False

            # Write each file to the candidate directory
            for file_entry in action.get("files", []):
                rel_path = file_entry.get("path", "")
                content = file_entry.get("content", "")
                if not rel_path or not content:
                    continue

                full_path = os.path.join(candidate_dir, rel_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w") as f:
                    f.write(content)
                logger.info(f"Patch wrote: {rel_path} ({len(content)} chars)")

            # Write test file
            test_entry = action.get("test_file", {})
            if test_entry.get("path") and test_entry.get("content"):
                test_path = os.path.join(candidate_dir, test_entry["path"])
                os.makedirs(os.path.dirname(test_path), exist_ok=True)
                with open(test_path, "w") as f:
                    f.write(test_entry["content"])
                logger.info(f"Patch wrote test: {test_entry['path']}")

            # Git add + commit in the worktree
            subprocess.run(
                ["git", "add", "."],
                cwd=candidate_dir, check=True, capture_output=True,
            )
            desc = action.get("description", gap_desc)[:72]
            subprocess.run(
                ["git", "commit", "-m", f"[self-improve] {desc}"],
                cwd=candidate_dir, check=True, capture_output=True,
            )

            return True

        except Exception as e:
            logger.error(f"LLM patch generation failed: {e}")
            return False

    def _validate_candidate(self, candidate_dir: str) -> Tuple[bool, str]:
        logs = []

        # Step 1: compileall — catches syntax errors
        compile_res = subprocess.run(
            ["python", "-m", "compileall", "-q", "src/core/"],
            cwd=candidate_dir,
            capture_output=True,
            text=True,
        )
        if compile_res.returncode != 0:
            return False, f"Compile Error:\n{compile_res.stderr}"
        logs.append("compileall: passed")

        # Step 2: pytest — run any tests in the candidate
        tests_dir = os.path.join(candidate_dir, "tests")
        if os.path.isdir(tests_dir):
            pytest_res = subprocess.run(
                ["python", "-m", "pytest", tests_dir, "-x", "--tb=short", "-q"],
                cwd=candidate_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if pytest_res.returncode != 0:
                return False, f"Test Failure:\n{pytest_res.stdout}\n{pytest_res.stderr}"
            logs.append(f"pytest: passed ({pytest_res.stdout.strip()})")
        else:
            logs.append("pytest: skipped (no tests/ directory)")

        # Step 3: Import check — try importing the modified modules
        import_check = subprocess.run(
            ["python", "-c", "import src.core.base_agent; import src.core.events"],
            cwd=candidate_dir,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if import_check.returncode != 0:
            return False, f"Import Error:\n{import_check.stderr}"
        logs.append("import check: passed")

        return True, "\n".join(logs)

    def _record_rollback_point(self, branch_name: str) -> str:
        """Save the current HEAD hash so we can revert if the merge causes issues."""
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
        )
        head_hash = result.stdout.strip()

        # Append to rollback log (persisted in /workspace/)
        rollbacks = []
        if os.path.exists(ROLLBACK_LOG):
            try:
                with open(ROLLBACK_LOG, "r") as f:
                    rollbacks = json.load(f)
            except (json.JSONDecodeError, OSError):
                rollbacks = []

        rollbacks.append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "branch": branch_name,
            "pre_merge_hash": head_hash,
        })

        # Keep last 50 rollback entries
        rollbacks = rollbacks[-50:]

        os.makedirs(os.path.dirname(ROLLBACK_LOG), exist_ok=True)
        with open(ROLLBACK_LOG, "w") as f:
            json.dump(rollbacks, f, indent=2)

        return head_hash

    def _promote_candidate(self, candidate_dir: str, branch_name: str):
        """Merge the candidate branch back into the current branch."""
        try:
            # Commit any uncommitted changes in candidate first
            subprocess.run(
                ["git", "add", "."],
                cwd=candidate_dir, capture_output=True,
            )

            subprocess.run(
                ["git", "merge", "--no-ff", branch_name, "-m",
                 f"[self-improve] Merge {branch_name}"],
                check=True, capture_output=True, text=True,
            )
            logger.info(f"Successfully promoted {branch_name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Merge failed for {branch_name}: {e.stderr}")
            # Abort the failed merge
            subprocess.run(["git", "merge", "--abort"], capture_output=True)

    def _cleanup_worktree(self, candidate_dir: str, branch_name: str):
        """Remove worktree and branch."""
        if not candidate_dir or not branch_name:
            return
        try:
            subprocess.run(
                ["git", "worktree", "remove", "-f", candidate_dir],
                capture_output=True,
            )
            subprocess.run(
                ["git", "branch", "-D", branch_name],
                capture_output=True,
            )
        except Exception as e:
            logger.error(f"Cleanup failure: {e}")

    def _record_failure(self, gap_desc: str, reason: str):
        """Record a failed improvement attempt to prevent retry loops."""
        failure_log_path = os.path.join(self.repo_workspace, "improvement_failures.json")

        failures = []
        if os.path.exists(failure_log_path):
            try:
                with open(failure_log_path, "r") as f:
                    failures = json.load(f)
            except (json.JSONDecodeError, OSError):
                failures = []

        failures.append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gap_description": gap_desc,
            "reason": reason[:500],
        })

        # Keep last 100 failure entries
        failures = failures[-100:]

        os.makedirs(os.path.dirname(failure_log_path), exist_ok=True)
        with open(failure_log_path, "w") as f:
            json.dump(failures, f, indent=2)

        logger.info(f"Recorded improvement failure: {gap_desc[:80]}")

    def _is_recently_failed(self, gap_desc: str, cooldown_hours: int = 24) -> bool:
        """Check if this exact gap failed within the cooldown window."""
        failure_log_path = os.path.join(self.repo_workspace, "improvement_failures.json")

        if not os.path.exists(failure_log_path):
            return False

        try:
            with open(failure_log_path, "r") as f:
                failures = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False

        cutoff = time.time() - (cooldown_hours * 3600)

        for entry in reversed(failures):
            if entry.get("gap_description") == gap_desc:
                try:
                    ts = time.mktime(time.strptime(entry["timestamp"], "%Y-%m-%dT%H:%M:%S"))
                    if ts > cutoff:
                        return True
                except (ValueError, KeyError):
                    continue

        return False

    async def rollback_last(self) -> str:
        """Roll back the most recent self-improvement merge."""
        if not os.path.exists(ROLLBACK_LOG):
            return "No rollback history found."

        try:
            with open(ROLLBACK_LOG, "r") as f:
                rollbacks = json.load(f)
        except (json.JSONDecodeError, OSError):
            return "Could not read rollback log."

        if not rollbacks:
            return "Rollback log is empty."

        last = rollbacks[-1]
        target_hash = last.get("pre_merge_hash")
        branch = last.get("branch", "?")

        if not target_hash:
            return "No valid rollback hash found."

        try:
            subprocess.run(
                ["git", "reset", "--hard", target_hash],
                check=True, capture_output=True, text=True,
            )

            # Remove from rollback log
            rollbacks.pop()
            with open(ROLLBACK_LOG, "w") as f:
                json.dump(rollbacks, f, indent=2)

            return f"Rolled back merge of {branch} to {target_hash[:8]}"
        except subprocess.CalledProcessError as e:
            return f"Rollback failed: {e.stderr}"
