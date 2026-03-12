import asyncio
import logging
import os
import subprocess
import uuid
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CodeModeSandbox:
    """
    Provides a localized 'Code Mode' sandbox to execute agent-generated scripts.
    Drastically reduces token transmission by offloading heavy data processing
    (CSV, JSON, deep DOM traversing) from the LLM context to a local runner.
    """
    def __init__(self, sandbox_dir: str = None):
        self.sandbox_dir = sandbox_dir or os.path.join(".merlin", "sandbox")
        os.makedirs(self.sandbox_dir, exist_ok=True)

    async def execute_python(self, code: str, timeout: int = 60) -> Dict[str, Any]:
        """Execute a Python script in the sandbox."""
        script_id = uuid.uuid4().hex[:8]
        script_path = os.path.join(self.sandbox_dir, f"run_{script_id}.py")
        
        with open(script_path, "w") as f:
            f.write(code)
            
        logger.info(f"Executing Code Mode Python script: {script_path}")
        
        try:
            proc = await asyncio.create_subprocess_exec(
                "python", script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                return {
                    "status": "success" if proc.returncode == 0 else "error",
                    "stdout": stdout.decode(errors="replace").strip(),
                    "stderr": stderr.decode(errors="replace").strip(),
                    "returncode": proc.returncode
                }
            except asyncio.TimeoutError:
                proc.kill()
                return {
                    "status": "error",
                    "reason": f"Execution timed out after {timeout}s",
                    "stdout": "",
                    "stderr": ""
                }
        except Exception as e:
            return {
                "status": "error",
                "reason": f"Failed to launch process: {str(e)}",
                "stdout": "",
                "stderr": ""
            }
        finally:
            # Optionally keep script for audit trail
            pass

    async def execute_javascript(self, code: str, timeout: int = 60) -> Dict[str, Any]:
        """Execute a JavaScript (Node.js) script in the sandbox."""
        script_id = uuid.uuid4().hex[:8]
        script_path = os.path.join(self.sandbox_dir, f"run_{script_id}.js")
        
        with open(script_path, "w") as f:
            f.write(code)
            
        logger.info(f"Executing Code Mode JS script: {script_path}")
        
        try:
            proc = await asyncio.create_subprocess_exec(
                "node", script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                return {
                    "status": "success" if proc.returncode == 0 else "error",
                    "stdout": stdout.decode(errors="replace").strip(),
                    "stderr": stderr.decode(errors="replace").strip(),
                    "returncode": proc.returncode
                }
            except asyncio.TimeoutError:
                proc.kill()
                return {
                    "status": "error",
                    "reason": f"Execution timed out after {timeout}s",
                    "stdout": "",
                    "stderr": ""
                }
        except Exception as e:
            return {
                "status": "error",
                "reason": f"Failed to launch process: {str(e)}",
                "stdout": "",
                "stderr": ""
            }
