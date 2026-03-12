#!/bin/bash
set -e

WORKSPACE="${MERLIN_WORKSPACE:-/workspace}"
AGENT_REQS="$WORKSPACE/requirements-agent.txt"

# Create workspace subdirectories
mkdir -p "$WORKSPACE/memory"
mkdir -p "$WORKSPACE/candidates"

# Restore agent-installed packages from previous sessions
if [ -f "$AGENT_REQS" ]; then
    echo "[entrypoint] Restoring agent-installed packages from $AGENT_REQS"
    pip install --quiet --no-cache-dir -r "$AGENT_REQS" 2>&1 | tail -5 || true
    echo "[entrypoint] Package restoration complete."
else
    echo "[entrypoint] No agent-installed packages to restore."
fi

# Initialize git repo if not present (needed for self-improvement worktrees)
if [ ! -d ".git" ]; then
    echo "[entrypoint] Initializing git repo for self-improvement system..."
    git config --global user.email "merlin@wzrd.local"
    git config --global user.name "WzrdMerlin"
    git init
    git add .
    git commit -m "Initial commit" --allow-empty 2>/dev/null || true
fi

echo "[entrypoint] Starting WzrdMerlin core with hot reload enabled..."
exec uvicorn src.core.main:app --host 0.0.0.0 --port 8000 --reload
