FROM python:3.11-slim-bookworm

# Set work directory
WORKDIR /app

# Set env variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Default NATS URL for docker-compose environment
ENV NATS_URL="nats://nats:4222"
# Default host internal URL for Ollama running on the host machine
ENV OLLAMA_BASE_URL="http://host.docker.internal:11434"
# Persistent workspace for memory, packages, rollback logs
ENV MERLIN_WORKSPACE="/workspace"

# Install system dependencies (git is needed for Alita pattern worktree self-improvement)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Configure git for self-improvement commits
RUN git config --global user.email "merlin@wzrd.local" \
    && git config --global user.name "WzrdMerlin"

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "chromadb>=0.5.0"

# Copy the core project
COPY src/ src/

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create workspace directory (will be overlaid by Docker volume)
RUN mkdir -p /workspace/memory /workspace/candidates /workspace/.merlin/chroma

# Use entrypoint that restores agent packages before starting uvicorn
CMD ["/entrypoint.sh"]
