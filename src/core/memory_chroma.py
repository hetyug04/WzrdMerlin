"""
WzrdMerlin v3 — ChromaDB memory (optional upgrade path).

This is a verbatim copy of v2's memory.py (EpisodicMemory class).
Only loaded when `chromadb` is installed.
Use: pip install chromadb

The get_memory() factory in memory.py will automatically use this
if chromadb is importable.
"""
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

WORKSPACE = os.getenv("MERLIN_WORKSPACE", "/workspace")
CHROMA_DIR = os.path.join(WORKSPACE, ".merlin", "chroma")
LEGACY_MEMORY_DIR = os.path.join(WORKSPACE, "memory")

EMBED_MODEL = os.getenv("MERLIN_EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

EPISODIC_COLLECTION = "episodic_memory"
TRAJECTORY_COLLECTION = "trajectories"


class EpisodicMemory:
    """
    Persistent vector memory backed by ChromaDB (local SQLite mode).

    Two collections:
      - episodic_memory: general facts, task results, agent-written memories
      - trajectories: full task traces for In-Context Distillation
    """

    def __init__(self):
        self._client = None
        self._episodic = None
        self._trajectories = None
        self._embed_ok: Optional[bool] = None
        self._initialized = False

    def _ensure_init(self):
        if self._initialized:
            return
        self._initialized = True
        try:
            import chromadb
            from chromadb.config import Settings
            os.makedirs(CHROMA_DIR, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=CHROMA_DIR,
                settings=Settings(anonymized_telemetry=False),
            )
            self._episodic = self._client.get_or_create_collection(
                name=EPISODIC_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            self._trajectories = self._client.get_or_create_collection(
                name=TRAJECTORY_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"EpisodicMemory: ChromaDB initialized at {CHROMA_DIR} "
                f"(episodic={self._episodic.count()}, trajectories={self._trajectories.count()})"
            )
        except Exception as e:
            logger.error(f"EpisodicMemory: ChromaDB init failed: {e}")
            self._client = None

    async def _embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        if self._embed_ok is False:
            return None
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{OLLAMA_BASE}/api/embed",
                    json={"model": EMBED_MODEL, "input": texts},
                )
                if resp.status_code != 200:
                    logger.warning(
                        f"EpisodicMemory: Ollama embed returned {resp.status_code}. "
                        "Falling back to ChromaDB default embeddings."
                    )
                    self._embed_ok = False
                    return None
                data = resp.json()
                self._embed_ok = True
                return data.get("embeddings", data.get("embedding"))
        except Exception as e:
            if self._embed_ok is None:
                logger.warning(f"EpisodicMemory: Ollama embed unavailable ({e}). Using ChromaDB defaults.")
                self._embed_ok = False
            return None

    async def add(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        collection: str = "episodic",
    ) -> str:
        self._ensure_init()
        if not self._client:
            return self._legacy_add(content, tags)

        doc_id = hashlib.sha256(
            f"{content[:200]}:{time.time()}".encode()
        ).hexdigest()[:16]

        meta = {
            "timestamp": int(time.time()),
            "tags": json.dumps(tags or []),
        }
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v

        col = self._trajectories if collection == "trajectories" else self._episodic
        embeddings = await self._embed([content])
        try:
            if embeddings:
                col.add(ids=[doc_id], documents=[content], embeddings=embeddings, metadatas=[meta])
            else:
                col.add(ids=[doc_id], documents=[content], metadatas=[meta])
            logger.debug(f"EpisodicMemory: Added doc {doc_id} to {collection}")
        except Exception as e:
            logger.error(f"EpisodicMemory: Failed to add document: {e}")
            return self._legacy_add(content, tags)
        return doc_id

    async def search(
        self,
        query: str,
        top_k: int = 5,
        collection: str = "episodic",
        min_score: float = 0.3,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        self._ensure_init()
        if not self._client:
            return self._legacy_search(query, top_k)

        col = self._trajectories if collection == "trajectories" else self._episodic
        if col.count() == 0:
            return []

        embeddings = await self._embed([query])
        try:
            if embeddings:
                results = col.query(
                    query_embeddings=embeddings,
                    n_results=min(top_k, col.count()),
                    where=where,
                )
            else:
                results = col.query(
                    query_texts=[query],
                    n_results=min(top_k, col.count()),
                    where=where,
                )
        except Exception as e:
            logger.error(f"EpisodicMemory: Search failed: {e}")
            return self._legacy_search(query, top_k)

        docs = []
        if results and results.get("documents"):
            for i, doc in enumerate(results["documents"][0]):
                dist = results["distances"][0][i] if results.get("distances") else 0
                score = 1.0 - dist
                if score < min_score:
                    continue
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                docs.append({
                    "id": results["ids"][0][i],
                    "content": doc,
                    "score": round(score, 4),
                    "metadata": meta,
                })
        return docs

    async def recall(self, query: str, top_k: int = 3) -> str:
        results = await self.search(query, top_k=top_k, min_score=0.3)
        if not results:
            return ""
        lines = []
        for r in results:
            score_pct = int(r["score"] * 100)
            lines.append(f"[relevance {score_pct}%] {r['content']}")
        return "\n---\n".join(lines)

    async def prune(self, max_age_days: int = 90) -> int:
        self._ensure_init()
        if not self._client or not self._episodic:
            return 0
        cutoff = int(time.time()) - (max_age_days * 86400)
        try:
            old = self._episodic.get(where={"timestamp": {"$lt": cutoff}})
            if old and old["ids"]:
                self._episodic.delete(ids=old["ids"])
                logger.info(f"EpisodicMemory: Pruned {len(old['ids'])} entries")
                return len(old["ids"])
        except Exception as e:
            logger.error(f"EpisodicMemory: Prune failed: {e}")
        return 0

    def count(self, collection: str = "episodic") -> int:
        self._ensure_init()
        if not self._client:
            return 0
        col = self._trajectories if collection == "trajectories" else self._episodic
        return col.count() if col else 0

    async def reindex(self, collection: str = "episodic") -> int:
        self._ensure_init()
        if not self._client:
            return 0
        col = self._trajectories if collection == "trajectories" else self._episodic
        if not col or col.count() == 0:
            return 0
        all_docs = col.get(include=["documents", "metadatas"])
        if not all_docs or not all_docs["ids"]:
            return 0
        ids = all_docs["ids"]
        docs = all_docs["documents"]
        metas = all_docs["metadatas"] or [{}] * len(ids)
        col_name = col.name
        col_meta = col.metadata
        self._client.delete_collection(col_name)
        col = self._client.get_or_create_collection(name=col_name, metadata=col_meta)
        if collection == "trajectories":
            self._trajectories = col
        else:
            self._episodic = col
        count = 0
        batch_size = 20
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start:start + batch_size]
            batch_docs = docs[start:start + batch_size]
            batch_metas = metas[start:start + batch_size]
            embeddings = await self._embed(batch_docs)
            try:
                if embeddings:
                    col.add(ids=batch_ids, documents=batch_docs, embeddings=embeddings, metadatas=batch_metas)
                else:
                    col.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
                count += len(batch_ids)
            except Exception as e:
                logger.warning(f"Reindex: Failed batch at {start}: {e}")
        logger.info(f"EpisodicMemory: Reindexed {count}/{len(ids)} in {collection}")
        return count

    async def migrate_legacy(self) -> int:
        self._ensure_init()
        if not self._client:
            return 0
        legacy_path = Path(LEGACY_MEMORY_DIR)
        if not legacy_path.exists():
            return 0
        imported = 0
        for fp in legacy_path.glob("*.json"):
            try:
                data = json.loads(fp.read_text())
                content = data.get("content", "")
                tags = data.get("tags", [])
                if not content:
                    continue
                await self.add(
                    content=content,
                    tags=tags,
                    metadata={"source": "legacy_migration", "original_file": fp.name},
                )
                imported += 1
            except Exception as e:
                logger.warning(f"EpisodicMemory: Failed to import {fp.name}: {e}")
        if imported:
            logger.info(f"EpisodicMemory: Migrated {imported} legacy memory files")
            migrated_dir = legacy_path.parent / "memory_migrated"
            try:
                legacy_path.rename(migrated_dir)
            except Exception:
                pass
        return imported

    @staticmethod
    def _legacy_add(content: str, tags: Optional[List[str]] = None) -> str:
        mid = str(int(time.time()))
        try:
            os.makedirs(LEGACY_MEMORY_DIR, exist_ok=True)
            with open(os.path.join(LEGACY_MEMORY_DIR, f"{mid}.json"), "w") as f:
                json.dump({"content": content, "tags": tags or [], "timestamp": mid}, f)
        except Exception as e:
            logger.warning(f"Legacy memory save failed: {e}")
        return mid

    @staticmethod
    def _legacy_search(query: str, top_k: int = 5) -> List[Dict]:
        memory_path = Path(LEGACY_MEMORY_DIR)
        if not memory_path.exists():
            return []
        query_lower = query.lower()
        words = set(w for w in query_lower.split() if len(w) > 3)
        scored = []
        for fp in memory_path.glob("*.json"):
            try:
                data = json.loads(fp.read_text())
                content = data.get("content", "")
                tags_str = " ".join(data.get("tags", []))
                combined = (content + " " + tags_str).lower()
                score = sum(1 for w in words if w in combined)
                if score > 0:
                    scored.append({
                        "id": fp.stem,
                        "content": content,
                        "score": score / max(len(words), 1),
                        "metadata": {"tags": json.dumps(data.get("tags", []))},
                    })
            except Exception:
                continue
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
