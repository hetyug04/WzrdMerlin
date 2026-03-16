"""
WzrdMerlin v3 — Memory

Primary backend: SQLiteMemoryStore using FTS5 full-text search.
No external dependencies, no embedding calls.

Optional ChromaDB upgrade: install chromadb and get_memory() will
automatically use memory_chroma.EpisodicMemory instead.

Public API is identical to v2 — all callers need zero changes.
"""
import asyncio
import json
import logging
import os
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_memory_instance = None

WORKSPACE = os.getenv("MERLIN_WORKSPACE", "/workspace")
DEFAULT_DB_PATH = os.path.join(WORKSPACE, ".merlin", "memory.db")


class SQLiteMemoryStore:
    """
    Primary memory backend. Uses SQLite FTS5 for full-text search.

    Schema:
      memories(id TEXT PK, content TEXT, tags TEXT, metadata TEXT,
               timestamp INTEGER, collection TEXT)
      memories_fts USING fts5(content, content='memories', content_rowid='rowid')

    Two collections: 'episodic' and 'trajectories'.
    FTS5 score normalization: score = 1.0 / (1.0 + abs(bm25_rank))
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                tags TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                timestamp INTEGER NOT NULL,
                collection TEXT NOT NULL DEFAULT 'episodic'
            )
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(content, content='memories', content_rowid='rowid')
        """)
        # Trigger to keep FTS index in sync
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ai
            AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content)
                VALUES (new.rowid, new.content);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ad
            AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_au
            AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
                INSERT INTO memories_fts(rowid, content)
                VALUES (new.rowid, new.content);
            END
        """)
        conn.commit()

    async def add(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        collection: str = "episodic",
    ) -> str:
        doc_id = str(uuid.uuid4())
        tags_json = json.dumps(tags or [])
        meta_json = json.dumps(metadata or {})
        ts = int(time.time())
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO memories (id, content, tags, metadata, timestamp, collection) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, content, tags_json, meta_json, ts, collection),
        )
        conn.commit()
        return doc_id

    async def search(
        self,
        query: str,
        top_k: int = 5,
        collection: str = "episodic",
        min_score: float = 0.0,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        if not query or not query.strip():
            return []
        conn = self._get_conn()

        # FTS5 MATCH query — escape special characters
        safe_query = query.replace('"', '""').replace("'", "''")
        try:
            rows = conn.execute(
                """
                SELECT m.id, m.content, m.tags, m.metadata, m.timestamp,
                       bm25(memories_fts) AS rank
                FROM memories_fts
                JOIN memories m ON m.rowid = memories_fts.rowid
                WHERE memories_fts MATCH ? AND m.collection = ?
                ORDER BY rank
                LIMIT ?
                """,
                (safe_query, collection, top_k * 2),  # fetch extra to filter by min_score
            ).fetchall()
        except sqlite3.OperationalError:
            # FTS query syntax error — fall back to LIKE
            try:
                rows = conn.execute(
                    """
                    SELECT id, content, tags, metadata, timestamp, 0.0 as rank
                    FROM memories
                    WHERE content LIKE ? AND collection = ?
                    LIMIT ?
                    """,
                    (f"%{query}%", collection, top_k),
                ).fetchall()
            except Exception:
                return []

        results = []
        for row in rows:
            # BM25 returns negative values; lower (more negative) = better match
            bm25_rank = row["rank"] if row["rank"] is not None else 0.0
            score = 1.0 / (1.0 + abs(bm25_rank))
            if score < min_score:
                continue
            try:
                meta = json.loads(row["metadata"]) if row["metadata"] else {}
            except (json.JSONDecodeError, TypeError):
                meta = {}
            results.append({
                "id": row["id"],
                "content": row["content"],
                "score": score,
                "metadata": meta,
            })
            if len(results) >= top_k:
                break

        return results

    async def recall(self, query: str, top_k: int = 3) -> str:
        """Return search results formatted for prompt injection."""
        results = await self.search(query, top_k=top_k)
        if not results:
            return ""
        lines = []
        for r in results:
            score_pct = int(r["score"] * 100)
            lines.append(f"[{score_pct}% match] {r['content']}")
        return "\n---\n".join(lines)

    async def prune(self, max_age_days: int = 90) -> int:
        cutoff = int(time.time()) - (max_age_days * 86400)
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM memories WHERE timestamp < ?", (cutoff,)
        )
        conn.commit()
        return cursor.rowcount

    def count(self, collection: str = "episodic") -> int:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as n FROM memories WHERE collection = ?", (collection,)
        ).fetchone()
        return row["n"] if row else 0

    async def migrate_legacy(self) -> int:
        """Import /workspace/memory/*.json files (idempotent, one-time)."""
        memory_dir = os.path.join(WORKSPACE, "memory")
        if not os.path.isdir(memory_dir):
            return 0
        migrated = 0
        for fname in os.listdir(memory_dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(memory_dir, fname)
            try:
                with open(fpath) as f:
                    data = json.load(f)
                content = data.get("content") or data.get("text") or str(data)
                tags = data.get("tags", [])
                meta = {k: v for k, v in data.items() if k not in ("content", "text", "tags")}
                await self.add(content=content, tags=tags, metadata=meta)
                migrated += 1
            except Exception as e:
                logger.warning(f"MEMORY: Failed to migrate {fname}: {e}")
        if migrated:
            logger.info(f"MEMORY: Migrated {migrated} legacy JSON entries to SQLite")
        return migrated

    async def reindex(self, collection: str = "episodic") -> int:
        """No-op for SQLite (FTS auto-indexes via triggers). Returns 0."""
        return 0


# ── Factory ───────────────────────────────────────────────────────────────────

def get_memory():
    """
    Singleton factory. Returns ChromaDB store if available, else SQLite FTS5.
    """
    global _memory_instance
    if _memory_instance is None:
        try:
            import chromadb  # noqa: F401
            from src.core.memory_chroma import EpisodicMemory
            _memory_instance = EpisodicMemory()
            logger.info("MEMORY: Using ChromaDB vector store")
        except (ImportError, TypeError):
            _memory_instance = SQLiteMemoryStore()
            logger.info("MEMORY: Using SQLite FTS5 store")
    return _memory_instance
