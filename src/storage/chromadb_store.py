from __future__ import annotations
import logging
import uuid
from pathlib import Path
import numpy as np

from src.schemas.models import MemoryEntry, FakeQuery
from src.storage.base import BaseMemoryStore

logger = logging.getLogger(__name__)


class ChromaDBStore(BaseMemoryStore):
    """ChromaDB-based memory store with native vector search."""

    def __init__(
        self,
        persist_directory: str = "data/chromadb",
        collection_name: str = "q2q_memories",
    ):
        import chromadb

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(self.persist_directory))

        # Collection for fake queries (primary search target)
        self._fq_collection = self._client.get_or_create_collection(
            name=f"{collection_name}_fake_queries",
            metadata={"hnsw:space": "cosine"},
        )

        # Collection for content chunks
        self._content_collection = self._client.get_or_create_collection(
            name=f"{collection_name}_content",
            metadata={"hnsw:space": "cosine"},
        )

        # Collection for memory metadata (stores full MemoryEntry without embeddings)
        self._meta_collection = self._client.get_or_create_collection(
            name=f"{collection_name}_meta",
        )

        logger.info(
            f"ChromaDB initialized: {persist_directory}, "
            f"fake_queries={self._fq_collection.count()}, "
            f"content_chunks={self._content_collection.count()}"
        )

    def save(self, entry: MemoryEntry) -> None:
        # Delete existing entry if updating
        self._delete_vectors(entry.memory_id)

        # Store fake query embeddings
        for fq in entry.fake_queries:
            if fq.embedding is not None:
                self._fq_collection.add(
                    ids=[fq.query_id],
                    embeddings=[fq.embedding.tolist()],
                    metadatas=[{"memory_id": entry.memory_id, "text": fq.text}],
                    documents=[fq.text],
                )

        # Store content chunk embeddings
        for i, emb in enumerate(entry.content_embeddings):
            chunk_id = f"{entry.memory_id}_chunk_{i}"
            self._content_collection.add(
                ids=[chunk_id],
                embeddings=[emb.tolist()],
                metadatas=[{"memory_id": entry.memory_id, "chunk_index": i}],
                documents=[entry.session_text[:500]],  # truncated for metadata
            )

        # Store full memory entry (without embedding data for space efficiency)
        meta_dict = {
            "memory_id": entry.memory_id,
            "session_text": entry.session_text,
            "fake_query_texts": "|".join(fq.text for fq in entry.fake_queries),
            "created_at": entry.created_at,
        }
        self._meta_collection.upsert(
            ids=[entry.memory_id],
            metadatas=[meta_dict],
            documents=[entry.session_text[:100]],
            embeddings=[[0.0]],
        )

        logger.info(
            f"ChromaDB saved: {entry.memory_id} | "
            f"{len(entry.fake_queries)} queries, {len(entry.content_embeddings)} chunks"
        )

    def load_all(self) -> list[MemoryEntry]:
        result = self._meta_collection.get(include=["metadatas"])
        entries = []
        for i, mid in enumerate(result["ids"]):
            meta = result["metadatas"][i]
            entry = MemoryEntry(
                memory_id=meta["memory_id"],
                session_text=meta.get("session_text", ""),
                created_at=meta.get("created_at", ""),
            )
            # Reconstruct fake queries (text only, embeddings loaded on demand)
            fq_texts = meta.get("fake_query_texts", "")
            if fq_texts:
                for text in fq_texts.split("|"):
                    if text.strip():
                        entry.fake_queries.append(
                            FakeQuery(text=text.strip(), memory_id=mid)
                        )
            entries.append(entry)
        return entries

    def get_by_id(self, memory_id: str) -> MemoryEntry | None:
        result = self._meta_collection.get(ids=[memory_id], include=["metadatas"])
        if not result["ids"]:
            return None
        meta = result["metadatas"][0]
        entry = MemoryEntry(
            memory_id=meta["memory_id"],
            session_text=meta.get("session_text", ""),
            created_at=meta.get("created_at", ""),
        )
        fq_texts = meta.get("fake_query_texts", "")
        if fq_texts:
            for text in fq_texts.split("|"):
                if text.strip():
                    entry.fake_queries.append(FakeQuery(text=text.strip(), memory_id=memory_id))
        return entry

    def delete(self, memory_id: str) -> bool:
        existing = self.get_by_id(memory_id)
        if existing is None:
            return False
        self._delete_vectors(memory_id)
        self._meta_collection.delete(ids=[memory_id])
        logger.info(f"ChromaDB deleted: {memory_id}")
        return True

    def count(self) -> int:
        return self._meta_collection.count()

    def clear(self) -> None:
        # Recreate collections
        client = self._client
        fq_name = self._fq_collection.name
        content_name = self._content_collection.name
        meta_name = self._meta_collection.name

        client.delete_collection(fq_name)
        client.delete_collection(content_name)
        client.delete_collection(meta_name)

        self._fq_collection = client.get_or_create_collection(
            name=fq_name, metadata={"hnsw:space": "cosine"}
        )
        self._content_collection = client.get_or_create_collection(
            name=content_name, metadata={"hnsw:space": "cosine"}
        )
        self._meta_collection = client.get_or_create_collection(name=meta_name)
        logger.info("ChromaDB cleared all collections")

    # --- Vector search methods (used by DualRetriever) ---

    def search_fake_queries(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[dict]:
        """Search fake query vectors, return [{memory_id, score, text}, ...]"""
        results = self._fq_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self._fq_collection.count() or 1),
            include=["metadatas", "distances", "documents"],
        )
        if not results["ids"] or not results["ids"][0]:
            return []

        hits = []
        for i, qid in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            score = 1.0 - distance  # chromadb cosine distance -> similarity
            meta = results["metadatas"][0][i]
            hits.append({
                "memory_id": meta["memory_id"],
                "score": max(score, 0.0),
                "text": meta.get("text", ""),
                "query_id": qid,
            })
        return hits

    def search_content(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[dict]:
        """Search content chunk vectors, return [{memory_id, score, chunk_index}, ...]"""
        results = self._content_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self._content_collection.count() or 1),
            include=["metadatas", "distances"],
        )
        if not results["ids"] or not results["ids"][0]:
            return []

        hits = []
        for i, cid in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            score = 1.0 - distance
            meta = results["metadatas"][0][i]
            hits.append({
                "memory_id": meta["memory_id"],
                "score": max(score, 0.0),
                "chunk_index": meta.get("chunk_index", 0),
            })
        return hits

    def search_content_for_memory(
        self,
        query_embedding: np.ndarray,
        memory_id: str,
        top_k: int = 3,
    ) -> float:
        """Search content chunks belonging to a specific memory_id, return best score."""
        try:
            count = self._content_collection.count()
            if count == 0:
                return 0.0
            results = self._content_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k * 10, count),
                where={"memory_id": memory_id},
                include=["distances"],
            )
            if not results["ids"] or not results["ids"][0]:
                return 0.0
            best_score = 0.0
            for dist in results["distances"][0]:
                score = max(1.0 - dist, 0.0)
                if score > best_score:
                    best_score = score
            return best_score
        except Exception:
            return 0.0

    def _delete_vectors(self, memory_id: str) -> None:
        """Delete all vectors associated with a memory_id."""
        # Delete fake queries
        try:
            fq_results = self._fq_collection.get(
                where={"memory_id": memory_id}, include=["metadatas"]
            )
            if fq_results["ids"]:
                self._fq_collection.delete(ids=fq_results["ids"])
        except Exception:
            pass

        # Delete content chunks
        try:
            content_results = self._content_collection.get(
                where={"memory_id": memory_id}, include=["metadatas"]
            )
            if content_results["ids"]:
                self._content_collection.delete(ids=content_results["ids"])
        except Exception:
            pass
