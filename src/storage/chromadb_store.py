from __future__ import annotations
import json
import logging
import uuid
from pathlib import Path
import numpy as np

from src.schemas.models import MemoryEntry, FakeQuery, KnowledgePoint
from src.storage.base import BaseMemoryStore

logger = logging.getLogger(__name__)


class ChromaDBStore(BaseMemoryStore):

    def __init__(
        self,
        persist_directory: str = "data/chromadb",
        collection_name: str = "q2q_memories",
    ):
        import chromadb

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(self.persist_directory))

        self._fq_collection = self._client.get_or_create_collection(
            name=f"{collection_name}_fake_queries",
            metadata={"hnsw:space": "cosine"},
        )

        self._content_collection = self._client.get_or_create_collection(
            name=f"{collection_name}_content",
            metadata={"hnsw:space": "cosine"},
        )

        self._paragraph_collection = self._client.get_or_create_collection(
            name=f"{collection_name}_paragraphs",
            metadata={"hnsw:space": "cosine"},
        )

        self._kp_collection = self._client.get_or_create_collection(
            name=f"{collection_name}_knowledge_points",
            metadata={"hnsw:space": "cosine"},
        )

        self._meta_collection = self._client.get_or_create_collection(
            name=f"{collection_name}_meta",
        )

        logger.info(
            f"ChromaDB initialized: {persist_directory}, "
            f"fake_queries={self._fq_collection.count()}, "
            f"content_chunks={self._content_collection.count()}, "
            f"paragraphs={self._paragraph_collection.count()}"
        )

    def save(self, entry: MemoryEntry) -> None:
        self._delete_vectors(entry.memory_id)

        # Store fake query embeddings
        for fq in entry.fake_queries:
            if fq.embedding is not None:
                self._fq_collection.add(
                    ids=[fq.query_id],
                    embeddings=[fq.embedding.tolist()],
                    metadatas=[{
                        "memory_id": entry.memory_id,
                        "text": fq.text,
                        "answer": "",
                        "parent_ids": ",".join(fq.parent_ids),
                        "depth": fq.depth,
                        "created_at": entry.created_at,
                    }],
                    documents=[fq.text],
                )

        # Store content chunk embeddings
        for i, emb in enumerate(entry.content_embeddings):
            chunk_id = f"{entry.memory_id}_chunk_{i}"
            self._content_collection.add(
                ids=[chunk_id],
                embeddings=[emb.tolist()],
                metadatas=[{"memory_id": entry.memory_id, "chunk_index": i}],
                documents=[entry.session_text[:500]],
            )

        # Store paragraph embeddings
        for i, (para, emb) in enumerate(zip(entry.paragraphs, entry.paragraph_embeddings)):
            para_id = f"{entry.memory_id}_para_{i}"
            self._paragraph_collection.add(
                ids=[para_id],
                embeddings=[emb.tolist()],
                metadatas=[{"memory_id": entry.memory_id, "paragraph_index": i}],
                documents=[para[:500]],
            )

        # Store knowledge point embeddings
        for i, (kp, emb) in enumerate(zip(entry.knowledge_points, entry.kp_embeddings)):
            kp_id = f"{entry.memory_id}_kp_{i}"
            kp_text = f"[{kp.time}] {kp.subject}: {kp.fact} ({kp.entities_or_values})"
            self._kp_collection.add(
                ids=[kp_id],
                embeddings=[emb.tolist()],
                metadatas=[{"memory_id": entry.memory_id, "kp_index": i}],
                documents=[kp_text[:500]],
            )

        # Store metadata with knowledge points
        kp_json = json.dumps(
            [kp.to_dict() for kp in entry.knowledge_points], ensure_ascii=False
        )
        paragraphs_json = json.dumps(entry.paragraphs, ensure_ascii=False)

        meta_dict = {
            "memory_id": entry.memory_id,
            "session_text": entry.session_text,
            "fake_query_texts": "|".join(fq.text for fq in entry.fake_queries),
            "fake_query_ids": "|".join(fq.query_id for fq in entry.fake_queries),
            "knowledge_points_json": kp_json,
            "paragraphs_json": paragraphs_json,
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
            f"{len(entry.fake_queries)} queries, {len(entry.content_embeddings)} chunks, "
            f"{len(entry.knowledge_points)} kps, {len(entry.paragraphs)} paragraphs"
        )

    def load_all(self) -> list[MemoryEntry]:
        result = self._meta_collection.get(include=["metadatas"])
        entries = []
        for i, mid in enumerate(result["ids"]):
            meta = result["metadatas"][i]
            entry = self._reconstruct_entry(meta, mid)
            entries.append(entry)
        return entries

    def get_by_id(self, memory_id: str) -> MemoryEntry | None:
        result = self._meta_collection.get(ids=[memory_id], include=["metadatas"])
        if not result["ids"]:
            return None
        meta = result["metadatas"][0]
        return self._reconstruct_entry(meta, memory_id)

    def _reconstruct_entry(self, meta: dict, memory_id: str) -> MemoryEntry:
        entry = MemoryEntry(
            memory_id=meta["memory_id"],
            session_text=meta.get("session_text", ""),
            created_at=meta.get("created_at", ""),
        )
        # Reconstruct fake queries
        fq_texts = meta.get("fake_query_texts", "")
        if fq_texts:
            texts = fq_texts.split("|")
            for text in texts:
                if text.strip():
                    entry.fake_queries.append(
                        FakeQuery(text=text.strip(), answer="", memory_id=memory_id)
                    )
        # Reconstruct knowledge points
        kp_json = meta.get("knowledge_points_json", "")
        if kp_json:
            try:
                kp_list = json.loads(kp_json)
                entry.knowledge_points = [KnowledgePoint.from_dict(kp) for kp in kp_list]
            except (json.JSONDecodeError, TypeError):
                pass
        # Reconstruct paragraphs
        para_json = meta.get("paragraphs_json", "")
        if para_json:
            try:
                entry.paragraphs = json.loads(para_json)
            except (json.JSONDecodeError, TypeError):
                pass
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
        client = self._client
        fq_name = self._fq_collection.name
        content_name = self._content_collection.name
        para_name = self._paragraph_collection.name
        kp_name = self._kp_collection.name
        meta_name = self._meta_collection.name

        client.delete_collection(fq_name)
        client.delete_collection(content_name)
        client.delete_collection(para_name)
        client.delete_collection(kp_name)
        client.delete_collection(meta_name)

        self._fq_collection = client.get_or_create_collection(
            name=fq_name, metadata={"hnsw:space": "cosine"}
        )
        self._content_collection = client.get_or_create_collection(
            name=content_name, metadata={"hnsw:space": "cosine"}
        )
        self._paragraph_collection = client.get_or_create_collection(
            name=para_name, metadata={"hnsw:space": "cosine"}
        )
        self._kp_collection = client.get_or_create_collection(
            name=kp_name, metadata={"hnsw:space": "cosine"}
        )
        self._meta_collection = client.get_or_create_collection(name=meta_name)
        logger.info("ChromaDB cleared all collections")

    # --- Vector search methods ---

    def search_fake_queries(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[dict]:
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
            score = 1.0 - distance
            meta = results["metadatas"][0][i]
            parent_ids_str = meta.get("parent_ids", "")
            hits.append({
                "memory_id": meta["memory_id"],
                "score": max(score, 0.0),
                "text": meta.get("text", ""),
                "query_id": qid,
                "answer": "",
                "parent_ids": [p for p in parent_ids_str.split(",") if p] if parent_ids_str else [],
                "depth": meta.get("depth", 0),
                "created_at": meta.get("created_at", ""),
            })
        return hits

    def search_content(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[dict]:
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

    def search_paragraphs_for_memory(
        self,
        query_embedding: np.ndarray,
        memory_id: str,
        top_k: int = 3,
    ) -> list[dict]:
        try:
            count = self._paragraph_collection.count()
            if count == 0:
                return []
            results = self._paragraph_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k * 5, count),
                where={"memory_id": memory_id},
                include=["distances", "documents"],
            )
            if not results["ids"] or not results["ids"][0]:
                return []

            hits = []
            for i, pid in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                score = max(1.0 - distance, 0.0)
                text = results["documents"][0][i] if results["documents"] else ""
                hits.append({"score": score, "text": text})

            hits.sort(key=lambda h: h["score"], reverse=True)
            return hits[:top_k]
        except Exception:
            return []

    def search_kps_for_memory(
        self,
        query_embedding: np.ndarray,
        memory_id: str,
        top_k: int = 10,
    ) -> list[dict]:
        try:
            count = self._kp_collection.count()
            if count == 0:
                return []
            results = self._kp_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k * 5, count),
                where={"memory_id": memory_id},
                include=["distances", "metadatas", "documents"],
            )
            if not results["ids"] or not results["ids"][0]:
                return []

            hits = []
            for i, kid in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                score = max(1.0 - distance, 0.0)
                meta = results["metadatas"][0][i]
                hits.append({
                    "score": score,
                    "kp_index": meta.get("kp_index", 0),
                    "text": results["documents"][0][i] if results["documents"] else "",
                })

            hits.sort(key=lambda h: h["score"], reverse=True)
            return hits[:top_k]
        except Exception:
            return []

    def search_all_fake_queries(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.80,
        max_results: int = 50,
    ) -> list[dict]:
        count = self._fq_collection.count()
        if count == 0:
            return []

        results = self._fq_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(max_results, count),
            include=["metadatas", "distances", "documents"],
        )
        if not results["ids"] or not results["ids"][0]:
            return []

        hits = []
        for i, qid in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            score = 1.0 - distance
            if score >= threshold:
                meta = results["metadatas"][0][i]
                parent_ids_str = meta.get("parent_ids", "")
                hits.append({
                    "query_id": qid,
                    "memory_id": meta.get("memory_id", ""),
                    "score": score,
                    "text": meta.get("text", ""),
                    "answer": "",
                    "parent_ids": [p for p in parent_ids_str.split(",") if p] if parent_ids_str else [],
                    "depth": meta.get("depth", 0),
                    "created_at": meta.get("created_at", ""),
                })
        return hits

    def get_fake_query_by_id(self, query_id: str) -> dict | None:
        try:
            results = self._fq_collection.get(ids=[query_id], include=["metadatas"])
            if not results["ids"]:
                return None
            meta = results["metadatas"][0]
            parent_ids_str = meta.get("parent_ids", "")
            return {
                "query_id": query_id,
                "text": meta.get("text", ""),
                "answer": "",
                "memory_id": meta.get("memory_id", ""),
                "parent_ids": [p for p in parent_ids_str.split(",") if p] if parent_ids_str else [],
                "depth": meta.get("depth", 0),
                "created_at": meta.get("created_at", ""),
            }
        except Exception:
            return None

    def get_fake_queries_by_ids(self, query_ids: list[str]) -> list[dict]:
        if not query_ids:
            return []
        try:
            results = self._fq_collection.get(ids=query_ids, include=["metadatas"])
            if not results["ids"]:
                return []
            nodes = []
            for i, qid in enumerate(results["ids"]):
                meta = results["metadatas"][i]
                parent_ids_str = meta.get("parent_ids", "")
                nodes.append({
                    "query_id": qid,
                    "text": meta.get("text", ""),
                    "answer": "",
                    "memory_id": meta.get("memory_id", ""),
                    "parent_ids": [p for p in parent_ids_str.split(",") if p] if parent_ids_str else [],
                    "depth": meta.get("depth", 0),
                    "created_at": meta.get("created_at", ""),
                })
            return nodes
        except Exception:
            return []

    def _delete_vectors(self, memory_id: str) -> None:
        try:
            fq_results = self._fq_collection.get(
                where={"memory_id": memory_id}, include=["metadatas"]
            )
            if fq_results["ids"]:
                self._fq_collection.delete(ids=fq_results["ids"])
        except Exception:
            pass

        try:
            content_results = self._content_collection.get(
                where={"memory_id": memory_id}, include=["metadatas"]
            )
            if content_results["ids"]:
                self._content_collection.delete(ids=content_results["ids"])
        except Exception:
            pass

        try:
            para_results = self._paragraph_collection.get(
                where={"memory_id": memory_id}, include=["metadatas"]
            )
            if para_results["ids"]:
                self._paragraph_collection.delete(ids=para_results["ids"])
        except Exception:
            pass

        try:
            kp_results = self._kp_collection.get(
                where={"memory_id": memory_id}, include=["metadatas"]
            )
            if kp_results["ids"]:
                self._kp_collection.delete(ids=kp_results["ids"])
        except Exception:
            pass
