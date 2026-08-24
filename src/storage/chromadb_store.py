"""ChromaDB storage backend with collections for FQs, propositions, content, edges, and meta."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.schemas.models import MemoryEntry, FakeQuery, Proposition, Edge
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
        self._prop_collection = self._client.get_or_create_collection(
            name=f"{collection_name}_propositions",
            metadata={"hnsw:space": "cosine"},
        )
        self._content_collection = self._client.get_or_create_collection(
            name=f"{collection_name}_content",
            metadata={"hnsw:space": "cosine"},
        )
        self._edges_collection = self._client.get_or_create_collection(
            name=f"{collection_name}_edges",
        )
        self._meta_collection = self._client.get_or_create_collection(
            name=f"{collection_name}_meta",
        )

        logger.info(
            f"ChromaDB initialized: {persist_directory}, "
            f"fq={self._fq_collection.count()}, "
            f"props={self._prop_collection.count()}, "
            f"content={self._content_collection.count()}, "
            f"edges={self._edges_collection.count()}"
        )

    # ==================== Save ====================

    def _save_sync(self, entry: MemoryEntry) -> None:
        self._delete_session_data_sync(entry.session_id)

        def _s(v) -> str:
            return v if isinstance(v, str) else ("" if v is None else str(v))

        for fq in entry.fake_queries:
            if fq.embedding is not None:
                self._fq_collection.add(
                    ids=[fq.query_id],
                    embeddings=[fq.embedding.tolist()],
                    metadatas=[{
                        "session_id": _s(entry.session_id),
                        "text": _s(fq.text),
                        "answer_source": ",".join(fq.answer_source or []),
                        "time": _s(fq.time),
                        "parent_ids": ",".join(fq.parent_ids or []),
                        "child_ids": ",".join(fq.child_ids or []),
                        "depth": fq.depth or 0,
                        "access_count": fq.access_count or 0,
                        "created_at": _s(fq.created_at),
                    }],
                    documents=[_s(fq.text)],
                )

        for prop in entry.propositions:
            if prop.embedding is not None:
                self._prop_collection.add(
                    ids=[f"{entry.session_id}_{prop.prop_id}"],
                    embeddings=[prop.embedding.tolist()],
                    metadatas=[{
                        "session_id": _s(entry.session_id),
                        "prop_id": _s(prop.prop_id),
                        "text": _s(prop.text),
                        "source": _s(prop.source),
                        "time": _s(prop.time),
                        "subject": _s(prop.subject),
                        "entities": ",".join(prop.entities or []),
                    }],
                    documents=[_s(prop.text)],
                )

        for i, emb in enumerate(entry.content_embeddings):
            chunk_id = f"{entry.session_id}_chunk_{i}"
            self._content_collection.add(
                ids=[chunk_id],
                embeddings=[emb.tolist()],
                metadatas=[{"session_id": entry.session_id, "chunk_index": i}],
                documents=[entry.pre_session_text[:500] if entry.pre_session_text else ""],
            )

        props_json = json.dumps([p.to_dict() for p in entry.propositions], ensure_ascii=False)
        fqs_json = json.dumps([fq.to_dict() for fq in entry.fake_queries], ensure_ascii=False)

        meta_dict = {
            "session_id": _s(entry.session_id),
            "session_text": _s(entry.session_text)[:10000],
            "pre_session_text": _s(entry.pre_session_text)[:10000],
            "propositions_json": props_json[:40000],
            "fake_queries_json": fqs_json[:40000],
            "created_at": _s(entry.created_at),
        }
        self._meta_collection.upsert(
            ids=[entry.session_id],
            metadatas=[meta_dict],
            documents=[entry.session_text[:200]],
            embeddings=[[0.0]],
        )

        logger.info(
            f"ChromaDB saved: {entry.session_id} | "
            f"{len(entry.fake_queries)} FQs, {len(entry.propositions)} props, "
            f"{len(entry.content_embeddings)} chunks"
        )

    def _delete_session_data_sync(self, session_id: str) -> None:
        for coll in [self._fq_collection, self._prop_collection, self._content_collection]:
            try:
                results = coll.get(where={"session_id": session_id}, include=["metadatas"])
                if results["ids"]:
                    coll.delete(ids=results["ids"])
            except Exception:
                pass

    async def save(self, entry: MemoryEntry) -> None:
        await asyncio.to_thread(self._save_sync, entry)

    # ==================== Load/Get ====================

    async def load_all(self) -> list[MemoryEntry]:
        def _load():
            result = self._meta_collection.get(include=["metadatas"])
            entries = []
            for i, sid in enumerate(result["ids"]):
                meta = result["metadatas"][i]
                entry = self._reconstruct_entry(meta)
                entries.append(entry)
            return entries
        return await asyncio.to_thread(_load)

    async def get_by_id(self, session_id: str) -> Optional[MemoryEntry]:
        def _get():
            result = self._meta_collection.get(ids=[session_id], include=["metadatas"])
            if not result["ids"]:
                return None
            return self._reconstruct_entry(result["metadatas"][0])
        return await asyncio.to_thread(_get)

    def _reconstruct_entry(self, meta: dict) -> MemoryEntry:
        entry = MemoryEntry(
            session_id=meta["session_id"],
            session_text=meta.get("session_text", ""),
            pre_session_text=meta.get("pre_session_text", ""),
            created_at=meta.get("created_at", ""),
        )
        props_json = meta.get("propositions_json", "")
        if props_json:
            try:
                for p_data in json.loads(props_json):
                    entry.propositions.append(Proposition.from_dict(p_data))
            except (json.JSONDecodeError, TypeError):
                pass
        fqs_json = meta.get("fake_queries_json", "")
        if fqs_json:
            try:
                for fq_data in json.loads(fqs_json):
                    entry.fake_queries.append(FakeQuery.from_dict(fq_data))
            except (json.JSONDecodeError, TypeError):
                pass
        return entry

    async def delete(self, session_id: str) -> bool:
        def _delete():
            result = self._meta_collection.get(ids=[session_id], include=["metadatas"])
            if not result["ids"]:
                return False
            self._delete_session_data_sync(session_id)
            self._meta_collection.delete(ids=[session_id])
            return True
        return await asyncio.to_thread(_delete)

    async def count(self) -> int:
        return await asyncio.to_thread(self._meta_collection.count)

    async def clear(self) -> None:
        def _clear():
            for coll_attr in ["_fq_collection", "_prop_collection", "_content_collection",
                              "_edges_collection", "_meta_collection"]:
                coll = getattr(self, coll_attr)
                name = coll.name
                space = coll.metadata.get("hnsw:space") if coll.metadata else None
                self._client.delete_collection(name)
                if space:
                    setattr(self, coll_attr, self._client.get_or_create_collection(
                        name=name, metadata={"hnsw:space": space}
                    ))
                else:
                    setattr(self, coll_attr, self._client.get_or_create_collection(name=name))
            logger.info("ChromaDB cleared all collections")
        await asyncio.to_thread(_clear)

    # ==================== FQ Search ====================

    async def search_fake_queries(
        self, query_embedding: np.ndarray, top_k: int = 20
    ) -> list[dict]:
        def _search():
            count = self._fq_collection.count()
            if count == 0:
                return []
            results = self._fq_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, count),
                include=["metadatas", "distances", "documents"],
            )
            if not results["ids"] or not results["ids"][0]:
                return []
            hits = []
            for i, qid in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                score = max(1.0 - distance, 0.0)
                meta = results["metadatas"][0][i]
                hits.append(self._parse_fq_hit(qid, score, meta))
            return hits
        return await asyncio.to_thread(_search)

    async def batch_search_fake_queries(
        self, query_embeddings: list[np.ndarray], top_k: int = 20
    ) -> list[list[dict]]:
        def _search():
            count = self._fq_collection.count()
            if count == 0:
                return [[] for _ in query_embeddings]
            n = min(top_k, count)
            emb_list = [e.tolist() for e in query_embeddings]
            results = self._fq_collection.query(
                query_embeddings=emb_list,
                n_results=n,
                include=["metadatas", "distances"],
            )
            all_hits = []
            for qi in range(len(query_embeddings)):
                if not results["ids"] or qi >= len(results["ids"]) or not results["ids"][qi]:
                    all_hits.append([])
                    continue
                hits = []
                for i, qid in enumerate(results["ids"][qi]):
                    distance = results["distances"][qi][i]
                    score = max(1.0 - distance, 0.0)
                    meta = results["metadatas"][qi][i]
                    hits.append(self._parse_fq_hit(qid, score, meta))
                all_hits.append(hits)
            return all_hits
        return await asyncio.to_thread(_search)

    def _parse_fq_hit(self, query_id: str, score: float, meta: dict) -> dict:
        answer_source_str = meta.get("answer_source", "")
        parent_ids_str = meta.get("parent_ids", "")
        child_ids_str = meta.get("child_ids", "")
        return {
            "query_id": query_id,
            "session_id": meta.get("session_id", ""),
            "text": meta.get("text", ""),
            "score": score,
            "answer_source": [s for s in answer_source_str.split(",") if s],
            "parent_ids": [p for p in parent_ids_str.split(",") if p],
            "child_ids": [c for c in child_ids_str.split(",") if c],
            "depth": meta.get("depth", 0),
            "time": meta.get("time", ""),
            "created_at": meta.get("created_at", ""),
        }

    async def get_fake_query_by_id(self, query_id: str) -> Optional[dict]:
        def _get():
            try:
                results = self._fq_collection.get(ids=[query_id], include=["metadatas"])
                if not results["ids"]:
                    return None
                meta = results["metadatas"][0]
                return self._parse_fq_hit(query_id, 0.0, meta)
            except Exception:
                return None
        return await asyncio.to_thread(_get)

    async def get_fake_queries_by_ids(self, query_ids: list[str]) -> list[dict]:
        def _get():
            if not query_ids:
                return []
            try:
                results = self._fq_collection.get(ids=query_ids, include=["metadatas"])
                if not results["ids"]:
                    return []
                nodes = []
                for i, qid in enumerate(results["ids"]):
                    meta = results["metadatas"][i]
                    nodes.append(self._parse_fq_hit(qid, 0.0, meta))
                return nodes
            except Exception:
                return []
        return await asyncio.to_thread(_get)

    # ==================== Proposition operations ====================

    async def get_propositions_by_ids(self, prop_ids: list[str], session_id: str) -> list[dict]:
        def _get():
            full_ids = [f"{session_id}_{pid}" for pid in prop_ids]
            try:
                results = self._prop_collection.get(ids=full_ids, include=["metadatas", "documents"])
                if not results["ids"]:
                    return []
                props = []
                for i, pid in enumerate(results["ids"]):
                    meta = results["metadatas"][i]
                    props.append({
                        "prop_id": meta.get("prop_id", ""),
                        "text": meta.get("text", results["documents"][i] if results["documents"] else ""),
                        "session_id": meta.get("session_id", ""),
                        "source": meta.get("source", ""),
                        "time": meta.get("time", ""),
                        "subject": meta.get("subject", ""),
                        "entities": meta.get("entities", "").split(",") if meta.get("entities") else [],
                    })
                return props
            except Exception:
                return []
        return await asyncio.to_thread(_get)

    async def search_propositions(
        self, query_embedding: np.ndarray, session_id: str, top_k: int = 5
    ) -> list[dict]:
        def _search():
            count = self._prop_collection.count()
            if count == 0:
                return []
            try:
                results = self._prop_collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=min(top_k * 3, count),
                    where={"session_id": session_id},
                    include=["metadatas", "distances", "documents"],
                )
                if not results["ids"] or not results["ids"][0]:
                    return []
                hits = []
                for i, pid in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    score = max(1.0 - distance, 0.0)
                    meta = results["metadatas"][0][i]
                    hits.append({
                        "prop_id": meta.get("prop_id", ""),
                        "text": meta.get("text", ""),
                        "score": score,
                        "session_id": meta.get("session_id", ""),
                    })
                return hits[:top_k]
            except Exception:
                return []
        return await asyncio.to_thread(_search)

    # ==================== Content search ====================

    async def search_content(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> list[dict]:
        def _search():
            count = self._content_collection.count()
            if count == 0:
                return []
            results = self._content_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, count),
                include=["metadatas", "distances"],
            )
            if not results["ids"] or not results["ids"][0]:
                return []
            hits = []
            for i, cid in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                score = max(1.0 - distance, 0.0)
                meta = results["metadatas"][0][i]
                hits.append({
                    "session_id": meta.get("session_id", ""),
                    "score": score,
                    "chunk_index": meta.get("chunk_index", 0),
                })
            return hits
        return await asyncio.to_thread(_search)

    async def batch_search_content(
        self, query_embeddings: list[np.ndarray], top_k: int = 10
    ) -> list[dict]:
        def _search():
            count = self._content_collection.count()
            if count == 0:
                return []
            emb_list = [e.tolist() for e in query_embeddings]
            results = self._content_collection.query(
                query_embeddings=emb_list,
                n_results=min(top_k, count),
                include=["metadatas", "distances"],
            )
            session_scores: dict[str, float] = {}
            for qi in range(len(query_embeddings)):
                if not results["ids"] or qi >= len(results["ids"]):
                    continue
                for i, cid in enumerate(results["ids"][qi]):
                    distance = results["distances"][qi][i]
                    score = max(1.0 - distance, 0.0)
                    sid = results["metadatas"][qi][i].get("session_id", "")
                    if sid not in session_scores or score > session_scores[sid]:
                        session_scores[sid] = score
            return [{"session_id": sid, "score": sc} for sid, sc in
                    sorted(session_scores.items(), key=lambda x: x[1], reverse=True)]
        return await asyncio.to_thread(_search)

    # ==================== Edge (DAG) operations ====================

    async def save_edges(self, edges: list[Edge]) -> None:
        def _save():
            for edge in edges:
                self._edges_collection.upsert(
                    ids=[edge.edge_id],
                    metadatas=[{
                        "src_id": edge.src_id,
                        "dst_id": edge.dst_id,
                        "edge_type": edge.edge_type,
                        "similarity": edge.similarity,
                        "prop_overlap": edge.prop_overlap,
                    }],
                    documents=[f"{edge.src_id}->{edge.dst_id}"],
                    embeddings=[[0.0]],
                )
        await asyncio.to_thread(_save)

    async def get_edges_from(self, src_id: str) -> list[Edge]:
        def _get():
            try:
                results = self._edges_collection.get(
                    where={"src_id": src_id}, include=["metadatas"]
                )
                if not results["ids"]:
                    return []
                return [self._parse_edge(results["ids"][i], results["metadatas"][i])
                        for i in range(len(results["ids"]))]
            except Exception:
                return []
        return await asyncio.to_thread(_get)

    async def get_edges_to(self, dst_id: str) -> list[Edge]:
        def _get():
            try:
                results = self._edges_collection.get(
                    where={"dst_id": dst_id}, include=["metadatas"]
                )
                if not results["ids"]:
                    return []
                return [self._parse_edge(results["ids"][i], results["metadatas"][i])
                        for i in range(len(results["ids"]))]
            except Exception:
                return []
        return await asyncio.to_thread(_get)

    async def get_all_edges(self) -> list[Edge]:
        def _get():
            results = self._edges_collection.get(include=["metadatas"])
            if not results["ids"]:
                return []
            return [self._parse_edge(results["ids"][i], results["metadatas"][i])
                    for i in range(len(results["ids"]))]
        return await asyncio.to_thread(_get)

    async def delete_edges_involving(self, node_id: str) -> None:
        def _delete():
            try:
                r1 = self._edges_collection.get(where={"src_id": node_id})
                if r1["ids"]:
                    self._edges_collection.delete(ids=r1["ids"])
            except Exception:
                pass
            try:
                r2 = self._edges_collection.get(where={"dst_id": node_id})
                if r2["ids"]:
                    self._edges_collection.delete(ids=r2["ids"])
            except Exception:
                pass
        await asyncio.to_thread(_delete)

    async def delete_node(self, query_id: str) -> None:
        def _delete():
            try:
                self._fq_collection.delete(ids=[query_id])
            except Exception:
                pass
        await asyncio.to_thread(_delete)
        await self.delete_edges_involving(query_id)

    def _parse_edge(self, edge_id: str, meta: dict) -> Edge:
        return Edge(
            edge_id=edge_id,
            src_id=meta.get("src_id", ""),
            dst_id=meta.get("dst_id", ""),
            edge_type=meta.get("edge_type", "EVOLVES_TO"),
            similarity=meta.get("similarity", 0.0),
            prop_overlap=meta.get("prop_overlap", 0.0),
        )

    # ==================== FQ metadata update ====================

    async def update_fq_metadata(self, query_id: str, updates: dict) -> None:
        def _update():
            try:
                existing = self._fq_collection.get(ids=[query_id], include=["metadatas", "embeddings"])
                if not existing["ids"]:
                    return
                meta = existing["metadatas"][0]
                meta.update(updates)
                self._fq_collection.update(ids=[query_id], metadatas=[meta])
            except Exception as e:
                logger.warning(f"Failed to update FQ metadata {query_id}: {e}")
        await asyncio.to_thread(_update)
