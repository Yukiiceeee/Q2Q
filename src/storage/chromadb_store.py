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

        # Serialize WITHOUT embeddings (embeddings live in dedicated collections).
        # Cap by dropping whole items so the stored string is always valid JSON.
        def _json_capped(items: list[dict], cap: int) -> str:
            s = json.dumps(items, ensure_ascii=False)
            while len(s) > cap and items:
                items = items[:-1]
                s = json.dumps(items, ensure_ascii=False)
            return s

        props_json = _json_capped(
            [{k: v for k, v in p.to_dict().items() if k != "embedding"}
             for p in entry.propositions],
            120000,
        )
        fqs_json = _json_capped(
            [{k: v for k, v in fq.to_dict().items() if k != "embedding"}
             for fq in entry.fake_queries],
            120000,
        )

        meta_dict = {
            "session_id": _s(entry.session_id),
            "session_text": _s(entry.session_text)[:60000],
            "pre_session_text": _s(entry.pre_session_text)[:60000],
            "propositions_json": props_json,
            "fake_queries_json": fqs_json,
            "created_at": _s(entry.created_at),
            # Build-time facts the retrieval side needs back. Chroma metadata is
            # flat scalars only, so the two consumed keys are stored by name
            # rather than the whole `entry.metadata` dict.
            "session_date": _s(entry.metadata.get("session_date", "")),
            "scene": _s(entry.metadata.get("scene", "")),
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
        for key in ("session_date", "scene"):
            if meta.get(key):
                entry.metadata[key] = meta[key]
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

    async def get_proposition_embeddings(
        self, prop_ids: list[str], session_id: str
    ) -> dict[str, np.ndarray]:
        def _get():
            if not prop_ids:
                return {}
            full_ids = [f"{session_id}_{pid}" for pid in prop_ids]
            try:
                results = self._prop_collection.get(
                    ids=full_ids, include=["embeddings", "metadatas"]
                )
                if results["ids"] is None or len(results["ids"]) == 0:
                    return {}
                emb_map = {}
                for i in range(len(results["ids"])):
                    meta = results["metadatas"][i]
                    emb = results["embeddings"][i]
                    if emb is not None:
                        emb_map[meta.get("prop_id", "")] = np.asarray(emb, dtype=np.float32)
                return emb_map
            except Exception:
                return {}
        return await asyncio.to_thread(_get)

    # ==================== Content search ====================

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
        # Collect neighbours *before* the edge rows disappear, so their
        # parent_ids/child_ids can be reconciled — otherwise the DAG keeps
        # dangling ids and get_chain_context's BFS enqueues dead nodes.
        neighbours = await self._collect_neighbours(query_id)
        await asyncio.to_thread(_delete)
        await self.delete_edges_involving(query_id)
        await self._detach_neighbours(query_id, neighbours)

    async def _collect_neighbours(self, query_id: str) -> tuple[set[str], set[str]]:
        parents: set[str] = set()
        children: set[str] = set()

        def _collect():
            try:
                r1 = self._edges_collection.get(where={"dst_id": query_id})
                parents.update((m or {}).get("src_id", "") for m in (r1.get("metadatas") or []))
            except Exception:
                pass
            try:
                r2 = self._edges_collection.get(where={"src_id": query_id})
                children.update((m or {}).get("dst_id", "") for m in (r2.get("metadatas") or []))
            except Exception:
                pass

        await asyncio.to_thread(_collect)
        return {n for n in parents if n}, {n for n in children if n}

    async def _detach_neighbours(self, query_id: str, neighbours: tuple[set[str], set[str]]) -> None:
        parents, children = neighbours
        if not parents and not children:
            return

        def _detach():
            for node_id in parents:
                meta = self._fetch_meta(node_id)
                if meta is None:
                    continue
                ids = [i for i in self._split_list(meta.get("child_ids", "")) if i != query_id]
                if len(ids) != len(self._split_list(meta.get("child_ids", ""))):
                    self._fq_collection.update(ids=[node_id], metadatas=[{**meta, "child_ids": ",".join(ids)}])
            for node_id in children:
                meta = self._fetch_meta(node_id)
                if meta is None:
                    continue
                ids = [i for i in self._split_list(meta.get("parent_ids", "")) if i != query_id]
                if len(ids) != len(self._split_list(meta.get("parent_ids", ""))):
                    self._fq_collection.update(ids=[node_id], metadatas=[{**meta, "parent_ids": ",".join(ids)}])

        await asyncio.to_thread(_detach)

    def _fetch_meta(self, query_id: str) -> dict | None:
        r = self._fq_collection.get(ids=[query_id], include=["metadatas"])
        return r["metadatas"][0] if r["ids"] else None

    @staticmethod
    def _split_list(value) -> list[str]:
        if isinstance(value, list):
            return [v for v in value if v]
        return [v for v in str(value).split(",") if v]

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
