"""Temporal DAG Graph Manager - builds and maintains the memory graph."""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

import numpy as np

from src.schemas.models import FakeQuery, Edge, EdgeType, Proposition
from src.embedding.base import BaseEmbeddingProvider
from src.storage.base import BaseMemoryStore

logger = logging.getLogger(__name__)


class MemoryGraphManager:

    def __init__(
        self,
        store: BaseMemoryStore,
        embedding_provider: BaseEmbeddingProvider,
        evolve_threshold: float = 0.6,
        diverge_low: float = 0.2,
        diverge_high: float = 0.6,
        max_parents: int = 3,
        similarity_top_k: int = 5,
        maintenance_interval: int = 10,
        merge_overlap_threshold: float = 0.9,
        max_chain_length: int = 6,
        chain_sample_interval: int = 2,
    ):
        self.store = store
        self.embedding_provider = embedding_provider
        self.evolve_threshold = evolve_threshold
        self.diverge_low = diverge_low
        self.diverge_high = diverge_high
        self.max_parents = max_parents
        self.similarity_top_k = similarity_top_k
        self.maintenance_interval = maintenance_interval
        self.merge_overlap_threshold = merge_overlap_threshold
        self.max_chain_length = max_chain_length
        self.chain_sample_interval = chain_sample_interval
        self._operation_count = 0

    async def integrate_new_fqs(self, new_fqs: list[FakeQuery]) -> list[Edge]:
        if not new_fqs:
            return []

        all_new_edges: list[Edge] = []

        for fq in new_fqs:
            if fq.embedding is None:
                continue
            edges = await self._build_edges_for_node(fq)
            all_new_edges.extend(edges)

        if all_new_edges:
            await self.store.save_edges(all_new_edges)
            await self._update_node_relationships(all_new_edges)

        self._operation_count += len(new_fqs)
        logger.info(
            f"GraphManager: integrated {len(new_fqs)} FQs, "
            f"created {len(all_new_edges)} edges, "
            f"op_count={self._operation_count}"
        )

        if self._operation_count >= self.maintenance_interval:
            await self._lifecycle_maintenance()
            self._operation_count = 0

        return all_new_edges

    async def _build_edges_for_node(self, fq: FakeQuery) -> list[Edge]:
        candidates = await self.store.search_fake_queries(
            fq.embedding, top_k=self.similarity_top_k * 3
        )

        candidates = [
            c for c in candidates
            if c["query_id"] != fq.query_id
            and c.get("created_at", "") < fq.created_at
        ]
        candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
        candidates = candidates[:self.similarity_top_k]

        if not candidates:
            return []

        raw_edges: list[Edge] = []
        for cand in candidates:
            overlap = self._compute_proposition_overlap(
                fq.answer_source, cand.get("answer_source", [])
            )
            similarity = cand["score"]

            if overlap >= self.evolve_threshold:
                edge = Edge(
                    src_id=cand["query_id"],
                    dst_id=fq.query_id,
                    edge_type=EdgeType.EVOLVES_TO.value,
                    similarity=similarity,
                    prop_overlap=overlap,
                )
                raw_edges.append(edge)
            elif overlap >= self.diverge_low:
                edge = Edge(
                    src_id=cand["query_id"],
                    dst_id=fq.query_id,
                    edge_type=EdgeType.DIVERGES_TO.value,
                    similarity=similarity,
                    prop_overlap=overlap,
                )
                raw_edges.append(edge)

        pruned = self._prune_edges(fq.query_id, raw_edges)
        return pruned

    def _compute_proposition_overlap(
        self, source_a: list[str], source_b: list[str]
    ) -> float:
        if not source_a or not source_b:
            return 0.0
        set_a = set(source_a)
        set_b = set(source_b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        if union == 0:
            return 0.0
        return intersection / union

    async def compute_soft_proposition_overlap(
        self,
        props_a: list[Proposition],
        props_b: list[Proposition],
        threshold: float = 0.7,
    ) -> float:
        if not props_a or not props_b:
            return 0.0

        emb_a = [p.embedding for p in props_a if p.embedding is not None]
        emb_b = [p.embedding for p in props_b if p.embedding is not None]

        if not emb_a or not emb_b:
            return 0.0

        match_count = 0
        for ea in emb_a:
            best_sim = max(
                BaseEmbeddingProvider.cosine_similarity(ea, eb)
                for eb in emb_b
            )
            if best_sim >= threshold:
                match_count += 1

        return match_count / len(emb_a)

    def _prune_edges(self, target_id: str, edges: list[Edge]) -> list[Edge]:
        if not edges:
            return []

        # Rule 1: Transitivity pruning
        src_ids = {e.src_id for e in edges}
        ancestor_of: dict[str, set] = defaultdict(set)
        for e in edges:
            ancestor_of[e.src_id].add(e.src_id)

        redundant_srcs: set[str] = set()
        for e1 in edges:
            for e2 in edges:
                if e1.src_id != e2.src_id:
                    if e2.src_id in ancestor_of.get(e1.src_id, set()):
                        redundant_srcs.add(e2.src_id)

        edges = [e for e in edges if e.src_id not in redundant_srcs]

        # Rule 2: Max parents limit
        edges = sorted(edges, key=lambda e: e.similarity, reverse=True)
        edges = edges[:self.max_parents]

        return edges

    async def _update_node_relationships(self, edges: list[Edge]) -> None:
        for edge in edges:
            dst_node = await self.store.get_fake_query_by_id(edge.dst_id)
            if dst_node:
                current_parents = dst_node.get("parent_ids", [])
                if edge.src_id not in current_parents:
                    current_parents.append(edge.src_id)
                    await self.store.update_fq_metadata(
                        edge.dst_id, {"parent_ids": ",".join(current_parents)}
                    )

            src_node = await self.store.get_fake_query_by_id(edge.src_id)
            if src_node:
                current_children = src_node.get("child_ids", [])
                if edge.dst_id not in current_children:
                    current_children.append(edge.dst_id)
                    await self.store.update_fq_metadata(
                        edge.src_id, {"child_ids": ",".join(current_children)}
                    )

    async def _lifecycle_maintenance(self) -> None:
        logger.info("GraphManager: running lifecycle maintenance...")
        await self._merge_high_overlap_nodes()
        await self._compress_long_chains()
        logger.info("GraphManager: lifecycle maintenance complete")

    async def _merge_high_overlap_nodes(self) -> None:
        all_edges = await self.store.get_all_edges()
        merge_candidates = [
            e for e in all_edges
            if e.edge_type == EdgeType.EVOLVES_TO.value
            and e.prop_overlap >= self.merge_overlap_threshold
        ]

        merged_count = 0
        for edge in merge_candidates:
            src_node = await self.store.get_fake_query_by_id(edge.src_id)
            dst_node = await self.store.get_fake_query_by_id(edge.dst_id)

            if not src_node or not dst_node:
                continue

            src_sources = src_node.get("answer_source", [])
            dst_sources = dst_node.get("answer_source", [])
            merged_sources = list(set(dst_sources + src_sources))
            await self.store.update_fq_metadata(
                edge.dst_id, {"answer_source": ",".join(merged_sources)}
            )

            await self.store.delete_node(edge.src_id)
            merged_count += 1

        if merged_count > 0:
            logger.info(f"  Merged {merged_count} high-overlap node pairs")

    async def _compress_long_chains(self) -> None:
        all_edges = await self.store.get_all_edges()

        children_map: dict[str, list[str]] = defaultdict(list)
        parents_map: dict[str, list[str]] = defaultdict(list)
        all_nodes: set[str] = set()

        for edge in all_edges:
            children_map[edge.src_id].append(edge.dst_id)
            parents_map[edge.dst_id].append(edge.src_id)
            all_nodes.add(edge.src_id)
            all_nodes.add(edge.dst_id)

        roots = [n for n in all_nodes if n not in parents_map]

        compressed_count = 0
        for root in roots:
            chain = self._trace_longest_chain(root, children_map)
            if len(chain) <= self.max_chain_length:
                continue

            keep_indices = self._select_keep_indices(len(chain))
            remove_indices = [i for i in range(len(chain)) if i not in keep_indices]

            for idx in remove_indices:
                remove_id = chain[idx]
                nearest_kept_idx = min(keep_indices, key=lambda k: abs(k - idx))
                nearest_kept_id = chain[nearest_kept_idx]

                remove_node = await self.store.get_fake_query_by_id(remove_id)
                kept_node = await self.store.get_fake_query_by_id(nearest_kept_id)

                if remove_node and kept_node:
                    src_sources = remove_node.get("answer_source", [])
                    dst_sources = kept_node.get("answer_source", [])
                    merged = list(set(dst_sources + src_sources))
                    await self.store.update_fq_metadata(
                        nearest_kept_id, {"answer_source": ",".join(merged)}
                    )

                await self.store.delete_node(remove_id)
                compressed_count += 1

        if compressed_count > 0:
            logger.info(f"  Compressed {compressed_count} nodes from long chains")

    def _trace_longest_chain(self, root: str, children_map: dict[str, list[str]]) -> list[str]:
        best_chain: list[str] = []

        def dfs(node: str, path: list[str]):
            nonlocal best_chain
            path.append(node)
            if len(path) > len(best_chain):
                best_chain = list(path)
            for child in children_map.get(node, []):
                if child not in path:
                    dfs(child, path)
            path.pop()

        dfs(root, [])
        return best_chain

    def _select_keep_indices(self, chain_length: int) -> set[int]:
        keep = {0, chain_length - 1}
        i = self.chain_sample_interval + 1
        while i < chain_length - 1:
            keep.add(i)
            i += self.chain_sample_interval + 1
        return keep

    # --- Public query: bidirectional traversal ---

    async def get_chain_context(
        self, query_ids: list[str], depth: int = 3
    ) -> list[dict]:
        visited: set[str] = set()
        result_nodes: list[dict] = []

        queue = list(query_ids)
        current_depth = 0

        while queue and current_depth < depth:
            next_queue: list[str] = []
            batch_ids = [qid for qid in queue if qid not in visited]
            if not batch_ids:
                break

            nodes = await self.store.get_fake_queries_by_ids(batch_ids)
            node_map = {n["query_id"]: n for n in nodes}

            for qid in batch_ids:
                if qid in visited:
                    continue
                visited.add(qid)
                node = node_map.get(qid)
                if node:
                    result_nodes.append(node)
                    # Bidirectional: parents + children
                    for pid in node.get("parent_ids", []):
                        if pid not in visited:
                            next_queue.append(pid)
                    for cid in node.get("child_ids", []):
                        if cid not in visited:
                            next_queue.append(cid)

            queue = next_queue
            current_depth += 1

        return result_nodes
