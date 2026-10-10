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
        link_similarity: float = 0.82,
        max_parents: int = 3,
        similarity_top_k: int = 5,
        maintenance_interval: int = 10,
        merge_overlap_threshold: float = 0.9,
        max_chain_length: int = 6,
        chain_sample_interval: int = 2,
        max_chain_nodes: int = 60,
    ):
        self.store = store
        self.embedding_provider = embedding_provider
        self.evolve_threshold = evolve_threshold
        self.diverge_low = diverge_low
        self.diverge_high = diverge_high
        # Proposition overlap alone leaves most nodes isolated: an FQ binds one
        # or two propositions, so same-session Jaccard is effectively binary (0
        # when the two FQs cite different propositions, 1 when they cite the
        # same one) and cross-session soft overlap thresholds the same way.
        # Measured on a LoCoMo chunk, only ~25% of FQs ended up with a parent.
        # Two FQs can be about the same thread while citing different evidence
        # — exactly the link temporal questions need — so high embedding
        # similarity also earns an edge. The candidate score is already
        # computed, making this free; `similarity_top_k` and `max_parents`
        # still bound how many edges any node can gain.
        self.link_similarity = link_similarity
        self.max_parents = max_parents
        self.similarity_top_k = similarity_top_k
        self.maintenance_interval = maintenance_interval
        self.merge_overlap_threshold = merge_overlap_threshold
        self.max_chain_length = max_chain_length
        self.chain_sample_interval = chain_sample_interval
        # Frontier budget for chain expansion; <= 0 disables the cap.
        self.max_chain_nodes = max_chain_nodes
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

        # Lazy-fetched embeddings of the new FQ's source propositions
        # (only needed for cross-session candidates)
        fq_prop_embs: list[np.ndarray] | None = None

        raw_edges: list[Edge] = []
        for cand in candidates:
            cand_session = cand.get("session_id", "")
            if cand_session and cand_session == fq.session_id:
                # Same session: prop IDs share a namespace, exact Jaccard is valid
                overlap = self._compute_proposition_overlap(
                    fq.answer_source, cand.get("answer_source", [])
                )
            else:
                # Cross session: prop IDs (P1, P2...) are session-scoped, so raw
                # ID Jaccard produces spurious matches — use embedding soft overlap
                if fq_prop_embs is None:
                    emb_map = await self.store.get_proposition_embeddings(
                        fq.answer_source, fq.session_id
                    )
                    fq_prop_embs = list(emb_map.values())
                cand_emb_map = await self.store.get_proposition_embeddings(
                    cand.get("answer_source", []), cand_session
                )
                overlap = self._soft_overlap_from_embeddings(
                    fq_prop_embs, list(cand_emb_map.values())
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
            elif overlap >= self.diverge_low or similarity >= self.link_similarity:
                edge = Edge(
                    src_id=cand["query_id"],
                    dst_id=fq.query_id,
                    edge_type=EdgeType.DIVERGES_TO.value,
                    similarity=similarity,
                    prop_overlap=overlap,
                )
                raw_edges.append(edge)

        pruned = await self._prune_edges(raw_edges)
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

    def _soft_overlap_from_embeddings(
        self,
        embs_a: list[np.ndarray],
        embs_b: list[np.ndarray],
        threshold: float = 0.7,
    ) -> float:
        """Jaccard-style soft overlap: matched props (bidirectional) / union size."""
        if not embs_a or not embs_b:
            return 0.0
        matched_a = sum(
            1 for ea in embs_a
            if max(BaseEmbeddingProvider.cosine_similarity(ea, eb) for eb in embs_b) >= threshold
        )
        matched_b = sum(
            1 for eb in embs_b
            if max(BaseEmbeddingProvider.cosine_similarity(eb, ea) for ea in embs_a) >= threshold
        )
        inter = (matched_a + matched_b) / 2.0
        union = len(embs_a) + len(embs_b) - inter
        return inter / union if union > 0 else 0.0

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

    async def _prune_edges(self, edges: list[Edge]) -> list[Edge]:
        if not edges:
            return []

        # Rule 1: Max parents limit — keep the strongest candidates first.
        # Edge type outranks similarity. An EVOLVES_TO edge means the two FQs
        # answer from the same propositions, which is stronger evidence of a
        # real thread than any embedding score; ordering by similarity alone
        # would let a merely similar-sounding DIVERGES_TO edge displace it.
        _rank = {EdgeType.EVOLVES_TO.value: 0}
        edges = sorted(
            edges,
            key=lambda e: (_rank.get(e.edge_type, 1), -e.similarity),
        )
        edges = edges[:self.max_parents]

        # Rule 2: Transitivity pruning. All surviving edges share the same
        # dst, so redundancy is only visible by walking the *existing* graph:
        # if candidate A is an ancestor of candidate B, the edge A->dst is
        # implied by B->dst and can be dropped, leaving the temporally closer
        # parent.
        if len(edges) > 1:
            ancestors = {
                e.src_id: await self._collect_ancestors(e.src_id) for e in edges
            }
            redundant = {
                e2.src_id
                for e1 in edges
                for e2 in edges
                if e1.src_id != e2.src_id and e2.src_id in ancestors[e1.src_id]
            }
            if redundant:
                edges = [e for e in edges if e.src_id not in redundant]

        return edges

    async def _collect_ancestors(self, query_id: str) -> set[str]:
        """Return the strict ancestors of a node by walking parent_ids upward."""
        ancestors: set[str] = set()
        visited: set[str] = {query_id}
        frontier = [query_id]

        while frontier and len(visited) < self.max_chain_nodes:
            batch = [q for q in frontier if q]
            frontier = []
            for node in await self.store.get_fake_queries_by_ids(batch):
                for pid in node.get("parent_ids", []):
                    if pid and pid not in visited:
                        visited.add(pid)
                        ancestors.add(pid)
                        frontier.append(pid)

        return ancestors

    async def _update_node_relationships(self, edges: list[Edge]) -> None:
        if not edges:
            return

        touched = {e.src_id for e in edges} | {e.dst_id for e in edges}
        nodes = {n["query_id"]: n for n in await self.store.get_fake_queries_by_ids(list(touched))}

        # Accumulate the full desired state per node, then write once — the
        # previous per-endpoint write issued two round-trips per edge and could
        # clobber a sibling edge's addition to the same node.
        parents: dict[str, list[str]] = {nid: list(node.get("parent_ids", [])) for nid, node in nodes.items()}
        children: dict[str, list[str]] = {nid: list(node.get("child_ids", [])) for nid, node in nodes.items()}

        for edge in edges:
            if edge.dst_id in parents and edge.src_id not in parents[edge.dst_id]:
                parents[edge.dst_id].append(edge.src_id)
            if edge.src_id in children and edge.dst_id not in children[edge.src_id]:
                children[edge.src_id].append(edge.dst_id)

        for node_id, node in nodes.items():
            updates = {}
            if parents[node_id] != list(node.get("parent_ids", [])):
                updates["parent_ids"] = ",".join(parents[node_id])
            if children[node_id] != list(node.get("child_ids", [])):
                updates["child_ids"] = ",".join(children[node_id])
            if updates:
                await self.store.update_fq_metadata(node_id, updates)

    async def _lifecycle_maintenance(self) -> None:
        logger.info("GraphManager: running lifecycle maintenance...")
        all_edges = await self.store.get_all_edges()
        removed = await self._merge_high_overlap_nodes(all_edges)
        if removed:
            # Compression must not work off the pre-merge snapshot: it would
            # re-introduce edges to nodes the merge just deleted.
            all_edges = await self.store.get_all_edges()
        await self._compress_long_chains(all_edges)
        logger.info("GraphManager: lifecycle maintenance complete")

    async def _merge_high_overlap_nodes(self, all_edges: list[Edge]) -> set[str]:
        candidates = [
            (e.src_id, e.dst_id)
            for e in all_edges
            if e.edge_type == EdgeType.EVOLVES_TO.value
            and e.prop_overlap >= self.merge_overlap_threshold
        ]
        if not candidates:
            return set()

        # Near-duplicate nodes form connected components; each component
        # collapses to a single survivor. Merging pair-by-pair against a fixed
        # snapshot lost the sources of transitively absorbed nodes.
        parent = {n: n for pair in candidates for n in pair}

        def find(n: str) -> str:
            while parent[n] != n:
                parent[n] = parent[parent[n]]
                n = parent[n]
            return n

        for a, b in candidates:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        groups: dict[str, set[str]] = defaultdict(set)
        for node in parent:
            groups[find(node)].add(node)

        removed: set[str] = set()
        for members in groups.values():
            if len(members) < 2:
                continue
            survivor = await self._pick_merge_survivor(members)
            if survivor is None:
                continue
            await self._contract_nodes(survivor, members - {survivor}, all_edges)
            removed |= members - {survivor}

        if removed:
            logger.info(f"  Merged {len(removed)} high-overlap nodes into survivors")

        return removed

    async def _pick_merge_survivor(self, members: set[str]) -> str | None:
        nodes = await self.store.get_fake_queries_by_ids(list(members))
        if not nodes:
            return None
        # Keep the most recent node so the merged entry carries the freshest
        # text/metadata; created_at is ISO-8601, so lexical order is temporal.
        return max(nodes, key=lambda n: n.get("created_at", ""))["query_id"]

    async def _contract_nodes(
        self, survivor_id: str, removed_ids: set[str], all_edges: list[Edge]
    ) -> None:
        if not removed_ids:
            return

        members = removed_ids | {survivor_id}
        edge_meta = {(e.src_id, e.dst_id): e for e in all_edges}
        parents_map, children_map = self._index_edges(all_edges)

        # Union of every member's sources, read in one batch.
        nodes = {
            n["query_id"]: n
            for n in await self.store.get_fake_queries_by_ids(list(members))
        }
        sources: list[str] = []
        for node in nodes.values():
            sources.extend(node.get("answer_source", []))
        if sources:
            await self.store.update_fq_metadata(
                survivor_id, {"answer_source": ",".join(dict.fromkeys(sources))}
            )

        # Re-point external neighbours at the survivor so contraction does not
        # sever the DAG.
        bridges: list[Edge] = []
        seen: set[tuple[str, str]] = set()
        for node_id in removed_ids:
            for src in parents_map.get(node_id, []):
                if src in members or (src, survivor_id) in edge_meta or (src, survivor_id) in seen:
                    continue
                seen.add((src, survivor_id))
                bridges.append(self._bridge_edge(src, survivor_id, edge_meta.get((src, node_id))))
            for dst in children_map.get(node_id, []):
                if dst in members or (survivor_id, dst) in edge_meta or (survivor_id, dst) in seen:
                    continue
                seen.add((survivor_id, dst))
                bridges.append(self._bridge_edge(survivor_id, dst, edge_meta.get((node_id, dst))))

        if bridges:
            await self.store.save_edges(bridges)
            await self._update_node_relationships(bridges)

        for node_id in removed_ids:
            await self.store.delete_node(node_id)

    @staticmethod
    def _bridge_edge(src_id: str, dst_id: str, original: Edge | None) -> Edge:
        return Edge(
            src_id=src_id,
            dst_id=dst_id,
            edge_type=original.edge_type if original else EdgeType.EVOLVES_TO.value,
            similarity=original.similarity if original else 0.0,
            prop_overlap=original.prop_overlap if original else 0.0,
        )

    async def _compress_long_chains(self, all_edges: list[Edge]) -> None:
        compressed_count = 0

        # Each pass shortens one chain and re-reads the graph. Compressing
        # several chains against one snapshot was unsound: a node scheduled for
        # removal on one chain could be picked as a bridge parent on another,
        # and a removal's own parent could already be gone by the time its
        # bridge was written.
        while True:
            graph = self._index_edges(all_edges)
            if graph is None:
                break
            children_map, parents_map = graph

            chain = None
            for root in [n for n in children_map if n not in parents_map]:
                candidate = self._trace_longest_chain(root, children_map)
                if len(candidate) > self.max_chain_length:
                    chain = candidate
                    break
            if chain is None:
                break

            compressed_count += await self._compress_one_chain(chain, parents_map, all_edges)
            all_edges = await self.store.get_all_edges()

        if compressed_count > 0:
            logger.info(f"  Compressed {compressed_count} nodes from long chains")

    def _index_edges(
        self, all_edges: list[Edge]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]] | None:
        if not all_edges:
            return None
        children_map: dict[str, list[str]] = defaultdict(list)
        parents_map: dict[str, list[str]] = defaultdict(list)
        for edge in all_edges:
            children_map[edge.src_id].append(edge.dst_id)
            parents_map[edge.dst_id].append(edge.src_id)
        return children_map, parents_map

    async def _compress_one_chain(
        self,
        chain: list[str],
        parents_map: dict[str, list[str]],
        all_edges: list[Edge],
    ) -> int:
        keep_indices = self._select_keep_indices(len(chain))

        # Every node on the chain is classified as kept or absorbed in one
        # snapshot, so a bridge parent that is itself about to be removed gets
        # attributed to the neighbour that absorbs it instead of dangling.
        removed: dict[str, str] = {}
        for idx in range(len(chain)):
            if idx in keep_indices:
                continue
            nearest = min(keep_indices, key=lambda k: abs(k - idx))
            removed[chain[idx]] = chain[nearest]

        def absorb(target: str) -> str:
            seen: set[str] = set()
            while target in removed and target not in seen:
                seen.add(target)
                target = removed[target]
            return target

        nodes = {n["query_id"]: n for n in await self.store.get_fake_queries_by_ids(list(chain))}
        sources: dict[str, list[str]] = {}

        for remove_id, direct_kept in removed.items():
            kept_id = absorb(direct_kept)

            remove_node = nodes.get(remove_id)
            kept_node = nodes.get(kept_id)
            if remove_node and kept_node:
                merged = sources.get(kept_id)
                if merged is None:
                    merged = list(kept_node.get("answer_source", []))
                merged.extend(remove_node.get("answer_source", []))
                sources[kept_id] = merged

            # Bridge the gap so removing the node does not cut the chain.
            await self._bridge_removed_node(
                remove_id, kept_id, parents_map, all_edges, absorb
            )
            await self.store.delete_node(remove_id)

        for kept_id, merged in sources.items():
            await self.store.update_fq_metadata(
                kept_id, {"answer_source": ",".join(dict.fromkeys(merged))}
            )

        return len(removed)

    async def _bridge_removed_node(
        self,
        remove_id: str,
        kept_id: str,
        parents_map: dict[str, list[str]],
        all_edges: list[Edge],
        absorb,
    ) -> None:
        if remove_id == kept_id:
            return

        edge_meta: dict[tuple[str, str], Edge] = {
            (e.src_id, e.dst_id): e for e in all_edges
        }
        bridges: list[Edge] = []
        seen: set[tuple[str, str]] = set()

        for parent_id in parents_map.get(remove_id, []):
            # The parent may itself be scheduled for removal in this same pass;
            # re-point from whichever node finally survives it, otherwise the
            # bridge introduces a fresh dangling endpoint.
            src_id = absorb(parent_id)
            if src_id == kept_id or (src_id, kept_id) in edge_meta or (src_id, kept_id) in seen:
                continue
            seen.add((src_id, kept_id))
            bridges.append(self._bridge_edge(src_id, kept_id, edge_meta.get((parent_id, remove_id))))

        if bridges:
            await self.store.save_edges(bridges)
            await self._update_node_relationships(bridges)

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

        # Bidirectional BFS over a dense graph grows exponentially with depth.
        # BFS order already follows the seed ranking, so truncating the frontier
        # keeps the nodes closest to the best seeds and bounds both store
        # round-trips and downstream proposition reranking.
        budget = self.max_chain_nodes

        while queue and current_depth < depth:
            batch_ids = [qid for qid in queue if qid not in visited]
            if not batch_ids:
                break
            if budget > 0:
                remaining = budget - len(result_nodes)
                if remaining <= 0:
                    break
                batch_ids = batch_ids[:remaining]

            next_queue: list[str] = []
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
