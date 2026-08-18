"""Experiment 1: Basic Alignment Comparison (Q2Q vs Q2C).

For each (true_query, answer_session) pair:
1. Compute sim_Q2Q = max cos(emb(true_query), emb(fake_query_i))
2. Compute sim_Q2C = max cos(emb(true_query), content_chunk_j)
3. Compute gap = sim_Q2Q - sim_Q2C
4. Aggregate statistics with paired t-test, Cohen's d, category breakdown.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import numpy as np

from experiment.src.experiment_base import ExperimentBase
from experiment.src.metrics import AlignmentMetrics, AlignmentResult

logger = logging.getLogger(__name__)


class Exp1Alignment(ExperimentBase):

    async def run(self) -> dict:
        logger.info("=" * 60)
        logger.info("Experiment 1: Basic Alignment Comparison (Q2Q vs Q2C)")
        logger.info("=" * 60)

        start_time = time.time()
        loader = self.load_data()

        # Phase 1: Ensure all embeddings are computed
        await self._ensure_embeddings(loader)

        # Phase 2: Compute alignment for each query
        results = await self._compute_alignments(loader)

        # Phase 3: Aggregate statistics
        overall_stats = AlignmentMetrics.paired_statistics(results)
        category_stats = AlignmentMetrics.category_breakdown(results, loader.queries)

        elapsed = time.time() - start_time
        output = {
            "experiment": "exp1_alignment",
            "dataset": self.dataset_config["dataset"]["name"],
            "n_queries": len(results),
            "elapsed_seconds": round(elapsed, 2),
            "overall": overall_stats,
            "by_category": category_stats,
            "sample_results": [
                {
                    "query_id": r.query_id,
                    "sim_q2q": round(r.sim_q2q, 4),
                    "sim_q2c": round(r.sim_q2c, 4),
                    "gap": round(r.gap, 4),
                    "best_fq_text": r.best_fq_text,
                }
                for r in results[:20]
            ],
        }

        self.save_results(output, "exp1_alignment.json")
        self._print_summary(output)
        return output

    async def _ensure_embeddings(self, loader) -> None:
        """Ensure all required embeddings exist in the store."""
        logger.info("Phase 1: Checking/computing embeddings...")

        # Collect unique answer session IDs
        answer_sids = set()
        for q in loader.queries:
            answer_sids.update(q.evidence_session_ids)

        # Embed sessions that need content embeddings
        sessions_to_embed = [
            s for s in loader.sessions
            if s.session_id in answer_sids and not self.store.has_session(s.session_id)
        ]
        if sessions_to_embed:
            logger.info(f"  Computing content embeddings for {len(sessions_to_embed)} sessions...")
            for i, sess in enumerate(sessions_to_embed):
                chunks, chunk_meta = await self.embedding_provider.embed_session_turns(sess.turns)
                self.store.save_session_content(sess.session_id, chunks)
                for m in chunk_meta:
                    m["session_id"] = sess.session_id
                self.store.save_chunk_metadata(sess.session_id, chunk_meta)
                if (i + 1) % 50 == 0:
                    logger.info(f"    Session embeddings: {i + 1}/{len(sessions_to_embed)}")
            logger.info(f"  Done: {len(sessions_to_embed)} sessions embedded.")

        # Check if fake queries exist (must be pre-generated)
        fq_missing = [
            sid for sid in answer_sids
            if not self.store.has_fake_queries(sid)
        ]
        if fq_missing:
            logger.warning(
                f"  {len(fq_missing)} sessions missing fake query embeddings. "
                f"Run generation step first (--step generate)."
            )

        # Embed true queries
        queries_to_embed = [
            q for q in loader.queries
            if not self.store.has_true_query(q.query_id)
        ]
        if queries_to_embed:
            logger.info(f"  Computing true query embeddings for {len(queries_to_embed)} queries...")
            texts = [q.text for q in queries_to_embed]
            embeddings = await self.embedding_provider.embed_batch(texts)
            for q, emb in zip(queries_to_embed, embeddings):
                self.store.save_true_query(q.query_id, emb)
            logger.info(f"  Done: {len(queries_to_embed)} queries embedded.")

    async def _compute_alignments(self, loader) -> list[AlignmentResult]:
        """Compute Q2Q and Q2C alignment for each query."""
        logger.info("Phase 2: Computing alignment scores...")
        results = []
        skipped = 0

        for q in loader.queries:
            query_emb = self.store.get_true_query_embedding(q.query_id)
            if query_emb is None:
                skipped += 1
                continue

            # Get best alignment across all evidence sessions
            best_result = None
            for sid in q.evidence_session_ids:
                fq_embs = self.store.get_fake_query_embeddings(sid)
                content_embs = self.store.get_session_content(sid)

                if fq_embs is None or content_embs is None:
                    continue

                sim_q2q, fq_idx, sim_q2c, chunk_idx = AlignmentMetrics.compute_alignment(
                    query_emb, fq_embs, content_embs
                )

                if best_result is None or sim_q2q > best_result.sim_q2q:
                    fq_texts = self.store.get_fake_query_texts(sid)
                    fq_text = fq_texts[fq_idx] if fq_texts and fq_idx >= 0 else ""
                    best_result = AlignmentResult(
                        query_id=q.query_id,
                        sim_q2q=sim_q2q,
                        sim_q2c=sim_q2c,
                        gap=sim_q2q - sim_q2c,
                        best_fq_text=fq_text,
                        best_fq_idx=fq_idx,
                        best_chunk_idx=chunk_idx,
                    )

            if best_result:
                results.append(best_result)

        if skipped:
            logger.warning(f"  Skipped {skipped} queries (missing embeddings)")
        logger.info(f"  Computed alignment for {len(results)} queries.")
        return results

    def _print_summary(self, output: dict) -> None:
        overall = output.get("overall", {})
        print("\n" + "=" * 60)
        print("  Experiment 1 Results: Q2Q vs Q2C Alignment")
        print("=" * 60)
        print(f"  Dataset: {output['dataset']}")
        print(f"  Queries analyzed: {output['n_queries']}")
        print(f"  Time: {output['elapsed_seconds']}s")
        print()
        print(f"  Q2Q Similarity:  mean={overall.get('sim_q2q_mean', 0):.4f} "
              f"(std={overall.get('sim_q2q_std', 0):.4f})")
        print(f"  Q2C Similarity:  mean={overall.get('sim_q2c_mean', 0):.4f} "
              f"(std={overall.get('sim_q2c_std', 0):.4f})")
        print(f"  Gap (Q2Q-Q2C):   mean={overall.get('gap_mean', 0):.4f} "
              f"(std={overall.get('gap_std', 0):.4f})")
        print()
        print(f"  Paired t-test:   t={overall.get('t_statistic', 0):.4f}, "
              f"p={overall.get('p_value', 1):.2e}")
        print(f"  Cohen's d:       {overall.get('cohens_d', 0):.4f}")
        print(f"  Q2Q win rate:    {overall.get('q2q_win_rate', 0):.1%} "
              f"({overall.get('q2q_wins', 0)}/{output['n_queries']})")
        print()

        # Category breakdown
        by_cat = output.get("by_category", {})
        if by_cat:
            print("  --- By Category ---")
            for cat, stats in sorted(by_cat.items()):
                print(
                    f"    [{cat}] n={stats.get('n_samples', 0):>3}, "
                    f"gap={stats.get('gap_mean', 0):+.4f}, "
                    f"win_rate={stats.get('q2q_win_rate', 0):.0%}"
                )
        print("=" * 60 + "\n")
