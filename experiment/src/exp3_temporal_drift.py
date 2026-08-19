"""Experiment 3: Temporal Drift Verification (Q2Q vs Q2P/Q2N/Q2R/Q2C).

Simulate progressive memory accumulation by injecting sessions chronologically.
At each checkpoint, measure retrieval rank for all 5 methods,
showing Q2Q maintains stable ranking while baselines degrade as the pool grows.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment.src.experiment_base import ExperimentBase

logger = logging.getLogger(__name__)

DATE_FORMATS = [
    "%I:%M %p on %d %B, %Y",
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%m/%d/%Y",
]

METHOD_KEYS = ["q2q", "q2p", "q2n", "q2r", "q2c"]
METHOD_COLORS = {
    "q2q": "#2196F3",
    "q2p": "#9C27B0",
    "q2n": "#E91E63",
    "q2r": "#009688",
    "q2c": "#FF9800",
}
METHOD_LABELS = {
    "q2q": "Q2Q",
    "q2p": "Q2P",
    "q2n": "Q2N",
    "q2r": "Q2R",
    "q2c": "Q2C",
}
METHOD_MARKERS = {
    "q2q": "o",
    "q2p": "D",
    "q2n": "^",
    "q2r": "v",
    "q2c": "s",
}


class Exp3TemporalDrift(ExperimentBase):

    async def run(self) -> dict:
        logger.info("=" * 60)
        logger.info("Experiment 3: Temporal Drift Verification (5 methods)")
        logger.info("=" * 60)

        start_time = time.time()
        loader = self.load_data()

        self.exp_logger.start_phase("embeddings")
        await self._ensure_embeddings(loader)
        self.exp_logger.end_phase("embeddings")

        self.exp_logger.start_phase("temporal_ordering")
        ordered_sids = self._build_temporal_order(loader)
        self.exp_logger.end_phase("temporal_ordering", {"n_sessions": len(ordered_sids)})

        n_checkpoints = self.base_config.get("experiment", {}).get("n_checkpoints", 10)
        step_size = max(1, len(ordered_sids) // n_checkpoints)
        checkpoints = list(range(step_size, len(ordered_sids), step_size))
        if checkpoints[-1] != len(ordered_sids):
            checkpoints.append(len(ordered_sids))

        self.exp_logger.start_phase("accumulation")
        checkpoint_results = self._simulate_accumulation(loader, ordered_sids, checkpoints)
        self.exp_logger.end_phase("accumulation", {"n_checkpoints": len(checkpoint_results)})

        self.exp_logger.start_phase("visualization")
        fig_paths = self._visualize(checkpoint_results)
        self.exp_logger.end_phase("visualization")

        elapsed = time.time() - start_time
        output = {
            "experiment": "exp3_temporal_drift",
            "dataset": self.dataset_config["dataset"]["name"],
            "n_sessions": len(ordered_sids),
            "n_checkpoints": len(checkpoint_results),
            "elapsed_seconds": round(elapsed, 2),
            "checkpoints": checkpoint_results,
            "figures": {k: str(v) for k, v in fig_paths.items()},
        }

        self.save_results(output, "exp3_temporal_drift.json")
        self.save_experiment_log(output, "exp3_temporal_drift_log.json")
        self._print_summary(output)
        return output

    async def _ensure_embeddings(self, loader) -> None:
        logger.info("Phase 1: Checking embeddings...")
        answer_sids = set()
        for q in loader.queries:
            answer_sids.update(q.evidence_session_ids)

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

        queries_to_embed = [
            q for q in loader.queries if not self.store.has_true_query(q.query_id)
        ]
        if queries_to_embed:
            logger.info(f"  Computing true query embeddings for {len(queries_to_embed)} queries...")
            texts = [q.text for q in queries_to_embed]
            embeddings = await self.embedding_provider.embed_batch(texts)
            for q, emb in zip(queries_to_embed, embeddings):
                self.store.save_true_query(q.query_id, emb)
        logger.info("  Embeddings ready.")

    def _parse_date(self, date_str: str) -> datetime | None:
        if not date_str:
            return None
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        return None

    def _build_temporal_order(self, loader) -> list[str]:
        logger.info("Phase 2: Building temporal order...")
        dated, undated = [], []
        for s in loader.sessions:
            dt = self._parse_date(s.date)
            if dt:
                dated.append((dt, s.session_id))
            else:
                undated.append(s.session_id)

        dated.sort(key=lambda x: x[0])
        ordered = [sid for _, sid in dated] + undated
        logger.info(
            f"  Ordered {len(dated)} dated sessions + {len(undated)} undated. "
            f"Date range: {dated[0][0].date() if dated else '?'} to {dated[-1][0].date() if dated else '?'}"
        )
        return ordered

    def _simulate_accumulation(
        self,
        loader,
        ordered_sids: list[str],
        checkpoints: list[int],
    ) -> list[dict]:
        logger.info("Phase 3: Simulating progressive accumulation (5 methods)...")

        query_evidence = {q.query_id: set(q.evidence_session_ids) for q in loader.queries}
        query_embs = {}
        for q in loader.queries:
            emb = self.store.get_true_query_embedding(q.query_id)
            if emb is not None:
                query_embs[q.query_id] = emb / (np.linalg.norm(emb) + 1e-10)

        dim = next(iter(query_embs.values())).shape[0] if query_embs else 1024

        pools = {}
        sid_arrays = {}
        for m in METHOD_KEYS:
            pools[m] = np.empty((0, dim), dtype=np.float32)
            sid_arrays[m] = []

        pool_sid_set: set[str] = set()
        prev_cp = 0
        results = []

        store_name_map = {
            "q2q": "fake_queries",
            "q2c": "content",
            "q2p": "propositions",
            "q2n": "notes",
            "q2r": "reflections",
        }

        for cp in checkpoints:
            for sid in ordered_sids[prev_cp:cp]:
                fq = self.store.get_fake_query_embeddings(sid)
                if fq is not None:
                    pools["q2q"] = np.vstack([pools["q2q"], fq])
                    sid_arrays["q2q"].extend([sid] * len(fq))

                c = self.store.get_session_content(sid)
                if c is not None:
                    pools["q2c"] = np.vstack([pools["q2c"], c])
                    sid_arrays["q2c"].extend([sid] * len(c))

                for vkey, vname in [("q2p", "propositions"), ("q2n", "notes"), ("q2r", "reflections")]:
                    v = self.store.get_variant_embeddings(vname, sid)
                    if v is not None:
                        pools[vkey] = np.vstack([pools[vkey], v])
                        sid_arrays[vkey].extend([sid] * len(v))

                pool_sid_set.add(sid)
            prev_cp = cp

            eligible = [
                q for q in loader.queries
                if q.query_id in query_embs
                and all(s in pool_sid_set for s in q.evidence_session_ids)
            ]

            if not eligible or any(len(pools[m]) == 0 for m in ["q2q", "q2c"]):
                entry = {
                    "pool_size": cp,
                    "n_queries": 0,
                }
                for m in METHOD_KEYS:
                    entry[f"n_{m}_vectors"] = len(pools[m])
                results.append(entry)
                continue

            normed_pools = {}
            sid_arrs = {}
            for m in METHOD_KEYS:
                if len(pools[m]) > 0:
                    norms = np.linalg.norm(pools[m], axis=1, keepdims=True) + 1e-10
                    normed_pools[m] = pools[m] / norms
                    sid_arrs[m] = np.array(sid_arrays[m])
                else:
                    normed_pools[m] = None
                    sid_arrs[m] = None

            ranks = {m: [] for m in METHOD_KEYS}
            for q in eligible:
                tq = query_embs[q.query_id]
                target_sids = query_evidence[q.query_id]

                for m in METHOD_KEYS:
                    if normed_pools[m] is not None and len(normed_pools[m]) > 0:
                        r = self._compute_rank(tq, normed_pools[m], sid_arrs[m], target_sids)
                    else:
                        r = cp  # no pool yet → worst possible rank
                    ranks[m].append(r)

            entry = {
                "pool_size": cp,
                "n_queries": len(eligible),
            }
            for m in METHOD_KEYS:
                entry[f"n_{m}_vectors"] = len(pools[m])
                rarr = np.array(ranks[m], dtype=np.float64)
                entry[f"mean_rank_{m}"] = round(float(np.mean(rarr)), 2)
                entry[f"median_rank_{m}"] = round(float(np.median(rarr)), 1)
                entry[f"mrr_{m}"] = round(float(np.mean(1.0 / rarr)), 4)
                entry[f"recall_at_1_{m}"] = round(float(np.mean(rarr <= 1)), 4)
                entry[f"recall_at_5_{m}"] = round(float(np.mean(rarr <= 5)), 4)
                entry[f"recall_at_10_{m}"] = round(float(np.mean(rarr <= 10)), 4)

            results.append(entry)
            logger.info(
                f"  Checkpoint {cp}/{len(ordered_sids)}: "
                f"queries={len(eligible)}, "
                + ", ".join(f"rank_{m.upper()}={entry[f'mean_rank_{m}']}" for m in METHOD_KEYS)
            )

        return results

    @staticmethod
    def _compute_rank(
        query_emb: np.ndarray,
        pool_normed: np.ndarray,
        pool_sids: np.ndarray,
        target_sids: set[str],
    ) -> int:
        sims = pool_normed @ query_emb
        sorted_indices = np.argsort(-sims)
        for rank, idx in enumerate(sorted_indices, 1):
            if pool_sids[idx] in target_sids:
                return rank
        return len(pool_normed)

    # ------------------------------------------------------------------ #
    #  Visualization (5 lines per chart)                                 #
    # ------------------------------------------------------------------ #

    def _visualize(self, checkpoint_results: list[dict]) -> dict[str, Path]:
        fig_paths = {}
        valid = [r for r in checkpoint_results if r.get("n_queries", 0) > 0]
        if not valid:
            return fig_paths
        fig_paths["rank_vs_pool"] = self._plot_rank_vs_pool(valid)
        fig_paths["mrr_vs_pool"] = self._plot_mrr_vs_pool(valid)
        fig_paths["rank_drift"] = self._plot_rank_drift(valid)
        return fig_paths

    def _plot_rank_vs_pool(self, results: list[dict]) -> Path:
        pools = [r["pool_size"] for r in results]

        fig, ax = plt.subplots(figsize=(12, 7))
        for m in METHOD_KEYS:
            key = f"mean_rank_{m}"
            vals = [r.get(key, 0) for r in results]
            ax.plot(
                pools, vals,
                f"{METHOD_MARKERS[m]}-",
                color=METHOD_COLORS[m],
                linewidth=2, markersize=6,
                label=f"{METHOD_LABELS[m]} Mean Rank",
            )

        ax.set_xlabel("Number of Sessions in Pool", fontsize=12)
        ax.set_ylabel("Mean Rank (lower is better)", fontsize=12)
        ax.set_title("Retrieval Rank vs Memory Pool Size (5 Methods)", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = self.figs_dir / "exp3_rank_vs_pool.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Rank vs pool saved to {out_path}")
        return out_path

    def _plot_mrr_vs_pool(self, results: list[dict]) -> Path:
        pools = [r["pool_size"] for r in results]

        fig, ax = plt.subplots(figsize=(12, 7))
        for m in METHOD_KEYS:
            key = f"mrr_{m}"
            vals = [r.get(key, 0) for r in results]
            ax.plot(
                pools, vals,
                f"{METHOD_MARKERS[m]}-",
                color=METHOD_COLORS[m],
                linewidth=2, markersize=6,
                label=f"{METHOD_LABELS[m]} MRR",
            )

        ax.set_xlabel("Number of Sessions in Pool", fontsize=12)
        ax.set_ylabel("MRR (higher is better)", fontsize=12)
        ax.set_title("Mean Reciprocal Rank vs Memory Pool Size (5 Methods)", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()

        out_path = self.figs_dir / "exp3_mrr_vs_pool.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  MRR vs pool saved to {out_path}")
        return out_path

    def _plot_rank_drift(self, results: list[dict]) -> Path:
        pools = [r["pool_size"] for r in results]

        fig, ax = plt.subplots(figsize=(12, 7))
        for m in METHOD_KEYS:
            key = f"mean_rank_{m}"
            base = results[0].get(key, 0)
            drift = [r.get(key, 0) - base for r in results]
            ax.plot(
                pools, drift,
                f"{METHOD_MARKERS[m]}-",
                color=METHOD_COLORS[m],
                linewidth=2, markersize=6,
                label=f"{METHOD_LABELS[m]} Rank Drift",
            )

        ax.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Number of Sessions in Pool", fontsize=12)
        ax.set_ylabel("Rank Drift (relative to first checkpoint)", fontsize=12)
        ax.set_title("Rank Drift as Memory Pool Grows (5 Methods)", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = self.figs_dir / "exp3_rank_drift.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Rank drift saved to {out_path}")
        return out_path

    # ------------------------------------------------------------------ #
    #  Console Summary                                                   #
    # ------------------------------------------------------------------ #

    def _print_summary(self, output: dict) -> None:
        cps = output.get("checkpoints", [])
        valid = [c for c in cps if c.get("n_queries", 0) > 0]

        print("\n" + "=" * 60)
        print("  Experiment 3 Results: Temporal Drift Verification (5 Methods)")
        print("=" * 60)
        print(f"  Dataset: {output['dataset']}")
        print(f"  Sessions: {output['n_sessions']}")
        print(f"  Checkpoints: {output['n_checkpoints']}")
        print(f"  Time: {output['elapsed_seconds']}s")
        print()

        if valid:
            header = f"  {'Pool':>6}  {'#Q':>5}"
            for m in METHOD_KEYS:
                header += f"  {'Rank ' + METHOD_LABELS[m]:>10}  {'MRR ' + METHOD_LABELS[m]:>9}"
            print(header)
            print("  " + "-" * (len(header) - 2))
            for c in valid:
                line = f"  {c['pool_size']:>6}  {c['n_queries']:>5}"
                for m in METHOD_KEYS:
                    line += f"  {c.get(f'mean_rank_{m}', 0):>10.2f}  {c.get(f'mrr_{m}', 0):>9.4f}"
                print(line)

            if len(valid) >= 2:
                first, last = valid[0], valid[-1]
                print()
                print("  Total Rank Drift:")
                for m in METHOD_KEYS:
                    drift = last.get(f"mean_rank_{m}", 0) - first.get(f"mean_rank_{m}", 0)
                    print(f"    {METHOD_LABELS[m]:>4}: {drift:+.2f}")

        figs = output.get("figures", {})
        if figs:
            print()
            print("  --- Figures ---")
            for name, path in figs.items():
                print(f"    {name}: {path}")
        print("=" * 60 + "\n")
