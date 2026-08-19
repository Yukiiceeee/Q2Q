"""Experiment 1: Spatial Distribution Consistency (Q2Q vs Q2P/Q2N/Q2R/Q2C).

Analyses:
1. Per-query alignment comparison (five methods, paired t-test, Cohen's d)
2. Pairwise distance matrix between TQ, FQ, C, P, N, R embedding spaces
3. Cluster analysis (KMeans) + ARI / NMI / Silhouette
4. t-SNE visualization (6 types) + Gap distribution + Method comparison bar chart
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from experiment.src.experiment_base import ExperimentBase
from experiment.src.metrics import AlignmentMetrics, AlignmentResult

logger = logging.getLogger(__name__)

VARIANT_KEYS = ["q2p", "q2n", "q2r"]
VARIANT_STORE_NAMES = ["propositions", "notes", "reflections"]

METHOD_COLORS = {
    "Q2Q": "#2196F3",
    "Q2P": "#9C27B0",
    "Q2N": "#E91E63",
    "Q2R": "#009688",
    "Q2C": "#FF9800",
}

EMBED_TYPES = {
    "TQ": {"label": "True Query (TQ)", "color": "#2196F3", "size": 40, "alpha": 0.9, "id": 0},
    "FQ": {"label": "Fake Query (FQ)", "color": "#FF9800", "size": 12, "alpha": 0.4, "id": 1},
    "C": {"label": "Content Chunk (C)", "color": "#4CAF50", "size": 12, "alpha": 0.4, "id": 2},
    "P": {"label": "Proposition (P)", "color": "#9C27B0", "size": 12, "alpha": 0.4, "id": 3},
    "N": {"label": "Note (N)", "color": "#E91E63", "size": 12, "alpha": 0.4, "id": 4},
    "R": {"label": "Reflection (R)", "color": "#009688", "size": 12, "alpha": 0.4, "id": 5},
}


class Exp1Alignment(ExperimentBase):

    async def run(self) -> dict:
        logger.info("=" * 60)
        logger.info("Experiment 1: Spatial Distribution Consistency")
        logger.info("=" * 60)

        start_time = time.time()
        loader = self.load_data()

        self.exp_logger.start_phase("embeddings")
        await self._ensure_embeddings(loader)
        self.exp_logger.end_phase("embeddings")

        self.exp_logger.start_phase("alignment")
        results = await self._compute_alignments(loader)
        self.exp_logger.end_phase("alignment", {"n_queries": len(results)})

        self.exp_logger.start_phase("statistics")
        overall_stats = AlignmentMetrics.paired_statistics(results)
        category_stats = AlignmentMetrics.category_breakdown(results, loader.queries)
        multi_stats = AlignmentMetrics.multi_paired_statistics({
            "q2q": [r.sim_q2q for r in results],
            "q2p": [r.sim_q2p for r in results],
            "q2n": [r.sim_q2n for r in results],
            "q2r": [r.sim_q2r for r in results],
            "q2c": [r.sim_q2c for r in results],
        })
        self.exp_logger.end_phase("statistics")

        self.exp_logger.start_phase("distance_matrix")
        dist_stats = self._compute_distance_matrix(loader)
        self.exp_logger.end_phase("distance_matrix")

        self.exp_logger.start_phase("cluster_analysis")
        cluster_stats = self._cluster_analysis(loader, n_samples=200)
        self.exp_logger.end_phase("cluster_analysis")

        self.exp_logger.start_phase("visualization")
        tsne_path = self._visualize_tsne(loader, n_samples=150)
        gap_path = self._plot_gap_distribution(results)
        bar_path = self._plot_method_comparison(results)
        self.exp_logger.end_phase("visualization")

        elapsed = time.time() - start_time
        output = {
            "experiment": "exp1_alignment",
            "dataset": self.dataset_config["dataset"]["name"],
            "n_queries": len(results),
            "elapsed_seconds": round(elapsed, 2),
            "overall": overall_stats,
            "multi_method_stats": multi_stats,
            "by_category": category_stats,
            "distance_matrix": dist_stats,
            "cluster_analysis": cluster_stats,
            "figures": {
                "tsne": str(tsne_path),
                "gap_distribution": str(gap_path),
                "method_comparison": str(bar_path),
            },
            "sample_results": [
                {
                    "query_id": r.query_id,
                    "sim_q2q": round(r.sim_q2q, 4),
                    "sim_q2p": round(r.sim_q2p, 4),
                    "sim_q2n": round(r.sim_q2n, 4),
                    "sim_q2r": round(r.sim_q2r, 4),
                    "sim_q2c": round(r.sim_q2c, 4),
                    "gap": round(r.gap, 4),
                    "best_fq_text": r.best_fq_text,
                }
                for r in results[:20]
            ],
        }

        self.save_results(output, "exp1_alignment.json")
        self.save_experiment_log(output, "exp1_alignment_log.json")
        self._print_summary(output)
        return output

    async def _ensure_embeddings(self, loader) -> None:
        logger.info("Phase 1: Checking/computing embeddings...")

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
                if (i + 1) % 50 == 0:
                    logger.info(f"    Session embeddings: {i + 1}/{len(sessions_to_embed)}")
            logger.info(f"  Done: {len(sessions_to_embed)} sessions embedded.")

        fq_missing = [
            sid for sid in answer_sids
            if not self.store.has_fake_queries(sid)
        ]
        if fq_missing:
            logger.warning(
                f"  {len(fq_missing)} sessions missing fake query embeddings. "
                f"Run generation step first (--step generate)."
            )

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
        logger.info("Phase 2: Computing alignment scores (5 methods)...")
        results = []
        skipped = 0

        for q in loader.queries:
            query_emb = self.store.get_true_query_embedding(q.query_id)
            if query_emb is None:
                skipped += 1
                continue

            best_result = None
            for sid in q.evidence_session_ids:
                fq_embs = self.store.get_fake_query_embeddings(sid)
                content_embs = self.store.get_session_content(sid)
                if fq_embs is None or content_embs is None:
                    continue

                sim_q2q, fq_idx, sim_q2c, chunk_idx = AlignmentMetrics.compute_alignment(
                    query_emb, fq_embs, content_embs
                )

                sim_q2p, _ = AlignmentMetrics.compute_max_sim(
                    query_emb,
                    self.store.get_variant_embeddings("propositions", sid),
                )
                sim_q2n, _ = AlignmentMetrics.compute_max_sim(
                    query_emb,
                    self.store.get_variant_embeddings("notes", sid),
                )
                sim_q2r, _ = AlignmentMetrics.compute_max_sim(
                    query_emb,
                    self.store.get_variant_embeddings("reflections", sid),
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
                        sim_q2p=sim_q2p,
                        sim_q2n=sim_q2n,
                        sim_q2r=sim_q2r,
                    )

            if best_result:
                results.append(best_result)

        if skipped:
            logger.warning(f"  Skipped {skipped} queries (missing embeddings)")
        logger.info(f"  Computed alignment for {len(results)} queries.")
        return results

    # ------------------------------------------------------------------ #
    #  Distance Matrix: pairwise avg cosine distances TQ/FQ/C/P/N/R      #
    # ------------------------------------------------------------------ #

    def _compute_distance_matrix(self, loader) -> dict:
        logger.info("Phase 4: Computing pairwise distance matrix (6 types)...")

        type_names = ["TQ", "FQ", "C", "P", "N", "R"]
        pair_dists = {f"{a}_{b}": [] for i, a in enumerate(type_names) for b in type_names[i + 1:]}

        for q in loader.queries:
            tq_emb = self.store.get_true_query_embedding(q.query_id)
            if tq_emb is None:
                continue

            for sid in q.evidence_session_ids:
                fq_embs = self.store.get_fake_query_embeddings(sid)
                c_embs = self.store.get_session_content(sid)
                if fq_embs is None or c_embs is None:
                    continue

                p_embs = self.store.get_variant_embeddings("propositions", sid)
                n_embs = self.store.get_variant_embeddings("notes", sid)
                r_embs = self.store.get_variant_embeddings("reflections", sid)

                tq = tq_emb / (np.linalg.norm(tq_emb) + 1e-10)

                def _norm_and_centroid(embs):
                    if embs is None or len(embs) == 0:
                        return None, None
                    normed = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
                    cent = normed.mean(axis=0)
                    cent = cent / (np.linalg.norm(cent) + 1e-10)
                    return normed, cent

                fq_norm, fq_cent = _norm_and_centroid(fq_embs)
                c_norm, c_cent = _norm_and_centroid(c_embs)
                p_norm, p_cent = _norm_and_centroid(p_embs)
                n_norm, n_cent = _norm_and_centroid(n_embs)
                r_norm, r_cent = _norm_and_centroid(r_embs)

                centroids = {"FQ": fq_cent, "C": c_cent, "P": p_cent, "N": n_cent, "R": r_cent}
                normed_arrays = {"FQ": fq_norm, "C": c_norm, "P": p_norm, "N": n_norm, "R": r_norm}

                for name, narr in normed_arrays.items():
                    if narr is not None:
                        pair_dists[f"TQ_{name}"].append(1.0 - float(np.mean(narr @ tq)))

                cent_names = [k for k in centroids if centroids[k] is not None]
                for i, a in enumerate(cent_names):
                    for b in cent_names[i + 1:]:
                        key = f"{a}_{b}"
                        if key in pair_dists:
                            pair_dists[key].append(
                                1.0 - float(centroids[a] @ centroids[b])
                            )

        def _stats(arr):
            if not arr:
                return {"mean": 0.0, "std": 0.0, "median": 0.0}
            a = np.array(arr)
            return {"mean": float(np.mean(a)), "std": float(np.std(a)), "median": float(np.median(a))}

        result = {f"dist_{k}": _stats(v) for k, v in pair_dists.items() if v}
        result["n_pairs"] = len(pair_dists.get("TQ_FQ", []))

        for k, v in result.items():
            if k.startswith("dist_") and isinstance(v, dict):
                label = k.replace("dist_", "").replace("_", " ↔ ")
                logger.info(f"  {label}: mean={v.get('mean', 0):.4f}")
        return result

    # ------------------------------------------------------------------ #
    #  Cluster Analysis: KMeans on TQ/FQ/C/P/N/R embeddings              #
    # ------------------------------------------------------------------ #

    def _cluster_analysis(self, loader, n_samples: int = 200) -> dict:
        logger.info("Phase 5: Cluster analysis (6 types)...")
        seed = self.base_config.get("experiment", {}).get("seed", 42)
        rng = np.random.RandomState(seed)

        sampled_queries = rng.choice(
            len(loader.queries), size=min(n_samples, len(loader.queries)), replace=False
        )

        all_vecs, all_labels = [], []
        for idx in sampled_queries:
            q = loader.queries[idx]
            tq_emb = self.store.get_true_query_embedding(q.query_id)
            if tq_emb is None:
                continue
            sid = q.evidence_session_ids[0]
            fq_embs = self.store.get_fake_query_embeddings(sid)
            c_embs = self.store.get_session_content(sid)
            if fq_embs is None or c_embs is None:
                continue

            all_vecs.append(tq_emb)
            all_labels.append(0)
            for v in fq_embs:
                all_vecs.append(v)
                all_labels.append(1)
            for v in c_embs:
                all_vecs.append(v)
                all_labels.append(2)

            for label_id, store_name in [(3, "propositions"), (4, "notes"), (5, "reflections")]:
                v_embs = self.store.get_variant_embeddings(store_name, sid)
                if v_embs is not None:
                    for v in v_embs:
                        all_vecs.append(v)
                        all_labels.append(label_id)

        X = np.array(all_vecs, dtype=np.float32)
        y_true = np.array(all_labels)

        n_types = len(set(y_true))
        type_counts = {name: int(np.sum(y_true == cfg["id"])) for name, cfg in EMBED_TYPES.items()}
        logger.info(f"  Collected {len(X)} vectors: {type_counts}")

        km = KMeans(n_clusters=n_types, random_state=seed, n_init=10)
        y_pred = km.fit_predict(X)

        ari = float(adjusted_rand_score(y_true, y_pred))
        nmi = float(normalized_mutual_info_score(y_true, y_pred))
        sil = float(silhouette_score(X, y_pred, sample_size=min(5000, len(X)), random_state=seed))

        result = {
            "n_vectors": len(X),
            "n_clusters": n_types,
            **{f"n_{name.lower()}": count for name, count in type_counts.items()},
            "adjusted_rand_index": round(ari, 4),
            "normalized_mutual_info": round(nmi, 4),
            "silhouette_score": round(sil, 4),
        }
        logger.info(f"  Cluster: ARI={ari:.4f}, NMI={nmi:.4f}, Silhouette={sil:.4f}")
        return result

    # ------------------------------------------------------------------ #
    #  t-SNE Visualization (6 types)                                     #
    # ------------------------------------------------------------------ #

    def _visualize_tsne(self, loader, n_samples: int = 150) -> Path:
        logger.info("Phase 6a: t-SNE visualization (6 types)...")
        seed = self.base_config.get("experiment", {}).get("seed", 42)
        rng = np.random.RandomState(seed)

        sampled_queries = rng.choice(
            len(loader.queries), size=min(n_samples, len(loader.queries)), replace=False
        )

        all_vecs, all_labels = [], []
        for idx in sampled_queries:
            q = loader.queries[idx]
            tq_emb = self.store.get_true_query_embedding(q.query_id)
            if tq_emb is None:
                continue
            sid = q.evidence_session_ids[0]
            fq_embs = self.store.get_fake_query_embeddings(sid)
            c_embs = self.store.get_session_content(sid)
            if fq_embs is None or c_embs is None:
                continue

            all_vecs.append(tq_emb)
            all_labels.append(0)
            for v in fq_embs:
                all_vecs.append(v)
                all_labels.append(1)
            for v in c_embs:
                all_vecs.append(v)
                all_labels.append(2)
            for label_id, store_name in [(3, "propositions"), (4, "notes"), (5, "reflections")]:
                v_embs = self.store.get_variant_embeddings(store_name, sid)
                if v_embs is not None:
                    for v in v_embs:
                        all_vecs.append(v)
                        all_labels.append(label_id)

        X = np.array(all_vecs, dtype=np.float32)
        labels = np.array(all_labels)

        tsne = TSNE(n_components=2, perplexity=30, random_state=seed, max_iter=1000)
        X_2d = tsne.fit_transform(X)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        for name, cfg in reversed(list(EMBED_TYPES.items())):
            mask = labels == cfg["id"]
            if not np.any(mask):
                continue
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=cfg["color"], label=cfg["label"],
                s=cfg["size"], alpha=cfg["alpha"], edgecolors="none",
            )

        ax.set_title("Embedding Space: TQ / FQ / C / P / N / R (t-SNE)", fontsize=14)
        ax.legend(fontsize=10, loc="best")
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        plt.tight_layout()

        out_path = self.figs_dir / "exp1_tsne.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  t-SNE saved to {out_path}")
        return out_path

    # ------------------------------------------------------------------ #
    #  Gap Distribution (4 overlaid histograms)                          #
    # ------------------------------------------------------------------ #

    def _plot_gap_distribution(self, results: list[AlignmentResult]) -> Path:
        logger.info("Phase 6b: Gap distribution visualization (4 gaps)...")
        dataset_name = self.dataset_config["dataset"]["name"]

        gaps = {
            "Q2Q - Q2C": ([r.sim_q2q - r.sim_q2c for r in results], METHOD_COLORS["Q2C"]),
            "Q2Q - Q2P": ([r.sim_q2q - r.sim_q2p for r in results], METHOD_COLORS["Q2P"]),
            "Q2Q - Q2N": ([r.sim_q2q - r.sim_q2n for r in results], METHOD_COLORS["Q2N"]),
            "Q2Q - Q2R": ([r.sim_q2q - r.sim_q2r for r in results], METHOD_COLORS["Q2R"]),
        }

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        for label, (vals, color) in gaps.items():
            arr = np.array(vals)
            ax.hist(
                arr, bins=50, color=color, alpha=0.45, density=True,
                label=f"{label} (mean={np.mean(arr):.4f})", edgecolor="white",
            )

        ax.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        ax.set_xlabel("Gap (sim_Q2Q - sim_baseline)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Distribution of Q2Q Advantage over Baselines ({dataset_name})", fontsize=14)
        ax.legend(fontsize=10)
        plt.tight_layout()

        out_path = self.figs_dir / "exp1_gap_distribution.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Gap distribution saved to {out_path}")
        return out_path

    # ------------------------------------------------------------------ #
    #  Method Comparison Bar Chart                                       #
    # ------------------------------------------------------------------ #

    def _plot_method_comparison(self, results: list[AlignmentResult]) -> Path:
        logger.info("Phase 6c: Method comparison bar chart...")
        dataset_name = self.dataset_config["dataset"]["name"]

        methods = ["Q2Q", "Q2P", "Q2N", "Q2R", "Q2C"]
        attr_keys = ["sim_q2q", "sim_q2p", "sim_q2n", "sim_q2r", "sim_q2c"]
        means, stds = [], []
        for key in attr_keys:
            vals = np.array([getattr(r, key) for r in results])
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        x = np.arange(len(methods))
        colors = [METHOD_COLORS[m] for m in methods]
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="white")

        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=10,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=12)
        ax.set_ylabel("Mean Similarity", fontsize=12)
        ax.set_title(f"Five-Method Alignment Comparison ({dataset_name})", fontsize=14)
        ax.set_ylim(0, max(means) + 0.15)
        plt.tight_layout()

        out_path = self.figs_dir / "exp1_method_comparison.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Method comparison saved to {out_path}")
        return out_path

    # ------------------------------------------------------------------ #
    #  Console Summary                                                   #
    # ------------------------------------------------------------------ #

    def _print_summary(self, output: dict) -> None:
        overall = output.get("overall", {})
        multi = output.get("multi_method_stats", {})
        dist = output.get("distance_matrix", {})
        cluster = output.get("cluster_analysis", {})

        print("\n" + "=" * 60)
        print("  Experiment 1 Results: Spatial Distribution Consistency")
        print("=" * 60)
        print(f"  Dataset: {output['dataset']}")
        print(f"  Queries analyzed: {output['n_queries']}")
        print(f"  Time: {output['elapsed_seconds']}s")
        print()

        print("  --- Five-Method Similarity ---")
        for method in ["q2q", "q2p", "q2n", "q2r", "q2c"]:
            info = multi.get(method, {})
            print(f"    {method.upper():>4}: mean={info.get('mean', 0):.4f} (std={info.get('std', 0):.4f})")
        print()

        print(f"  Gap (Q2Q-Q2C):   mean={overall.get('gap_mean', 0):.4f}")
        print(f"  Paired t-test:   t={overall.get('t_statistic', 0):.4f}, p={overall.get('p_value', 1):.2e}")
        print(f"  Cohen's d:       {overall.get('cohens_d', 0):.4f}")
        print(f"  Q2Q win rate:    {overall.get('q2q_win_rate', 0):.1%}")
        print()

        pairwise = multi.get("pairwise", {})
        if pairwise:
            print("  --- Pairwise Comparisons ---")
            for pair, info in sorted(pairwise.items()):
                label = pair.replace("_vs_", " vs ").upper()
                print(f"    {label}: d={info.get('cohens_d', 0):+.4f}, p={info.get('p_value', 1):.2e}")
            print()

        if cluster:
            print("  --- Cluster Analysis ---")
            print(f"    Vectors: {cluster.get('n_vectors', 0)}, Clusters: {cluster.get('n_clusters', 0)}")
            print(f"    ARI={cluster.get('adjusted_rand_index', 0):.4f}, "
                  f"NMI={cluster.get('normalized_mutual_info', 0):.4f}, "
                  f"Silhouette={cluster.get('silhouette_score', 0):.4f}")
            print()

        figs = output.get("figures", {})
        if figs:
            print("  --- Figures ---")
            for name, path in figs.items():
                print(f"    {name}: {path}")

        print("=" * 60 + "\n")
