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
        tsne_paths = self._visualize_tsne(loader, n_samples=150)
        gap_path = self._plot_gap_distribution(results)
        bar_path = self._plot_method_comparison(results)
        self.exp_logger.end_phase("visualization")

        self.exp_logger.start_phase("directional_alignment")
        dir_stats = self._compute_directional_alignment(loader)
        dir_fig = self._plot_directional_alignment(dir_stats)
        self.exp_logger.end_phase("directional_alignment")

        self.exp_logger.start_phase("discriminability")
        disc_stats = self._compute_discriminability(loader)
        disc_fig = self._plot_discriminability(disc_stats)
        self.exp_logger.end_phase("discriminability")

        self.exp_logger.start_phase("precision_at_k")
        prec_stats = self._compute_precision_at_k(loader)
        prec_fig = self._plot_precision_at_k(prec_stats)
        self.exp_logger.end_phase("precision_at_k")

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
            "directional_alignment": dir_stats,
            "discriminability": disc_stats,
            "precision_at_k": prec_stats,
            "figures": {
                "tsne_all": str(tsne_paths.get("tsne_all", "")),
                "tsne_nearest": str(tsne_paths.get("tsne_nearest", "")),
                "gap_distribution": str(gap_path),
                "method_comparison": str(bar_path),
                "directional_alignment": str(dir_fig),
                "discriminability": str(disc_fig),
                "precision_at_k": str(prec_fig),
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
    #  t-SNE Visualization (2 figures × 4 subplots)                     #
    # ------------------------------------------------------------------ #

    def _visualize_tsne(self, loader, n_samples: int = 150) -> dict[str, Path]:
        logger.info("Phase 6a: t-SNE visualization (2 figures × 4 subplots)...")
        seed = self.base_config.get("experiment", {}).get("seed", 42)
        rng = np.random.RandomState(seed)

        sampled_queries = rng.choice(
            len(loader.queries), size=min(n_samples, len(loader.queries)), replace=False
        )

        # label: 0=TQ, 1=FQ, 2=Content, 3=Proposition, 4=Note, 5=Reflection
        all_vecs, all_labels, query_group = [], [], []
        qidx = 0
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
            query_group.append(qidx)

            for v in fq_embs:
                all_vecs.append(v)
                all_labels.append(1)
                query_group.append(qidx)

            for v in c_embs:
                all_vecs.append(v)
                all_labels.append(2)
                query_group.append(qidx)

            for label_id, store_name in [(3, "propositions"), (4, "notes"), (5, "reflections")]:
                v_embs = self.store.get_variant_embeddings(store_name, sid)
                if v_embs is not None:
                    for v in v_embs:
                        all_vecs.append(v)
                        all_labels.append(label_id)
                        query_group.append(qidx)

            qidx += 1

        X = np.array(all_vecs, dtype=np.float32)
        labels = np.array(all_labels)
        groups = np.array(query_group)

        tsne = TSNE(n_components=2, perplexity=30, random_state=seed, max_iter=1000)
        X_2d = tsne.fit_transform(X)

        tq_indices = np.where(labels == 0)[0]
        nearest_masks = {}
        for baseline_label in [1, 2, 3, 4, 5]:
            keep = set()
            for tq_idx in tq_indices:
                grp = groups[tq_idx]
                candidates = np.where((labels == baseline_label) & (groups == grp))[0]
                if len(candidates) == 0:
                    continue
                dists = np.linalg.norm(X_2d[candidates] - X_2d[tq_idx], axis=1)
                keep.add(candidates[np.argmin(dists)])
            nearest_masks[baseline_label] = keep

        subplot_cfgs = [
            ("TQ vs FQ vs Content", [(1, "#2196F3", "Fake Query"), (2, "#FF9800", "Content")]),
            ("TQ vs FQ vs Proposition", [(1, "#2196F3", "Fake Query"), (3, "#9C27B0", "Proposition")]),
            ("TQ vs FQ vs Note", [(1, "#2196F3", "Fake Query"), (4, "#E91E63", "Note")]),
            ("TQ vs FQ vs Reflection", [(1, "#2196F3", "Fake Query"), (5, "#009688", "Reflection")]),
        ]

        paths = {}

        # --- Figure 1: All points ---
        fig1, axes1 = plt.subplots(1, 4, figsize=(28, 6))
        for i, (title, baselines) in enumerate(subplot_cfgs):
            ax = axes1[i]
            for bl_label, bl_color, bl_name in reversed(baselines):
                mask = labels == bl_label
                if np.any(mask):
                    ax.scatter(
                        X_2d[mask, 0], X_2d[mask, 1],
                        c=bl_color, s=8, alpha=0.3, edgecolors="none",
                        label=bl_name, zorder=2,
                    )
            tq_mask = labels == 0
            ax.scatter(
                X_2d[tq_mask, 0], X_2d[tq_mask, 1],
                c="#E53935", s=28, alpha=0.85, edgecolors="white",
                linewidths=0.3, label="True Query", zorder=3,
            )
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.legend(fontsize=9, loc="upper right", framealpha=0.8)
            ax.set_xticks([])
            ax.set_yticks([])

        fig1.suptitle("Embedding Space — All Points (t-SNE)", fontsize=15, y=1.02)
        fig1.tight_layout()
        p1 = self.figs_dir / "exp1_tsne_all.png"
        fig1.savefig(p1, dpi=300, bbox_inches="tight")
        plt.close(fig1)
        paths["tsne_all"] = p1
        logger.info(f"  t-SNE (all) saved to {p1}")

        # --- Figure 2: Nearest neighbor only ---
        fig2, axes2 = plt.subplots(1, 4, figsize=(28, 6))
        for i, (title, baselines) in enumerate(subplot_cfgs):
            ax = axes2[i]
            for bl_label, bl_color, bl_name in reversed(baselines):
                keep_set = nearest_masks[bl_label]
                keep_idx = np.array(sorted(keep_set)) if keep_set else np.array([], dtype=int)
                if len(keep_idx) > 0:
                    ax.scatter(
                        X_2d[keep_idx, 0], X_2d[keep_idx, 1],
                        c=bl_color, s=18, alpha=0.65, edgecolors="none",
                        label=bl_name, zorder=2,
                    )
            tq_mask = labels == 0
            ax.scatter(
                X_2d[tq_mask, 0], X_2d[tq_mask, 1],
                c="#E53935", s=28, alpha=0.85, edgecolors="white",
                linewidths=0.3, label="True Query", zorder=3,
            )
            for tq_idx in tq_indices:
                for bl_label, bl_color, _ in baselines:
                    grp = groups[tq_idx]
                    candidates = np.where((labels == bl_label) & (groups == grp))[0]
                    if len(candidates) == 0:
                        continue
                    dists = np.linalg.norm(X_2d[candidates] - X_2d[tq_idx], axis=1)
                    nn_idx = candidates[np.argmin(dists)]
                    ax.plot(
                        [X_2d[tq_idx, 0], X_2d[nn_idx, 0]],
                        [X_2d[tq_idx, 1], X_2d[nn_idx, 1]],
                        color=bl_color, linewidth=0.4, alpha=0.25, zorder=1,
                    )
            ax.set_title(title + " (nearest)", fontsize=13, fontweight="bold")
            ax.legend(fontsize=9, loc="upper right", framealpha=0.8)
            ax.set_xticks([])
            ax.set_yticks([])

        fig2.suptitle("Embedding Space — Nearest Neighbor per Query (t-SNE)", fontsize=15, y=1.02)
        fig2.tight_layout()
        p2 = self.figs_dir / "exp1_tsne_nearest.png"
        fig2.savefig(p2, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        paths["tsne_nearest"] = p2
        logger.info(f"  t-SNE (nearest) saved to {p2}")

        return paths

    # ------------------------------------------------------------------ #
    #  Gap Distribution (4 overlaid histograms)                          #
    # ------------------------------------------------------------------ #

    def _plot_gap_distribution(self, results: list[AlignmentResult]) -> Path:
        logger.info("Phase 6b: Gap distribution visualization (violin + strip)...")
        dataset_name = self.dataset_config["dataset"]["name"]

        gap_labels = ["Q2Q−Q2C", "Q2Q−Q2P", "Q2Q−Q2N", "Q2Q−Q2R"]
        gap_colors = ["#FF9800", "#9C27B0", "#E91E63", "#009688"]
        gap_data = [
            np.array([r.sim_q2q - r.sim_q2c for r in results]),
            np.array([r.sim_q2q - r.sim_q2p for r in results]),
            np.array([r.sim_q2q - r.sim_q2n for r in results]),
            np.array([r.sim_q2q - r.sim_q2r for r in results]),
        ]

        fig, ax = plt.subplots(figsize=(10, 6))

        parts = ax.violinplot(
            gap_data, positions=range(len(gap_labels)),
            showmeans=False, showextrema=False, showmedians=False,
        )
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(gap_colors[i])
            body.set_alpha(0.35)
            body.set_edgecolor(gap_colors[i])

        bp = ax.boxplot(
            gap_data, positions=range(len(gap_labels)),
            widths=0.15, patch_artist=True, showfliers=False,
            medianprops={"color": "white", "linewidth": 1.5},
            whiskerprops={"color": "#555", "linewidth": 0.8},
            capprops={"color": "#555", "linewidth": 0.8},
        )
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(gap_colors[i])
            patch.set_alpha(0.85)

        for i, d in enumerate(gap_data):
            mean_v = np.mean(d)
            ax.text(
                i, ax.get_ylim()[1] * 0.95,
                f"μ={mean_v:.4f}\nwin={np.mean(d > 0):.0%}",
                ha="center", va="top", fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 2},
            )

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xticks(range(len(gap_labels)))
        ax.set_xticklabels(gap_labels, fontsize=11)
        ax.set_ylabel("Similarity Gap", fontsize=12)
        ax.set_title(f"Q2Q Advantage over Baselines ({dataset_name})", fontsize=14)
        ax.grid(axis="y", alpha=0.2)
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
    #  Directional Alignment Analysis                                    #
    # ------------------------------------------------------------------ #

    def _compute_directional_alignment(self, loader) -> dict:
        logger.info("Phase 7: Directional alignment analysis...")
        method_dirs = {m: [] for m in ["q2q", "q2p", "q2n", "q2r", "q2c"]}
        store_map = {"q2q": "fq", "q2c": "content", "q2p": "propositions", "q2n": "notes", "q2r": "reflections"}

        for q in loader.queries:
            query_emb = self.store.get_true_query_embedding(q.query_id)
            if query_emb is None:
                continue

            for sid in q.evidence_session_ids:
                c_embs = self.store.get_session_content(sid)
                if c_embs is None or len(c_embs) == 0:
                    continue
                c_norm = c_embs / (np.linalg.norm(c_embs, axis=1, keepdims=True) + 1e-10)
                centroid = c_norm.mean(axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-10)

                embs_map = {
                    "q2q": self.store.get_fake_query_embeddings(sid),
                    "q2c": c_embs,
                    "q2p": self.store.get_variant_embeddings("propositions", sid),
                    "q2n": self.store.get_variant_embeddings("notes", sid),
                    "q2r": self.store.get_variant_embeddings("reflections", sid),
                }

                for m, embs in embs_map.items():
                    da = AlignmentMetrics.compute_directional_alignment(
                        query_emb, centroid, embs,
                    )
                    method_dirs[m].append(da)

        result = {}
        for m in method_dirs:
            arr = np.array(method_dirs[m]) if method_dirs[m] else np.array([0.0])
            result[m] = {
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr)), 4),
                "median": round(float(np.median(arr)), 4),
            }
        logger.info(
            "  Directional alignment: "
            + ", ".join(f"{m.upper()}={result[m]['mean']:.4f}" for m in method_dirs)
        )
        return result

    # ------------------------------------------------------------------ #
    #  Semantic Discriminability Analysis                                #
    # ------------------------------------------------------------------ #

    def _compute_discriminability(self, loader) -> dict:
        logger.info("Phase 8: Semantic discriminability analysis...")
        seed = self.base_config.get("experiment", {}).get("seed", 42)
        rng = np.random.RandomState(seed)

        answer_sids = set()
        for q in loader.queries:
            answer_sids.update(q.evidence_session_ids)
        all_sids = list(answer_sids)

        method_disc = {m: [] for m in ["q2q", "q2p", "q2n", "q2r", "q2c"]}

        for q in loader.queries:
            query_emb = self.store.get_true_query_embedding(q.query_id)
            if query_emb is None:
                continue
            target_sids = set(q.evidence_session_ids)
            neg_candidates = [s for s in all_sids if s not in target_sids]
            if not neg_candidates:
                continue
            neg_sid = rng.choice(neg_candidates)

            for sid in q.evidence_session_ids:
                pos_map = {
                    "q2q": self.store.get_fake_query_embeddings(sid),
                    "q2c": self.store.get_session_content(sid),
                    "q2p": self.store.get_variant_embeddings("propositions", sid),
                    "q2n": self.store.get_variant_embeddings("notes", sid),
                    "q2r": self.store.get_variant_embeddings("reflections", sid),
                }
                neg_map = {
                    "q2q": self.store.get_fake_query_embeddings(neg_sid),
                    "q2c": self.store.get_session_content(neg_sid),
                    "q2p": self.store.get_variant_embeddings("propositions", neg_sid),
                    "q2n": self.store.get_variant_embeddings("notes", neg_sid),
                    "q2r": self.store.get_variant_embeddings("reflections", neg_sid),
                }

                for m in method_disc:
                    if pos_map[m] is not None and neg_map[m] is not None:
                        disc = AlignmentMetrics.compute_discriminability(
                            query_emb, pos_map[m], neg_map[m],
                        )
                        method_disc[m].append(disc)

        result = {}
        for m in method_disc:
            arr = np.array(method_disc[m]) if method_disc[m] else np.array([0.0])
            result[m] = {
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr)), 4),
                "positive_rate": round(float(np.mean(arr > 0)), 4),
            }
        logger.info(
            "  Discriminability: "
            + ", ".join(f"{m.upper()}={result[m]['mean']:.4f}" for m in method_disc)
        )
        return result

    # ------------------------------------------------------------------ #
    #  Retrieval Precision@K Analysis                                    #
    # ------------------------------------------------------------------ #

    def _compute_precision_at_k(self, loader, k_values: list[int] | None = None) -> dict:
        if k_values is None:
            k_values = [1, 5, 10]
        logger.info(f"Phase 9: Precision@K analysis (K={k_values})...")

        answer_sids = set()
        for q in loader.queries:
            answer_sids.update(q.evidence_session_ids)
        all_sids = list(answer_sids)

        pools = {m: [] for m in ["q2q", "q2p", "q2n", "q2r", "q2c"]}
        pool_sids = {m: [] for m in pools}
        store_getters = {
            "q2q": self.store.get_fake_query_embeddings,
            "q2c": self.store.get_session_content,
            "q2p": lambda sid: self.store.get_variant_embeddings("propositions", sid),
            "q2n": lambda sid: self.store.get_variant_embeddings("notes", sid),
            "q2r": lambda sid: self.store.get_variant_embeddings("reflections", sid),
        }

        for sid in all_sids:
            for m, getter in store_getters.items():
                embs = getter(sid)
                if embs is not None and len(embs) > 0:
                    pools[m].append(embs)
                    pool_sids[m].extend([sid] * len(embs))

        merged_pools = {}
        merged_sids = {}
        for m in pools:
            if pools[m]:
                merged_pools[m] = np.vstack(pools[m])
                norms = np.linalg.norm(merged_pools[m], axis=1, keepdims=True) + 1e-10
                merged_pools[m] = merged_pools[m] / norms
                merged_sids[m] = np.array(pool_sids[m])
            else:
                merged_pools[m] = None
                merged_sids[m] = None

        precision = {m: {k: [] for k in k_values} for m in pools}

        for q in loader.queries:
            query_emb = self.store.get_true_query_embedding(q.query_id)
            if query_emb is None:
                continue
            q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
            target_sids = set(q.evidence_session_ids)

            for m in pools:
                if merged_pools[m] is None:
                    continue
                sims = merged_pools[m] @ q_norm
                top_indices = np.argsort(-sims)
                seen_sids = set()
                ranked_sids = []
                for idx in top_indices:
                    sid = merged_sids[m][idx]
                    if sid not in seen_sids:
                        seen_sids.add(sid)
                        ranked_sids.append(sid)
                    if len(ranked_sids) >= max(k_values):
                        break

                for k in k_values:
                    top_k = ranked_sids[:k]
                    hit = any(s in target_sids for s in top_k)
                    precision[m][k].append(1.0 if hit else 0.0)

        result = {}
        for m in pools:
            result[m] = {}
            for k in k_values:
                arr = precision[m][k]
                result[m][f"P@{k}"] = round(float(np.mean(arr)), 4) if arr else 0.0
        logger.info(
            "  P@1: " + ", ".join(f"{m.upper()}={result[m].get('P@1', 0):.4f}" for m in pools)
        )
        return result

    # ------------------------------------------------------------------ #
    #  Deep Analysis Visualizations                                      #
    # ------------------------------------------------------------------ #

    def _plot_directional_alignment(self, dir_results: dict) -> Path:
        logger.info("Phase 10a: Directional alignment visualization...")
        dataset_name = self.dataset_config["dataset"]["name"]

        methods = ["Q2Q", "Q2P", "Q2N", "Q2R", "Q2C"]
        method_keys = ["q2q", "q2p", "q2n", "q2r", "q2c"]
        means = [dir_results[m]["mean"] for m in method_keys]
        stds = [dir_results[m]["std"] for m in method_keys]
        colors = [METHOD_COLORS[m] for m in methods]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="white")
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=10,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=12)
        ax.set_ylabel("Directional Alignment (cosine)", fontsize=12)
        ax.set_title(f"Intent Direction Consistency ({dataset_name})", fontsize=14)
        ax.grid(axis="y", alpha=0.2)
        plt.tight_layout()

        out_path = self.figs_dir / "exp1_directional_alignment.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Directional alignment saved to {out_path}")
        return out_path

    def _plot_discriminability(self, disc_results: dict) -> Path:
        logger.info("Phase 10b: Discriminability visualization...")
        dataset_name = self.dataset_config["dataset"]["name"]

        methods = ["Q2Q", "Q2P", "Q2N", "Q2R", "Q2C"]
        method_keys = ["q2q", "q2p", "q2n", "q2r", "q2c"]
        colors = [METHOD_COLORS[m] for m in methods]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        means = [disc_results[m]["mean"] for m in method_keys]
        stds = [disc_results[m]["std"] for m in method_keys]
        x = np.arange(len(methods))
        bars = ax1.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="white")
        for bar, mean in zip(bars, means):
            ax1.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=9,
            )
        ax1.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, fontsize=11)
        ax1.set_ylabel("Mean Discriminability", fontsize=12)
        ax1.set_title("Signal − Noise Gap", fontsize=13)
        ax1.grid(axis="y", alpha=0.2)

        pos_rates = [disc_results[m]["positive_rate"] for m in method_keys]
        bars2 = ax2.bar(x, pos_rates, color=colors, alpha=0.85, edgecolor="white")
        for bar, rate in zip(bars2, pos_rates):
            ax2.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{rate:.1%}", ha="center", va="bottom", fontsize=9,
            )
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, fontsize=11)
        ax2.set_ylabel("Positive Discrimination Rate", fontsize=12)
        ax2.set_title("Correct Ranking Rate", fontsize=13)
        ax2.set_ylim(0, 1.1)
        ax2.grid(axis="y", alpha=0.2)

        plt.suptitle(f"Semantic Discriminability ({dataset_name})", fontsize=14, y=1.01)
        plt.tight_layout()

        out_path = self.figs_dir / "exp1_discriminability.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Discriminability saved to {out_path}")
        return out_path

    def _plot_precision_at_k(self, prec_results: dict) -> Path:
        logger.info("Phase 10c: Precision@K visualization...")
        dataset_name = self.dataset_config["dataset"]["name"]

        methods = ["Q2Q", "Q2P", "Q2N", "Q2R", "Q2C"]
        method_keys = ["q2q", "q2p", "q2n", "q2r", "q2c"]
        colors = [METHOD_COLORS[m] for m in methods]
        k_values = ["P@1", "P@5", "P@10"]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        x = np.arange(len(methods))
        width = 0.6

        for ki, k_label in enumerate(k_values):
            ax = axes[ki]
            vals = [prec_results[m].get(k_label, 0) for m in method_keys]
            bars = ax.bar(x, vals, width, color=colors, alpha=0.85, edgecolor="white")
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2%}", ha="center", va="bottom", fontsize=9,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(methods, fontsize=10)
            ax.set_title(k_label, fontsize=13, fontweight="bold")
            ax.set_ylim(0, 1.15)
            ax.grid(axis="y", alpha=0.2)

        axes[0].set_ylabel("Precision", fontsize=12)
        plt.suptitle(f"Retrieval Precision@K ({dataset_name})", fontsize=14, y=1.01)
        plt.tight_layout()

        out_path = self.figs_dir / "exp1_precision_at_k.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Precision@K saved to {out_path}")
        return out_path

    # ------------------------------------------------------------------ #
    #  Console Summary                                                   #
    # ------------------------------------------------------------------ #

    def _print_summary(self, output: dict) -> None:
        overall = output.get("overall", {})
        multi = output.get("multi_method_stats", {})
        dist = output.get("distance_matrix", {})
        cluster = output.get("cluster_analysis", {})
        dir_align = output.get("directional_alignment", {})
        disc = output.get("discriminability", {})
        prec = output.get("precision_at_k", {})

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

        if dir_align:
            print("  --- Directional Alignment (Intent Direction Consistency) ---")
            for method in ["q2q", "q2p", "q2n", "q2r", "q2c"]:
                info = dir_align.get(method, {})
                print(f"    {method.upper():>4}: mean={info.get('mean', 0):.4f} (std={info.get('std', 0):.4f})")
            print()

        if disc:
            print("  --- Semantic Discriminability ---")
            for method in ["q2q", "q2p", "q2n", "q2r", "q2c"]:
                info = disc.get(method, {})
                print(f"    {method.upper():>4}: gap={info.get('mean', 0):.4f}, "
                      f"correct_rate={info.get('positive_rate', 0):.1%}")
            print()

        if prec:
            print("  --- Retrieval Precision@K ---")
            header = f"    {'':>4}"
            for k in ["P@1", "P@5", "P@10"]:
                header += f"  {k:>8}"
            print(header)
            for method in ["q2q", "q2p", "q2n", "q2r", "q2c"]:
                info = prec.get(method, {})
                line = f"    {method.upper():>4}"
                for k in ["P@1", "P@5", "P@10"]:
                    line += f"  {info.get(k, 0):>8.2%}"
                print(line)
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
