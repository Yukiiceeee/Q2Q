"""Experiment 2: Robustness Analysis (Q2Q vs Q2P/Q2N/Q2R/Q2C under query reformulation).

For each true query with indirect-query variants (5 styles):
1. Compute original sim for all 5 methods using the true query
2. Compute sim for each indirect query style across all 5 methods
3. Compare variance, drop rate, and stability across styles
4. Friedman test for systematic style differences
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

from experiment.src.experiment_base import ExperimentBase
from experiment.src.metrics import AlignmentMetrics, RobustnessResult

logger = logging.getLogger(__name__)

STYLE_LABELS = {
    "implication": "Implication",
    "scenario": "Scenario",
    "consequence": "Consequence",
    "peripheral": "Peripheral",
    "negation_contrast": "Negation",
}
DROP_THRESHOLD = 0.9

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


class Exp2Robustness(ExperimentBase):

    async def run(self) -> dict:
        logger.info("=" * 60)
        logger.info("Experiment 2: Robustness Analysis (5 methods)")
        logger.info("=" * 60)

        start_time = time.time()
        loader = self.load_data()

        self.exp_logger.start_phase("embeddings")
        await self._ensure_embeddings(loader)
        self.exp_logger.end_phase("embeddings")

        self.exp_logger.start_phase("robustness")
        results = self._compute_robustness(loader)
        self.exp_logger.end_phase("robustness", {"n_queries": len(results)})

        self.exp_logger.start_phase("statistics")
        overall_stats = self._overall_statistics(results)
        style_stats = self._per_style_analysis(results)
        friedman = self._friedman_test(results)
        drop_rates = self._compute_drop_rates(results)
        self.exp_logger.end_phase("statistics")

        self.exp_logger.start_phase("visualization")
        fig_paths = self._visualize(results)
        self.exp_logger.end_phase("visualization")

        elapsed = time.time() - start_time
        output = {
            "experiment": "exp2_robustness",
            "dataset": self.dataset_config["dataset"]["name"],
            "n_queries": len(results),
            "elapsed_seconds": round(elapsed, 2),
            "overall": overall_stats,
            "per_style": style_stats,
            "friedman_test": friedman,
            "drop_rates": drop_rates,
            "figures": {k: str(v) for k, v in fig_paths.items()},
            "sample_results": [
                {
                    "query_id": r.query_id,
                    "original_q2q": round(r.original_sim_q2q, 4),
                    "original_q2p": round(r.original_sim_q2p, 4),
                    "original_q2n": round(r.original_sim_q2n, 4),
                    "original_q2r": round(r.original_sim_q2r, 4),
                    "original_q2c": round(r.original_sim_q2c, 4),
                    "var_q2q": round(r.var_q2q, 6),
                    "var_q2c": round(r.var_q2c, 6),
                    "var_q2p": round(r.var_q2p, 6),
                    "var_q2n": round(r.var_q2n, 6),
                    "var_q2r": round(r.var_q2r, 6),
                    "robustness_q2q": round(r.robustness_q2q, 4),
                    "robustness_q2c": round(r.robustness_q2c, 4),
                    "robustness_q2p": round(r.robustness_q2p, 4),
                    "robustness_q2n": round(r.robustness_q2n, 4),
                    "robustness_q2r": round(r.robustness_q2r, 4),
                }
                for r in results[:20]
            ],
        }

        self.save_results(output, "exp2_robustness.json")
        self.save_experiment_log(output, "exp2_robustness_log.json")
        self._print_summary(output)
        return output

    async def _ensure_embeddings(self, loader) -> None:
        logger.info("Phase 1: Checking embeddings...")
        n_missing_tq = sum(1 for q in loader.queries if not self.store.has_true_query(q.query_id))
        n_missing_para = sum(1 for q in loader.queries if not self.store.has_paraphrases(q.query_id))
        n_missing_fq = 0
        for q in loader.queries:
            for sid in q.evidence_session_ids:
                if not self.store.has_fake_queries(sid):
                    n_missing_fq += 1
                    break

        if n_missing_tq or n_missing_para or n_missing_fq:
            logger.warning(
                f"  Missing: TQ={n_missing_tq}, Paraphrases={n_missing_para}, "
                f"FQ sessions={n_missing_fq}. Run --step embed first."
            )
        else:
            logger.info("  All embeddings available.")

    def _compute_robustness(self, loader) -> list[RobustnessResult]:
        logger.info("Phase 2: Computing robustness scores (5 methods)...")
        results = []
        skipped = 0

        for q in loader.queries:
            tq_emb = self.store.get_true_query_embedding(q.query_id)
            para_embs = self.store.get_paraphrase_embeddings(q.query_id)
            styles = self.store.get_paraphrase_styles(q.query_id)

            if tq_emb is None or para_embs is None or styles is None:
                skipped += 1
                continue

            best_result = None
            for sid in q.evidence_session_ids:
                fq_embs = self.store.get_fake_query_embeddings(sid)
                c_embs = self.store.get_session_content(sid)
                if fq_embs is None or c_embs is None:
                    continue

                p_embs = self.store.get_variant_embeddings("propositions", sid)
                n_embs = self.store.get_variant_embeddings("notes", sid)
                r_embs = self.store.get_variant_embeddings("reflections", sid)

                def _normed(embs):
                    if embs is None or len(embs) == 0:
                        return None
                    return embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)

                fq_normed = _normed(fq_embs)
                c_normed = _normed(c_embs)
                p_normed = _normed(p_embs)
                n_normed = _normed(n_embs)
                r_normed = _normed(r_embs)

                tq_norm = tq_emb / (np.linalg.norm(tq_emb) + 1e-10)

                def _max_sim(normed_pool, query_norm):
                    if normed_pool is None:
                        return 0.0
                    return float(np.max(normed_pool @ query_norm))

                orig_q2q = _max_sim(fq_normed, tq_norm)
                orig_q2c = _max_sim(c_normed, tq_norm)
                orig_q2p = _max_sim(p_normed, tq_norm)
                orig_q2n = _max_sim(n_normed, tq_norm)
                orig_q2r = _max_sim(r_normed, tq_norm)

                pools = {
                    "q2q": fq_normed, "q2c": c_normed,
                    "q2p": p_normed, "q2n": n_normed, "q2r": r_normed,
                }
                originals = {
                    "q2q": orig_q2q, "q2c": orig_q2c,
                    "q2p": orig_q2p, "q2n": orig_q2n, "q2r": orig_q2r,
                }

                style_sims = {m: {} for m in METHOD_KEYS}
                for i, style in enumerate(styles):
                    iq_emb = para_embs[i]
                    iq_norm = iq_emb / (np.linalg.norm(iq_emb) + 1e-10)
                    for m in METHOD_KEYS:
                        style_sims[m][style] = _max_sim(pools[m], iq_norm)

                if best_result is None or orig_q2q > best_result.original_sim_q2q:
                    def _var_and_rob(orig, style_dict):
                        vals = list(style_dict.values())
                        if not vals:
                            return 0.0, 0.0
                        v = float(np.var(vals))
                        drop = max(0.0, orig - min(vals))
                        rob = 1.0 - drop / (orig + 1e-10)
                        return v, rob

                    var_q2q, rob_q2q = _var_and_rob(orig_q2q, style_sims["q2q"])
                    var_q2c, rob_q2c = _var_and_rob(orig_q2c, style_sims["q2c"])
                    var_q2p, rob_q2p = _var_and_rob(orig_q2p, style_sims["q2p"])
                    var_q2n, rob_q2n = _var_and_rob(orig_q2n, style_sims["q2n"])
                    var_q2r, rob_q2r = _var_and_rob(orig_q2r, style_sims["q2r"])

                    best_result = RobustnessResult(
                        query_id=q.query_id,
                        original_sim_q2q=orig_q2q,
                        original_sim_q2c=orig_q2c,
                        paraphrase_sims_q2q=dict(style_sims["q2q"]),
                        paraphrase_sims_q2c=dict(style_sims["q2c"]),
                        var_q2q=var_q2q,
                        var_q2c=var_q2c,
                        robustness_q2q=rob_q2q,
                        robustness_q2c=rob_q2c,
                        original_sim_q2p=orig_q2p,
                        original_sim_q2n=orig_q2n,
                        original_sim_q2r=orig_q2r,
                        paraphrase_sims_q2p=dict(style_sims["q2p"]),
                        paraphrase_sims_q2n=dict(style_sims["q2n"]),
                        paraphrase_sims_q2r=dict(style_sims["q2r"]),
                        var_q2p=var_q2p,
                        var_q2n=var_q2n,
                        var_q2r=var_q2r,
                        robustness_q2p=rob_q2p,
                        robustness_q2n=rob_q2n,
                        robustness_q2r=rob_q2r,
                    )

            if best_result:
                results.append(best_result)

        if skipped:
            logger.warning(f"  Skipped {skipped} queries (missing embeddings)")
        logger.info(f"  Computed robustness for {len(results)} queries.")
        return results

    def _overall_statistics(self, results: list[RobustnessResult]) -> dict:
        if not results:
            return {}

        stats = {}
        for m in METHOD_KEYS:
            vars_arr = np.array([getattr(r, f"var_{m}") for r in results])
            rob_arr = np.array([getattr(r, f"robustness_{m}") for r in results])
            stats[f"var_{m}_mean"] = float(np.mean(vars_arr))
            stats[f"robustness_{m}_mean"] = float(np.mean(rob_arr))
            stats[f"robustness_{m}_std"] = float(np.std(rob_arr))

        stats["n_samples"] = len(results)
        stats["variance_ratio_q2c_over_q2q"] = float(
            stats["var_q2c_mean"] / (stats["var_q2q_mean"] + 1e-10)
        )
        return stats

    def _per_style_analysis(self, results: list[RobustnessResult]) -> dict:
        if not results:
            return {}
        all_styles = list(results[0].paraphrase_sims_q2q.keys())
        style_stats = {}
        for style in all_styles:
            entry = {}
            for m in METHOD_KEYS:
                sims_key = f"paraphrase_sims_{m}"
                orig_key = f"original_sim_{m}"
                vals = [getattr(r, sims_key).get(style, 0) for r in results]
                origs = [getattr(r, orig_key) for r in results]
                entry[f"mean_sim_{m}"] = round(float(np.mean(vals)), 4)
                entry[f"mean_drop_{m}"] = round(float(np.mean(
                    np.array(origs) - np.array(vals)
                )), 4)
            style_stats[style] = entry
        return style_stats

    def _friedman_test(self, results: list[RobustnessResult]) -> dict:
        if len(results) < 10:
            return {"note": "insufficient samples"}
        styles = list(results[0].paraphrase_sims_q2q.keys())

        friedman_results = {}
        for m in METHOD_KEYS:
            sims_key = f"paraphrase_sims_{m}"
            matrix = np.array([
                [getattr(r, sims_key).get(s, 0) for s in styles]
                for r in results
            ])
            stat, p_val = sp_stats.friedmanchisquare(*[matrix[:, i] for i in range(len(styles))])
            friedman_results[m] = {
                "statistic": round(float(stat), 4),
                "p_value": float(p_val),
            }

        friedman_results["interpretation"] = (
            "Lower Friedman statistic indicates more robust behavior "
            "(less systematic variation across reformulation styles)."
        )
        return friedman_results

    def _compute_drop_rates(self, results: list[RobustnessResult]) -> dict:
        if not results:
            return {}
        styles = list(results[0].paraphrase_sims_q2q.keys())

        drop_rates = {"threshold": DROP_THRESHOLD}
        for m in METHOD_KEYS:
            sims_key = f"paraphrase_sims_{m}"
            orig_key = f"original_sim_{m}"
            per_style = {}
            total_drop = 0
            total = 0
            for style in styles:
                n_drop = 0
                for r in results:
                    sim = getattr(r, sims_key).get(style, 0)
                    orig = getattr(r, orig_key)
                    if sim < orig * DROP_THRESHOLD:
                        n_drop += 1
                per_style[style] = round(n_drop / len(results), 4)
                total_drop += n_drop
                total += len(results)
            drop_rates[f"per_style_{m}"] = per_style
            drop_rates[f"overall_{m}"] = round(total_drop / total, 4) if total else 0
        return drop_rates

    # ------------------------------------------------------------------ #
    #  Visualization                                                     #
    # ------------------------------------------------------------------ #

    def _visualize(self, results: list[RobustnessResult]) -> dict[str, Path]:
        fig_paths = {}
        if not results:
            return fig_paths
        fig_paths["style_boxplot"] = self._plot_style_boxplot(results)
        fig_paths["variance_scatter"] = self._plot_variance_scatter(results)
        fig_paths["drop_rate"] = self._plot_drop_rate(results)
        return fig_paths

    def _plot_style_boxplot(self, results: list[RobustnessResult]) -> Path:
        styles = list(results[0].paraphrase_sims_q2q.keys())
        n_styles = len(styles)
        n_methods = len(METHOD_KEYS)
        group_width = n_methods + 1

        fig, ax = plt.subplots(figsize=(16, 7))
        for mi, m in enumerate(METHOD_KEYS):
            sims_key = f"paraphrase_sims_{m}"
            data = [
                [getattr(r, sims_key).get(s, 0) for r in results]
                for s in styles
            ]
            positions = np.arange(n_styles) * group_width + mi
            bp = ax.boxplot(
                data, positions=positions, widths=0.7,
                patch_artist=True, showfliers=False,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(METHOD_COLORS[m])
                patch.set_alpha(0.7)

        tick_positions = np.arange(n_styles) * group_width + (n_methods - 1) / 2
        labels = [STYLE_LABELS.get(s, s) for s in styles]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("Max Cosine Similarity", fontsize=12)
        ax.set_title("Robustness by Indirect Query Style (5 Methods)", fontsize=14)

        legend_patches = [
            plt.Rectangle((0, 0), 1, 1, facecolor=METHOD_COLORS[m], alpha=0.7)
            for m in METHOD_KEYS
        ]
        ax.legend(legend_patches, [METHOD_LABELS[m] for m in METHOD_KEYS],
                  fontsize=10, loc="lower left")
        plt.tight_layout()

        out_path = self.figs_dir / "exp2_style_boxplot.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Style boxplot saved to {out_path}")
        return out_path

    def _plot_variance_scatter(self, results: list[RobustnessResult]) -> Path:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        baselines = ["q2c", "q2p", "q2n", "q2r"]

        for ax, bl in zip(axes.flat, baselines):
            var_q2q = np.array([r.var_q2q for r in results])
            var_bl = np.array([getattr(r, f"var_{bl}") for r in results])

            ax.scatter(var_q2q, var_bl, alpha=0.3, s=12, c=METHOD_COLORS[bl], edgecolors="none")
            max_val = max(np.max(var_q2q), np.max(var_bl)) * 1.1
            ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5)
            ax.set_xlabel("Var(Q2Q)", fontsize=10)
            ax.set_ylabel(f"Var({METHOD_LABELS[bl]})", fontsize=10)
            ax.set_title(f"Q2Q vs {METHOD_LABELS[bl]}", fontsize=11)

            above = np.sum(var_bl > var_q2q)
            ax.text(0.05, 0.95, f"{METHOD_LABELS[bl]} more variable: {above}/{len(results)}",
                    transform=ax.transAxes, fontsize=9, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.suptitle("Variance Comparison: Q2Q vs Baselines", fontsize=13)
        plt.tight_layout()

        out_path = self.figs_dir / "exp2_variance_scatter.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Variance scatter saved to {out_path}")
        return out_path

    def _plot_drop_rate(self, results: list[RobustnessResult]) -> Path:
        drop_rates = self._compute_drop_rates(results)
        styles = list(drop_rates["per_style_q2q"].keys())
        labels = [STYLE_LABELS.get(s, s) for s in styles]
        n_styles = len(styles)
        n_methods = len(METHOD_KEYS)

        x = np.arange(n_styles)
        width = 0.8 / n_methods

        fig, ax = plt.subplots(figsize=(14, 7))
        for mi, m in enumerate(METHOD_KEYS):
            rates = [drop_rates[f"per_style_{m}"][s] for s in styles]
            offset = (mi - (n_methods - 1) / 2) * width
            bars = ax.bar(
                x + offset, rates, width,
                label=METHOD_LABELS[m], color=METHOD_COLORS[m], alpha=0.8,
            )
            for bar, v in zip(bars, rates):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.0%}", ha="center", va="bottom", fontsize=7,
                )

        ax.set_ylabel(f"Drop Rate (threshold={DROP_THRESHOLD})", fontsize=12)
        ax.set_title("Similarity Drop Rate by Style (5 Methods)", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.0)
        plt.tight_layout()

        out_path = self.figs_dir / "exp2_drop_rate.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Drop rate chart saved to {out_path}")
        return out_path

    # ------------------------------------------------------------------ #
    #  Console Summary                                                   #
    # ------------------------------------------------------------------ #

    def _print_summary(self, output: dict) -> None:
        overall = output.get("overall", {})
        friedman = output.get("friedman_test", {})
        drops = output.get("drop_rates", {})

        print("\n" + "=" * 60)
        print("  Experiment 2 Results: Robustness Analysis (5 Methods)")
        print("=" * 60)
        print(f"  Dataset: {output['dataset']}")
        print(f"  Queries analyzed: {output['n_queries']}")
        print(f"  Time: {output['elapsed_seconds']}s")
        print()

        print("  --- Variance & Robustness ---")
        for m in METHOD_KEYS:
            var_val = overall.get(f"var_{m}_mean", 0)
            rob_val = overall.get(f"robustness_{m}_mean", 0)
            rob_std = overall.get(f"robustness_{m}_std", 0)
            print(f"    {METHOD_LABELS[m]:>4}: var={var_val:.6f}, robustness={rob_val:.4f} (std={rob_std:.4f})")
        print()

        if friedman:
            print("  --- Friedman Test ---")
            for m in METHOD_KEYS:
                f_info = friedman.get(m, {})
                if f_info:
                    print(f"    {METHOD_LABELS[m]:>4}: chi2={f_info.get('statistic', 0):.2f}, "
                          f"p={f_info.get('p_value', 1):.2e}")
            print()

        if drops:
            print(f"  --- Overall Drop Rate (threshold={drops.get('threshold', 0.9)}) ---")
            for m in METHOD_KEYS:
                print(f"    {METHOD_LABELS[m]:>4}: {drops.get(f'overall_{m}', 0):.1%}")
            print()

        figs = output.get("figures", {})
        if figs:
            print("  --- Figures ---")
            for name, path in figs.items():
                print(f"    {name}: {path}")
        print("=" * 60 + "\n")
