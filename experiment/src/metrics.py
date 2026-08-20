"""Metric computation utilities for Q2Q motivation experiments."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class AlignmentResult:
    query_id: str
    sim_q2q: float
    sim_q2c: float
    gap: float  # sim_q2q - sim_q2c
    best_fq_text: str = ""
    best_fq_idx: int = -1
    best_chunk_idx: int = -1
    sim_q2p: float = 0.0
    sim_q2n: float = 0.0
    sim_q2r: float = 0.0


@dataclass
class RobustnessResult:
    query_id: str
    original_sim_q2q: float
    original_sim_q2c: float
    paraphrase_sims_q2q: dict[str, float] = field(default_factory=dict)
    paraphrase_sims_q2c: dict[str, float] = field(default_factory=dict)
    var_q2q: float = 0.0
    var_q2c: float = 0.0
    robustness_q2q: float = 0.0
    robustness_q2c: float = 0.0
    original_sim_q2p: float = 0.0
    original_sim_q2n: float = 0.0
    original_sim_q2r: float = 0.0
    paraphrase_sims_q2p: dict[str, float] = field(default_factory=dict)
    paraphrase_sims_q2n: dict[str, float] = field(default_factory=dict)
    paraphrase_sims_q2r: dict[str, float] = field(default_factory=dict)
    var_q2p: float = 0.0
    var_q2n: float = 0.0
    var_q2r: float = 0.0
    robustness_q2p: float = 0.0
    robustness_q2n: float = 0.0
    robustness_q2r: float = 0.0


class AlignmentMetrics:

    @staticmethod
    def compute_max_sim(
        query_emb: np.ndarray,
        embeddings: np.ndarray,
    ) -> tuple[float, int]:
        if embeddings is not None and len(embeddings) > 0:
            sims = embeddings @ query_emb / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-10
            )
            best_idx = int(np.argmax(sims))
            return float(sims[best_idx]), best_idx
        return 0.0, -1

    @staticmethod
    def compute_alignment(
        query_emb: np.ndarray,
        fq_embeddings: np.ndarray,
        content_embeddings: np.ndarray,
    ) -> tuple[float, int, float, int]:
        """Compute Q2Q and Q2C alignment scores.

        Returns:
            (sim_q2q, best_fq_idx, sim_q2c, best_chunk_idx)
        """
        sim_q2q, best_fq_idx = AlignmentMetrics.compute_max_sim(query_emb, fq_embeddings)
        sim_q2c, best_chunk_idx = AlignmentMetrics.compute_max_sim(query_emb, content_embeddings)
        return sim_q2q, best_fq_idx, sim_q2c, best_chunk_idx

    @staticmethod
    def paired_statistics(results: list[AlignmentResult]) -> dict:
        """Compute aggregate statistics from alignment results."""
        if not results:
            return {}

        sims_q2q = np.array([r.sim_q2q for r in results])
        sims_q2c = np.array([r.sim_q2c for r in results])
        gaps = np.array([r.gap for r in results])

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(sims_q2q, sims_q2c)

        # Effect size (Cohen's d for paired samples)
        diff = sims_q2q - sims_q2c
        cohens_d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-10))

        # Win rate
        q2q_wins = int(np.sum(gaps > 0))
        q2c_wins = int(np.sum(gaps < 0))
        ties = int(np.sum(gaps == 0))

        return {
            "n_samples": len(results),
            "sim_q2q_mean": float(np.mean(sims_q2q)),
            "sim_q2q_std": float(np.std(sims_q2q)),
            "sim_q2c_mean": float(np.mean(sims_q2c)),
            "sim_q2c_std": float(np.std(sims_q2c)),
            "gap_mean": float(np.mean(gaps)),
            "gap_std": float(np.std(gaps)),
            "gap_median": float(np.median(gaps)),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": cohens_d,
            "q2q_win_rate": q2q_wins / len(results),
            "q2c_win_rate": q2c_wins / len(results),
            "q2q_wins": q2q_wins,
            "q2c_wins": q2c_wins,
            "ties": ties,
        }

    @staticmethod
    def robustness_statistics(results: list[RobustnessResult]) -> dict:
        """Compute aggregate robustness statistics."""
        if not results:
            return {}

        vars_q2q = np.array([r.var_q2q for r in results])
        vars_q2c = np.array([r.var_q2c for r in results])
        rob_q2q = np.array([r.robustness_q2q for r in results])
        rob_q2c = np.array([r.robustness_q2c for r in results])

        variance_ratio = float(np.mean(vars_q2c) / (np.mean(vars_q2q) + 1e-10))

        return {
            "n_samples": len(results),
            "var_q2q_mean": float(np.mean(vars_q2q)),
            "var_q2c_mean": float(np.mean(vars_q2c)),
            "variance_ratio_q2c_over_q2q": variance_ratio,
            "robustness_q2q_mean": float(np.mean(rob_q2q)),
            "robustness_q2c_mean": float(np.mean(rob_q2c)),
            "robustness_q2q_std": float(np.std(rob_q2q)),
            "robustness_q2c_std": float(np.std(rob_q2c)),
        }

    @staticmethod
    def category_breakdown(
        results: list[AlignmentResult],
        queries: list,
    ) -> dict[str, dict]:
        """Break down alignment results by query category."""
        cat_map: dict[str, list[AlignmentResult]] = {}
        query_cat = {q.query_id: q.category for q in queries}

        for r in results:
            cat = query_cat.get(r.query_id, "unknown")
            if cat not in cat_map:
                cat_map[cat] = []
            cat_map[cat].append(r)

        return {
            cat: AlignmentMetrics.paired_statistics(items)
            for cat, items in cat_map.items()
        }

    @staticmethod
    def multi_paired_statistics(sims_dict: dict[str, list[float]]) -> dict:
        """Compute pairwise statistics for all method pairs.

        Args:
            sims_dict: Mapping method_name -> list of per-query similarities,
                       e.g. {"q2q": [...], "q2c": [...], "q2p": [...], ...}

        Returns:
            Dict with per-method stats and pairwise comparisons.
        """
        methods = list(sims_dict.keys())
        result = {}

        for m in methods:
            arr = np.array(sims_dict[m])
            result[m] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
            }

        pairwise = {}
        for i, ma in enumerate(methods):
            for mb in methods[i + 1:]:
                a = np.array(sims_dict[ma])
                b = np.array(sims_dict[mb])
                diff = a - b
                t_stat, p_val = stats.ttest_rel(a, b)
                d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-10))
                win_a = int(np.sum(diff > 0))
                win_b = int(np.sum(diff < 0))
                pairwise[f"{ma}_vs_{mb}"] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_val),
                    "cohens_d": d,
                    f"{ma}_wins": win_a,
                    f"{mb}_wins": win_b,
                    f"{ma}_win_rate": win_a / len(a) if len(a) > 0 else 0,
                }
        result["pairwise"] = pairwise
        return result

    @staticmethod
    def compute_directional_alignment(
        query_emb: np.ndarray,
        content_centroid: np.ndarray,
        method_embs: np.ndarray,
    ) -> float:
        """Compute directional alignment: how well a method's best-match
        direction from content centroid aligns with the true query direction.

        Measures whether the method captures the *intent direction* — the
        displacement from general content toward the specific query need.

        Returns cosine similarity between the two direction vectors.
        """
        if method_embs is None or len(method_embs) == 0:
            return 0.0
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        c_norm = content_centroid / (np.linalg.norm(content_centroid) + 1e-10)

        intent_dir = q_norm - c_norm
        intent_norm = np.linalg.norm(intent_dir)
        if intent_norm < 1e-8:
            return 1.0
        intent_dir = intent_dir / intent_norm

        sims = method_embs @ q_norm / (
            np.linalg.norm(method_embs, axis=1) + 1e-10
        )
        best_idx = int(np.argmax(sims))
        best_emb = method_embs[best_idx]
        best_norm = best_emb / (np.linalg.norm(best_emb) + 1e-10)

        method_dir = best_norm - c_norm
        method_norm = np.linalg.norm(method_dir)
        if method_norm < 1e-8:
            return 0.0
        method_dir = method_dir / method_norm

        return float(intent_dir @ method_dir)

    @staticmethod
    def compute_discriminability(
        query_emb: np.ndarray,
        positive_embs: np.ndarray,
        negative_embs: np.ndarray,
    ) -> float:
        """Compute semantic discriminability: difference between max similarity
        to the correct session and max similarity to a distractor session.

        Positive value means the method correctly ranks the target higher.
        """
        pos_sim, _ = AlignmentMetrics.compute_max_sim(query_emb, positive_embs)
        neg_sim, _ = AlignmentMetrics.compute_max_sim(query_emb, negative_embs)
        return pos_sim - neg_sim
