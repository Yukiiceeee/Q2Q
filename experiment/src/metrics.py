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


class AlignmentMetrics:

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
        # Q2Q: max cosine similarity with any fake query
        if fq_embeddings is not None and len(fq_embeddings) > 0:
            sims_q2q = fq_embeddings @ query_emb / (
                np.linalg.norm(fq_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-10
            )
            best_fq_idx = int(np.argmax(sims_q2q))
            sim_q2q = float(sims_q2q[best_fq_idx])
        else:
            sim_q2q, best_fq_idx = 0.0, -1

        # Q2C: max cosine similarity with any content chunk
        if content_embeddings is not None and len(content_embeddings) > 0:
            sims_q2c = content_embeddings @ query_emb / (
                np.linalg.norm(content_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-10
            )
            best_chunk_idx = int(np.argmax(sims_q2c))
            sim_q2c = float(sims_q2c[best_chunk_idx])
        else:
            sim_q2c, best_chunk_idx = 0.0, -1

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
