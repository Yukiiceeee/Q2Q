"""Structured experiment logger for timing, config snapshot, and result tracking."""
from __future__ import annotations

import copy
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class ExperimentLogger:

    def __init__(self, experiment_name: str, dataset_name: str, config: dict):
        self.experiment_name = experiment_name
        self.dataset_name = dataset_name
        self.config_snapshot = copy.deepcopy(config)
        self.phases: list[dict] = []
        self._phase_starts: dict[str, float] = {}
        self._start_time = time.time()
        self._start_iso = datetime.now(timezone.utc).isoformat()

    def start_phase(self, name: str) -> None:
        self._phase_starts[name] = time.time()
        logger.debug(f"Phase started: {name}")

    def end_phase(self, name: str, metadata: dict | None = None) -> None:
        start = self._phase_starts.pop(name, None)
        elapsed = time.time() - start if start else 0.0
        self.phases.append({
            "name": name,
            "elapsed_seconds": round(elapsed, 3),
            "metadata": metadata or {},
        })
        logger.debug(f"Phase ended: {name} ({elapsed:.2f}s)")

    def finalize(self, results: dict) -> dict:
        total_elapsed = time.time() - self._start_time
        summary = {}
        for k, v in results.items():
            if isinstance(v, (int, float, str, bool)):
                summary[k] = v
            elif isinstance(v, dict) and k in ("overall",):
                summary[k] = v

        return {
            "experiment_name": self.experiment_name,
            "dataset_name": self.dataset_name,
            "start_time": self._start_iso,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "total_elapsed_seconds": round(total_elapsed, 3),
            "config": self.config_snapshot,
            "phases": self.phases,
            "results_summary": summary,
        }

    def save(self, results_dir: Path, filename: str) -> Path:
        out_dir = results_dir / self.dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        log_data = self.finalize({}) if not hasattr(self, "_finalized") else self._finalized
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Experiment log saved to {out_path}")
        return out_path
