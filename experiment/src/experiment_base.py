"""Base class for all experiments with shared infrastructure."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

import yaml

from experiment.embedding import create_embedding_provider, BaseEmbedding
from experiment.store import EmbeddingStore
from experiment.src.data_loader import BaseDataLoader, load_dataset

logger = logging.getLogger(__name__)


class ExperimentBase(ABC):
    """Base class providing config loading, embedding, and storage access."""

    def __init__(self, config_path: str, dataset_config_path: str):
        self.base_config = self._load_yaml(config_path)
        self.dataset_config = self._load_yaml(dataset_config_path)
        self._merge_env_vars()

        self.embedding_provider: BaseEmbedding = create_embedding_provider(
            provider=self.base_config["embedding"]["provider"],
            model_name=self.base_config["embedding"]["model_name"],
            device=self.base_config["embedding"].get("device", "cpu"),
            dimension=self.base_config["embedding"].get("dimension", 0),
            max_seq_length=self.base_config["embedding"].get("max_seq_length", 512),
            batch_size=self.base_config["embedding"].get("batch_size", 32),
            api_key=self.base_config["embedding"].get("api_key", ""),
            base_url=self.base_config["embedding"].get("base_url", ""),
        )

        store_path = self.dataset_config.get("store", {}).get(
            "base_dir",
            f"experiment/store/{self.dataset_config['dataset']['name']}",
        )
        self.store = EmbeddingStore(store_path)

        self.data_loader: BaseDataLoader | None = None
        self.results_dir = Path(
            self.base_config.get("experiment", {}).get("results_dir", "experiment/results")
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> BaseDataLoader:
        if self.data_loader is None:
            self.data_loader = load_dataset(self.dataset_config["dataset"])
        return self.data_loader

    def create_llm_client(self):
        """Create LLM client using the main project's infrastructure."""
        import sys
        project_root = str(Path(__file__).parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from src.utils.llm_client import create_llm_client

        llm_cfg = self.base_config["llm"]
        return create_llm_client(
            provider=llm_cfg["provider"],
            model=llm_cfg["model"],
            temperature=llm_cfg.get("temperature", 0.7),
            max_tokens=llm_cfg.get("max_tokens", 2000),
            api_key=llm_cfg.get("api_key") or None,
            base_url=llm_cfg.get("base_url") or None,
        )

    @abstractmethod
    async def run(self) -> dict:
        ...

    def save_results(self, results: dict, filename: str) -> Path:
        dataset_name = self.dataset_config["dataset"]["name"]
        out_dir = self.results_dir / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Results saved to {out_path}")
        return out_path

    def _load_yaml(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _merge_env_vars(self) -> None:
        """Override config values from environment variables."""
        llm = self.base_config.get("llm", {})
        if not llm.get("api_key"):
            llm["api_key"] = os.environ.get("OPENAI_API_KEY", "")
        if not llm.get("base_url"):
            llm["base_url"] = os.environ.get("OPENAI_BASE_URL", "")

        emb = self.base_config.get("embedding", {})
        if not emb.get("model_name"):
            emb["model_name"] = os.environ.get("DEFAULT_EMBEDDING_MODEL", "")
        if not emb.get("api_key"):
            emb["api_key"] = os.environ.get("OPENAI_API_KEY", "")
        if not emb.get("base_url"):
            emb["base_url"] = os.environ.get("OPENAI_BASE_URL", "")
