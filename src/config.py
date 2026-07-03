from __future__ import annotations
from dataclasses import dataclass, field
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2000
    api_key: str = ""
    base_url: str = ""


@dataclass
class EmbeddingConfig:
    provider: str = "local"
    model_name: str = ""
    device: str = "cpu"
    dimension: int = 512
    max_seq_length: int = 512


@dataclass
class RetrievalConfig:
    alpha: float = 0.7
    top_k_per_sub: int = 20
    top_n: int = 5
    top_k_q2c: int = 3
    num_fake_queries: int = 10


@dataclass
class StorageConfig:
    backend: str = "chromadb"
    json_path: str = "data/memories.json"
    chromadb_path: str = "data/chromadb"
    chromadb_collection: str = "q2q_memories"


@dataclass
class LogConfig:
    level: str = "INFO"
    log_dir: str = "logs"
    output_dir: str = "outputs"


@dataclass
class Q2QConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    log: LogConfig = field(default_factory=LogConfig)
    language: str = "zh"

    @classmethod
    def from_env(cls, env_path: str | None = None) -> Q2QConfig:
        """Load environment-level config from .env file.

        Only loads infrastructure settings (API keys, paths, model names).
        Runtime hyperparameters use defaults and should be overridden by scripts/CLI.
        """
        if env_path:
            load_dotenv(env_path, override=True)
        else:
            project_root = Path(__file__).parent.parent
            env_file = project_root / ".env"
            if env_file.exists():
                load_dotenv(str(env_file), override=True)

        config = cls()

        # --- Environment-level: API keys, endpoints, model paths ---
        config.llm.provider = os.getenv("LLM_PROVIDER", "openai")
        config.llm.model = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
        config.llm.api_key = os.getenv("OPENAI_API_KEY", "")
        config.llm.base_url = os.getenv("OPENAI_BASE_URL", "")

        config.embedding.provider = os.getenv("EMBEDDING_PROVIDER", "local")
        config.embedding.model_name = os.getenv("DEFAULT_EMBEDDING_MODEL", "")
        config.embedding.device = os.getenv("EMBEDDING_DEVICE", "cpu")

        # Storage paths (from env)
        config.storage.chromadb_path = os.getenv("STORAGE_CHROMADB_PATH", "data/chromadb")
        config.storage.chromadb_collection = os.getenv("STORAGE_CHROMADB_COLLECTION", "q2q_memories")
        config.storage.json_path = os.getenv("STORAGE_JSON_PATH", "data/memories.json")

        # Log paths (from env)
        config.log.level = os.getenv("LOG_LEVEL", "INFO")
        config.log.log_dir = os.getenv("LOG_DIR", "logs")
        config.log.output_dir = os.getenv("OUTPUT_DIR", "outputs")

        # --- Defaults for runtime hyperparameters (overridden by scripts/CLI) ---
        # These use class defaults, NOT read from .env
        # config.retrieval.alpha = 0.7 (default)
        # config.retrieval.top_k_per_sub = 5 (default)
        # config.retrieval.top_n = 10 (default)
        # config.retrieval.num_fake_queries = 5 (default)
        # config.storage.backend = "chromadb" (default)
        # config.language = "zh" (default)

        return config
