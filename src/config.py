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
class PreprocessingConfig:
    compressor_model: str = ""
    compressor_device: str = "cpu"
    perplexity_threshold: float = 5.0
    proposition_mode: str = "llm"
    nlp_rule_model: str = "en_core_web_sm"
    nlp_inference_model: str = ""


@dataclass
class GraphConfig:
    evolve_threshold: float = 0.6
    diverge_low: float = 0.2
    diverge_high: float = 0.6
    max_parents: int = 3
    maintenance_interval: int = 10
    merge_overlap_threshold: float = 0.9
    max_chain_length: int = 6
    chain_sample_interval: int = 2
    similarity_top_k: int = 5


@dataclass
class RetrievalConfig:
    top_k_per_sub: int = 20
    top_n: int = 5
    top_k_q2c: int = 3
    num_fake_queries: int = 15
    max_context_tokens: int = 10000
    chain_depth: int = 3
    prop_top_k_per_fq: int = 3
    binding_verify_threshold: float = 0.8
    fq_style: str = "formal"    # formal | colloquial | keyword | elliptical
    fq_role: str = "user"       # user | otherside


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
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    log: LogConfig = field(default_factory=LogConfig)
    language: str = "zh"

    @classmethod
    def from_env(cls, env_path: str | None = None) -> Q2QConfig:
        if env_path:
            load_dotenv(env_path, override=True)
        else:
            project_root = Path(__file__).parent.parent
            env_file = project_root / ".env"
            if env_file.exists():
                load_dotenv(str(env_file), override=True)

        config = cls()

        config.llm.provider = os.getenv("LLM_PROVIDER", "openai")
        config.llm.model = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
        config.llm.api_key = os.getenv("OPENAI_API_KEY", "")
        config.llm.base_url = os.getenv("OPENAI_BASE_URL", "")

        config.embedding.provider = os.getenv("EMBEDDING_PROVIDER", "local")
        config.embedding.model_name = os.getenv("DEFAULT_EMBEDDING_MODEL", "")
        config.embedding.device = os.getenv("EMBEDDING_DEVICE", "cpu")

        config.preprocessing.compressor_model = os.getenv(
            "COMPRESSOR_MODEL", "<MODELS_DIR>/Qwen2.5-0.5B-Instruct"
        )
        config.preprocessing.compressor_device = os.getenv("COMPRESSOR_DEVICE", "cpu")
        config.preprocessing.proposition_mode = os.getenv("PROPOSITION_MODE", "llm")
        config.preprocessing.nlp_inference_model = os.getenv(
            "NLP_INFERENCE_MODEL", "<MODELS_DIR>/flan-t5-large"
        )

        config.storage.chromadb_path = os.getenv("STORAGE_CHROMADB_PATH", "data/chromadb")
        config.storage.chromadb_collection = os.getenv("STORAGE_CHROMADB_COLLECTION", "q2q_memories")
        config.storage.json_path = os.getenv("STORAGE_JSON_PATH", "data/memories.json")

        config.retrieval.fq_style = os.getenv("FQ_STYLE", "formal")
        config.retrieval.fq_role = os.getenv("FQ_ROLE", "user")

        config.log.level = os.getenv("LOG_LEVEL", "INFO")
        config.log.log_dir = os.getenv("LOG_DIR", "logs")
        config.log.output_dir = os.getenv("OUTPUT_DIR", "outputs")

        return config
