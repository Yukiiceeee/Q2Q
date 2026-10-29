from __future__ import annotations
from dataclasses import dataclass, field
import os
from pathlib import Path

from dotenv import load_dotenv

from src.paths import model_path


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


@dataclass
class PreprocessingConfig:
    compressor_model: str = ""
    compressor_device: str = "cpu"
    perplexity_threshold: float = 3.0
    proposition_mode: str = "llm"
    nlp_inference_model: str = ""


@dataclass
class GraphConfig:
    evolve_threshold: float = 0.6
    diverge_low: float = 0.2
    diverge_high: float = 0.6
    # Embedding-similarity fallback for edge creation, used when proposition
    # overlap is too sparse to link two FQs on the same thread.
    link_similarity: float = 0.82
    max_parents: int = 3
    maintenance_interval: int = 10
    merge_overlap_threshold: float = 0.9
    max_chain_length: int = 6
    chain_sample_interval: int = 2
    max_chain_nodes: int = 60
    similarity_top_k: int = 5


@dataclass
class RetrievalConfig:
    top_k_per_sub: int = 20
    top_n: int = 5
    top_k_q2c: int = 3
    num_fake_queries: int = 15
    max_context_tokens: int = 10000
    session_text_budget_tokens: int = 0
    chain_depth: int = 3
    prop_top_k_per_fq: int = 3
    max_fqs_per_session: int = 3
    binding_verify_threshold: float = 0.8
    max_extra_bindings: int = 3  # cap on embedding-added links per fake query
    fq_style: str = "formal"    # formal | colloquial | keyword | elliptical
    fq_role: str = "user"       # user | otherside


@dataclass
class StorageConfig:
    backend: str = "chromadb"
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

        # Model defaults resolve under <project_root>/models, which MODELS_DIR
        # redirects to a shared cache. Never hardcode an absolute path here:
        # it only works on the machine it was written on.
        config.preprocessing.compressor_model = os.getenv(
            "COMPRESSOR_MODEL", model_path("Qwen2.5-0.5B-Instruct")
        )
        config.preprocessing.compressor_device = os.getenv("COMPRESSOR_DEVICE", "cpu")
        config.preprocessing.proposition_mode = os.getenv("PROPOSITION_MODE", "llm")
        config.preprocessing.nlp_inference_model = os.getenv(
            "NLP_INFERENCE_MODEL", model_path("flan-t5-large")
        )

        config.storage.chromadb_path = os.getenv("STORAGE_CHROMADB_PATH", "data/chromadb")
        config.storage.chromadb_collection = os.getenv("STORAGE_CHROMADB_COLLECTION", "q2q_memories")

        config.retrieval.fq_style = os.getenv("FQ_STYLE", "formal")
        config.retrieval.fq_role = os.getenv("FQ_ROLE", "user")

        # Prompt language for memory construction and answering. Falls back to
        # the dataclass default when unset, so an existing .env keeps its
        # current behaviour. The MedMemoryBench adapter sets this field
        # directly from the dataset config instead of through the environment.
        config.language = os.getenv("LANGUAGE", config.language)

        config.log.level = os.getenv("LOG_LEVEL", "INFO")
        config.log.log_dir = os.getenv("LOG_DIR", "logs")
        config.log.output_dir = os.getenv("OUTPUT_DIR", "outputs")

        return config
