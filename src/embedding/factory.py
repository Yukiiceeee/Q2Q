from __future__ import annotations
import os
import logging

from src.embedding.base import BaseEmbeddingProvider
from src.embedding.local_provider import LocalEmbeddingProvider
from src.embedding.api_provider import APIEmbeddingProvider
from src.paths import models_dir, resolve

logger = logging.getLogger(__name__)


def create_embedding_provider(
    provider: str | None = None,
    model_name: str | None = None,
    **kwargs,
) -> BaseEmbeddingProvider:
    provider = provider or os.environ.get("EMBEDDING_PROVIDER", "local")
    model_name = model_name or os.environ.get("DEFAULT_EMBEDDING_MODEL", "")

    if provider == "local":
        if not model_name:
            raise ValueError("model_name (local path) required for local embedding provider")
        device = kwargs.pop("device", "cpu")
        # A bare directory name is resolved under <project_root>/models (see
        # MODELS_DIR) so configs stay portable; absolute paths pass through.
        model_path = str(resolve(model_name, root=models_dir()))
        return LocalEmbeddingProvider(model_path=model_path, device=device, **kwargs)
    elif provider in ("openai", "api"):
        return APIEmbeddingProvider(model_name=model_name or "text-embedding-3-small", **kwargs)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
