from __future__ import annotations
import os
import logging

from src.embedding.base import BaseEmbeddingProvider
from src.embedding.local_provider import LocalEmbeddingProvider
from src.embedding.api_provider import APIEmbeddingProvider

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
        return LocalEmbeddingProvider(model_path=model_name, device=device, **kwargs)
    elif provider in ("openai", "api"):
        return APIEmbeddingProvider(model_name=model_name or "text-embedding-3-small", **kwargs)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
