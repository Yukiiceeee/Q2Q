"""Factory for creating embedding providers."""
from __future__ import annotations

from experiment.embedding.base import BaseEmbedding
from experiment.embedding.local_provider import LocalEmbedding
from experiment.embedding.api_provider import APIEmbedding


def create_embedding_provider(
    provider: str = "local",
    model_name: str = "",
    device: str = "cpu",
    dimension: int = 0,
    max_seq_length: int = 512,
    batch_size: int = 32,
    api_key: str = "",
    base_url: str = "",
) -> BaseEmbedding:
    if provider == "local":
        return LocalEmbedding(
            model_name=model_name,
            device=device,
            dimension=dimension,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
        )
    elif provider == "api":
        return APIEmbedding(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            dimension=dimension,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
