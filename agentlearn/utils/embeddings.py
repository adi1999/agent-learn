"""Embedding generation and similarity utilities."""

from typing import Optional

import numpy as np

from .logging import get_logger

logger = get_logger("embeddings")

EMBEDDING_DIM = 1536  # text-embedding-3-small default

_openai_client = None


def _get_openai_client():
    """Lazy singleton for OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI()
    return _openai_client


def get_embedding(
    text: str,
    model: str = "text-embedding-3-small",
    client=None,
) -> np.ndarray:
    """Get embedding vector for a text string."""
    client = client or _get_openai_client()
    text = text.replace("\n", " ").strip()
    if not text:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding, dtype=np.float32)


def batch_embeddings(
    texts: list[str],
    model: str = "text-embedding-3-small",
    client=None,
) -> list[np.ndarray]:
    """Get embeddings for multiple texts in a single API call."""
    if not texts:
        return []
    client = client or _get_openai_client()
    cleaned = [t.replace("\n", " ").strip() or " " for t in texts]
    response = client.embeddings.create(input=cleaned, model=model)
    return [np.array(d.embedding, dtype=np.float32) for d in response.data]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def serialize_embedding(embedding: np.ndarray) -> bytes:
    """Serialize embedding to bytes for SQLite BLOB storage."""
    return embedding.astype(np.float32).tobytes()


def deserialize_embedding(data: bytes, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Deserialize embedding from bytes."""
    return np.frombuffer(data, dtype=np.float32).copy()
