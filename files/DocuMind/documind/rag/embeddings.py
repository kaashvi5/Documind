"""Embedding generation: Sentence-Transformers (default) with an offline hashing fallback.

All embedders L2-normalise their output (so inner product == cosine similarity), process texts in
batches and cache vectors by text hash, so identical text is never embedded twice.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Callable

import numpy as np

from config import DEFAULT_EMBEDDING_MODEL, get_device
from rag.textutils import STOPWORDS, stem, tokenize

ProgressFn = Callable[[int, int], None]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


class BaseEmbedder:
    name = "base"
    is_fallback = False

    def __init__(self, batch_size: int = 32, cache_size: int = 50_000):
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.generated = 0     # vectors computed by the model
        self.cache_hits = 0    # vectors served from the cache
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._lock = threading.RLock()

    @property
    def dim(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def _embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def encode(self, texts: list[str], batch_size: int | None = None, progress: ProgressFn | None = None) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dim) with unit-length rows."""
        texts = [t if isinstance(t, str) else str(t) for t in texts]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        keys = [hashlib.sha1(t.encode("utf-8", "ignore")).hexdigest() for t in texts]
        with self._lock:
            missing: dict[str, str] = {}
            for key, text in zip(keys, texts):
                if key in self._cache:
                    self.cache_hits += 1
                    self._cache.move_to_end(key)
                elif key not in missing:
                    missing[key] = text
            items = list(missing.items())
            size = max(1, batch_size or self.batch_size)
            for start in range(0, len(items), size):
                batch = items[start : start + size]
                vectors = _normalize(self._embed([text for _, text in batch]))
                for (key, _), vector in zip(batch, vectors):
                    self._cache[key] = vector
                self.generated += len(batch)
                if progress:
                    progress(min(start + size, len(items)), len(items))
            result = np.vstack([self._cache[k] for k in keys]).astype(np.float32)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return result

    def info(self) -> dict:
        return {"name": self.name, "fallback": self.is_fallback, "dim": self.dim,
                "generated": self.generated, "cache_hits": self.cache_hits}


class SentenceTransformerEmbedder(BaseEmbedder):
    """Wraps `sentence_transformers.SentenceTransformer` (GPU when available)."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str | None = None, batch_size: int = 32):
        super().__init__(batch_size=batch_size)
        self.name = model_name
        self.device = get_device(device)
        self._model = None
        self._dim: int | None = None

    def load(self) -> "SentenceTransformerEmbedder":
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.name, device=self.device)
                self._dim = int(self._model.encode(["dimension probe"], convert_to_numpy=True).shape[1])
        return self

    @property
    def dim(self) -> int:
        self.load()
        return int(self._dim)

    def _embed(self, texts: list[str]) -> np.ndarray:
        self.load()
        try:
            return self._model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True,
                                      normalize_embeddings=True, show_progress_bar=False)
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
        # GPU out of memory: free the cache, split the batch, and finally fall back to CPU.
        import torch

        torch.cuda.empty_cache()
        if len(texts) > 1:
            half = len(texts) // 2
            return np.vstack([self._embed(texts[:half]), self._embed(texts[half:])])
        self._model.to("cpu")
        self.device = "cpu"
        return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)


class HashingEmbedder(BaseEmbedder):
    """Dependency-free lexical embedder (hashed bag of stems + bigrams).

    Used for unit tests and as an emergency fallback when the neural model cannot be loaded.
    It matches on shared vocabulary, not meaning, so retrieval quality is lower.
    """

    name = "hashing-fallback"
    is_fallback = True

    def __init__(self, dim: int = 384, batch_size: int = 64):
        super().__init__(batch_size=batch_size)
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _slot(self, token: str) -> tuple[int, float]:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") % self._dim, 1.0 if digest[4] & 1 else -1.0

    def _embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = [stem(t) for t in tokenize(text) if t not in STOPWORDS]
            counts: dict[str, float] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0.0) + 1.0
            for token, count in counts.items():
                idx, sign = self._slot(token)
                out[row, idx] += sign * (1.0 + np.log(count))
            for left, right in zip(tokens, tokens[1:]):
                idx, sign = self._slot(f"{left}_{right}")
                out[row, idx] += 0.5 * sign
        return out


def load_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL, device: str | None = None,
                  allow_fallback: bool = True) -> tuple[BaseEmbedder, str | None]:
    """Load the neural embedder; on failure return the hashing fallback plus a warning."""
    try:
        return SentenceTransformerEmbedder(model_name, device).load(), None
    except Exception as exc:
        if not allow_fallback:
            raise
        warning = (f"Could not load embedding model '{model_name}' ({type(exc).__name__}: {exc}). "
                   "Using the lexical hashing fallback; retrieval quality is reduced.")
        return HashingEmbedder(), warning
