import numpy as np

from rag import embeddings
from rag.embeddings import HashingEmbedder, load_embedder

TEXTS = ["The cat sat on the mat.", "Dogs love long walks outside.", "Quarterly revenue grew by ten percent.", "The cat sat on the mat."]


def test_shape_dtype_and_normalisation(embedder):
    vectors = embedder.encode(TEXTS)
    assert vectors.shape == (4, embedder.dim) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert np.allclose(vectors[0], vectors[3])


def test_batch_size_does_not_change_results():
    a = HashingEmbedder().encode(TEXTS, batch_size=1)
    b = HashingEmbedder().encode(TEXTS, batch_size=64)
    assert np.allclose(a, b)


def test_cache_avoids_regeneration(embedder):
    embedder.encode(TEXTS)
    assert embedder.generated == 3  # the duplicate text is only embedded once
    embedder.encode(TEXTS)
    assert embedder.generated == 3 and embedder.cache_hits >= 4


def test_progress_callback_and_empty_input(embedder):
    seen = []
    embedder.encode(TEXTS, batch_size=2, progress=lambda done, total: seen.append((done, total)))
    assert seen[-1] == (3, 3)
    assert embedder.encode([]).shape == (0, embedder.dim)


def test_similar_text_scores_higher(embedder):
    q, near, far = embedder.encode(["How much is the stipend?", "The stipend is $500 for remote staff.", "Volcanoes erupt lava."])
    assert float(q @ near) > float(q @ far)


def test_loader_falls_back_when_model_cannot_load(monkeypatch):
    def boom(self):
        raise OSError("no network")
    monkeypatch.setattr(embeddings.SentenceTransformerEmbedder, "load", boom)
    embedder, warning = load_embedder("some/model")
    assert embedder.is_fallback and "fallback" in warning.lower()
