import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import RAGSettings  # noqa: E402
from rag.embeddings import HashingEmbedder  # noqa: E402
from rag.generator import ExtractiveGenerator  # noqa: E402
from rag.pipeline import DocuMindPipeline  # noqa: E402
from scripts.make_samples import main as make_samples  # noqa: E402


@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory) -> Path:
    """The sample repository (MD, DOCX, TXT, PDF) generated into a temp folder."""
    return make_samples(tmp_path_factory.mktemp("samples"))


@pytest.fixture
def sample_files(sample_dir) -> dict:
    return {p.name: p for p in sorted(sample_dir.iterdir())}


@pytest.fixture
def settings() -> RAGSettings:
    # The hashing embedder gives lower cosine scores than MiniLM, so tests use a lower threshold.
    return RAGSettings(similarity_threshold=0.12)


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def make_pipeline(tmp_path):
    def factory(name: str = "data") -> DocuMindPipeline:
        return DocuMindPipeline(tmp_path / name, embedder=HashingEmbedder(), generator=ExtractiveGenerator())
    return factory


@pytest.fixture
def pipeline(make_pipeline) -> DocuMindPipeline:
    return make_pipeline()


@pytest.fixture
def loaded_pipeline(pipeline, sample_files, settings) -> DocuMindPipeline:
    results = pipeline.ingest_files(list(sample_files.values()), settings)
    assert all(r.status == "indexed" for r in results), results
    return pipeline
