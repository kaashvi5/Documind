"""Central configuration for DocuMind: paths, limits, model choices and RAG defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(os.environ.get("DOCUMIND_HOME") or Path(__file__).resolve().parent)
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_DIR = DATA_DIR / "vector_db"
SAMPLE_DIR = BASE_DIR / "sample_docs"

ALLOWED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md", ".markdown")
MAX_FILE_MB = 15
MAX_FILES_PER_UPLOAD = 25
MIN_QUESTION_CHARS = 3
MAX_QUESTION_CHARS = 1000

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": "MiniLM-L6 (22M params, fast, default)",
    "sentence-transformers/all-MiniLM-L12-v2": "MiniLM-L12 (33M params, slightly more accurate)",
    "BAAI/bge-small-en-v1.5": "BGE-small (33M params, strong retrieval quality)",
}

DEFAULT_GENERATION_MODEL = "google/flan-t5-base"
GENERATION_MODELS = {
    "google/flan-t5-small": "FLAN-T5 Small (77M) - fastest, weakest answers",
    "google/flan-t5-base": "FLAN-T5 Base (248M) - recommended for free Colab",
    "google/flan-t5-large": "FLAN-T5 Large (783M) - best quality, needs a GPU runtime",
}

# Max prompt size (in tokens) sent to the generator. FLAN-T5 was trained on 512 tokens
# but tolerates somewhat longer inputs; 768 keeps quality and memory in balance.
MAX_INPUT_TOKENS = 768

NOT_FOUND_MESSAGE = "I couldn't find this information in the uploaded documents."


@dataclass
class RAGSettings:
    """User-tunable settings. Passed explicitly to every pipeline call (no hidden state)."""

    chunk_size: int = 600          # characters per chunk
    chunk_overlap: int = 100       # characters shared between neighbouring chunks
    top_k: int = 4                 # chunks passed to the LLM
    similarity_threshold: float = 0.25  # minimum cosine similarity to keep a chunk
    keyword_rerank: bool = True    # blend keyword coverage into the ranking
    detailed_answers: bool = False  # longer answers
    use_ocr: bool = False          # OCR scanned PDFs (needs Tesseract)
    doc_ids: tuple = ()            # restrict retrieval to these documents (empty = all)

    def validate(self) -> "RAGSettings":
        if not 100 <= int(self.chunk_size) <= 4000:
            raise ValueError("Chunk size must be between 100 and 4000 characters.")
        if not 0 <= int(self.chunk_overlap) < int(self.chunk_size):
            raise ValueError("Chunk overlap must be at least 0 and smaller than the chunk size.")
        self.chunk_size = int(self.chunk_size)
        self.chunk_overlap = int(self.chunk_overlap)
        self.top_k = max(1, min(20, int(self.top_k)))
        self.similarity_threshold = max(0.0, min(1.0, float(self.similarity_threshold)))
        self.doc_ids = tuple(self.doc_ids or ())
        return self


def get_device(prefer: str | None = None) -> str:
    """Return 'cuda' when a GPU is available, otherwise 'cpu' (or the explicit choice)."""
    if prefer and prefer != "auto":
        return prefer
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # torch missing or broken: CPU-only paths still work
        return "cpu"


def ensure_dirs() -> None:
    for folder in (UPLOAD_DIR, VECTOR_DIR):
        folder.mkdir(parents=True, exist_ok=True)
