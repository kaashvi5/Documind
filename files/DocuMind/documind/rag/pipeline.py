"""DocuMind pipeline: ingestion (upload -> FAISS) and question answering (question -> answer + citations)."""
from __future__ import annotations

import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from config import (DATA_DIR, DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATION_MODEL, MAX_FILES_PER_UPLOAD,
                    MAX_QUESTION_CHARS, MIN_QUESTION_CHARS, NOT_FOUND_MESSAGE, RAGSettings, get_device)
from rag.chunking import chunk_pages
from rag.embeddings import BaseEmbedder, load_embedder
from rag.generator import ExtractiveGenerator, FlanT5Generator, find_supporting_chunks
from rag.loaders import DocumentError, load_document
from rag.retrieval import Retriever, RetrievedChunk, contextualize_question
from rag.vectorstore import VectorStore, VectorStoreError

Progress = Callable[[float, str], None]


@dataclass
class IngestResult:
    filename: str
    status: str            # "indexed" | "replaced" | "duplicate" | "failed"
    message: str = ""
    doc_id: str = ""
    pages: int = 0
    chunks: int = 0
    chars: int = 0
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class Source:
    number: int
    filename: str
    page: int | None
    chunk_index: int
    text: str
    score: float
    final_score: float
    cited: bool


@dataclass
class Answer:
    question: str
    text: str
    kind: str                       # "answer" | "not_found" | "empty" | "invalid" | "error"
    found: bool = False
    sources: list[Source] = field(default_factory=list)
    standalone_question: str = ""
    model: str = ""
    seconds: float = 0.0
    low_confidence: bool = False
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def safe_filename(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name).strip("._") or "document"
    return base[-120:]


class DocuMindPipeline:
    """Coordinates loaders, chunking, embeddings, the vector store, retrieval and generation."""

    def __init__(self, data_dir: str | Path | None = None, embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                 generation_model: str = DEFAULT_GENERATION_MODEL, device: str | None = None,
                 embedder: BaseEmbedder | None = None, generator=None):
        self.data_dir = Path(data_dir or DATA_DIR)
        self.upload_dir = self.data_dir / "uploads"
        self.vector_dir = self.data_dir / "vector_db"
        for folder in (self.upload_dir, self.vector_dir):
            folder.mkdir(parents=True, exist_ok=True)
        self.device = get_device(device)
        self.warnings: list[str] = []
        self.busy = ""  # human-readable activity ("Indexing 3 files..."); empty when idle
        self._lock = threading.RLock()

        if embedder is None:
            embedder, warning = load_embedder(embedding_model, self.device)
            if warning:
                self.warnings.append(warning)
        self.embedder = embedder
        self.generator = generator or FlanT5Generator(generation_model, self.device)
        self.store = VectorStore(self.vector_dir, self.embedder.dim, self.embedder.name)
        self.retriever = Retriever(self.embedder, self.store)
        if self.store.load_warning:
            self.warnings.append(self.store.load_warning)
        if self.store.stale_documents:
            self._reindex_records(self.store.stale_documents, RAGSettings())

    # ---------------------------------------------------------------- models
    @property
    def generator_ready(self) -> bool:
        return bool(getattr(self.generator, "loaded", True))

    def load_models(self, progress: Progress | None = None) -> list[str]:
        """Load the generator now (instead of on the first question). Returns warnings."""
        if progress:
            progress(0.1, f"Loading {self.generator.name}...")
        self.ensure_generator()
        if progress:
            progress(1.0, "Models ready")
        return list(self.warnings)

    def ensure_generator(self) -> None:
        with self._lock:
            if self.generator_ready:
                return
            try:
                self.generator.load()
            except Exception as exc:
                self.warnings.append(f"Could not load '{self.generator.name}' ({type(exc).__name__}: {exc}). "
                                     "Answers fall back to extractive mode (copied passages).")
                self.generator = ExtractiveGenerator()

    def set_generation_model(self, name: str) -> str:
        """Switch the LLM. Keeps the current one if the new model cannot be loaded."""
        with self._lock:
            if name == getattr(self.generator, "name", ""):
                return f"{name} is already active."
            candidate = FlanT5Generator(name, self.device)
            try:
                candidate.load()
            except Exception as exc:
                return f"Could not load {name}: {type(exc).__name__}: {exc}. Keeping {self.generator.name}."
            self.generator = candidate
            return f"Switched generation model to {name}."

    def set_embedding_model(self, name: str, settings: RAGSettings | None = None,
                            progress: Progress | None = None) -> str:
        """Switch the embedding model and rebuild the index for all stored documents."""
        with self._lock:
            if name == self.embedder.name:
                return f"{name} is already active."
            embedder, warning = load_embedder(name, self.device)
            if warning:
                return warning + " The previous model stays active."
            records = self.store.list_documents()
            self.embedder = embedder
            self.retriever.embedder = embedder
            self.store.reset(embedder.dim, embedder.name)
            self._reindex_records(records, settings or RAGSettings(), progress)
            return f"Switched embedding model to {name} and re-indexed {len(records)} document(s)."

    def status(self) -> dict:
        return {"embedding_model": self.embedder.name, "embedding_fallback": self.embedder.is_fallback,
                "generation_model": getattr(self.generator, "name", "?"),
                "generation_fallback": getattr(self.generator, "is_fallback", False),
                "generator_loaded": self.generator_ready, "device": self.device,
                "embeddings_generated": self.embedder.generated, "cache_hits": self.embedder.cache_hits,
                "busy": self.busy, "warnings": list(self.warnings)}

    # ------------------------------------------------------------- ingestion
    def ingest_files(self, paths: list[str | Path], settings: RAGSettings | None = None,
                     progress: Progress | None = None) -> list[IngestResult]:
        settings = (settings or RAGSettings()).validate()
        results: list[IngestResult] = []
        paths = list(paths or [])
        total = max(len(paths), 1)
        for position, path in enumerate(paths):
            name = Path(path).name
            if position >= MAX_FILES_PER_UPLOAD:
                results.append(IngestResult(name, "failed", f"Skipped: at most {MAX_FILES_PER_UPLOAD} files per upload."))
                continue
            lo, hi = position / total, (position + 1) / total

            def step(fraction: float, text: str, lo=lo, hi=hi, name=name) -> None:
                if progress:
                    progress(lo + (hi - lo) * fraction, f"{name}: {text}")

            results.append(self.ingest_file(path, settings, progress=step))
        if progress:
            progress(1.0, "Done")
        return results

    def ingest_file(self, path: str | Path, settings: RAGSettings | None = None, display_name: str | None = None,
                    progress: Progress | None = None) -> IngestResult:
        """Run one file through Extract -> Clean -> Chunk -> Metadata -> Embed -> FAISS."""
        settings = (settings or RAGSettings()).validate()
        name = display_name or Path(path).name
        started = time.time()
        step = progress or (lambda fraction, text: None)
        try:
            step(0.05, "extracting text")
            doc = load_document(path, display_name=name, use_ocr=settings.use_ocr)
        except DocumentError as exc:
            return IngestResult(name, "failed", str(exc), seconds=time.time() - started)
        doc_id = doc.sha256[:16]

        with self._lock:
            existing = self.store.documents.get(doc_id)
            if existing:
                return IngestResult(name, "duplicate", f"Already indexed as '{existing['filename']}'.", doc_id,
                                    seconds=time.time() - started)
            step(0.2, "splitting into chunks")
            chunks = chunk_pages(doc.pages, settings.chunk_size, settings.chunk_overlap)
            if not chunks:
                return IngestResult(name, "failed", "The document has too little text to index.", doc_id,
                                    seconds=time.time() - started)
            try:
                vectors = self.embedder.encode(
                    [c.text for c in chunks],
                    progress=lambda done, total: step(0.25 + 0.6 * done / max(total, 1), f"embedding {done}/{total}"))
            except Exception as exc:
                return IngestResult(name, "failed", f"Embedding failed ({type(exc).__name__}: {exc}).", doc_id,
                                    seconds=time.time() - started)

            stored_as = f"{doc_id}_{safe_filename(name)}"
            dest = self.upload_dir / stored_as
            try:
                if Path(path).resolve() != dest.resolve():
                    shutil.copy2(path, dest)
                step(0.9, "saving to the vector database")
                previous = self.store.find_by_filename(name)
                record = {"doc_id": doc_id, "filename": name, "file_type": doc.file_type,
                          "size_bytes": doc.size_bytes, "pages": doc.total_pages, "chars": doc.char_count,
                          "stored_as": stored_as, "added_at": datetime.now(timezone.utc).isoformat(),
                          "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap}
                chunk_dicts = [{"page": c.page, "chunk_index": c.chunk_index, "text": c.text} for c in chunks]
                self.store.add_document(record, chunk_dicts, vectors)
                if previous:  # same filename, new content: replace the stale version
                    self._remove_files(previous)
                    self.store.delete_document(previous["doc_id"])
            except (VectorStoreError, OSError) as exc:
                if dest.exists() and not self.store.has_document(doc_id):
                    dest.unlink(missing_ok=True)
                return IngestResult(name, "failed", f"Could not store the document: {exc}", doc_id,
                                    seconds=time.time() - started)
        status = "replaced" if previous else "indexed"
        message = "Replaced the previous version of this file." if previous else "Indexed."
        return IngestResult(name, status, message, doc_id, doc.total_pages, len(chunks), doc.char_count,
                            time.time() - started, doc.warnings)

    def _remove_files(self, record: dict) -> None:
        stored = record.get("stored_as")
        if stored:
            (self.upload_dir / stored).unlink(missing_ok=True)

    def _reindex_records(self, records: list[dict], settings: RAGSettings,
                         progress: Progress | None = None) -> list[IngestResult]:
        results = []
        total = max(len(records), 1)
        for position, record in enumerate(records):
            path = self.upload_dir / record.get("stored_as", "")
            if not path.is_file():
                results.append(IngestResult(record["filename"], "failed", "The stored original file is missing."))
                continue
            if progress:
                progress(position / total, f"Re-indexing {record['filename']}")
            results.append(self.ingest_file(path, settings, display_name=record["filename"]))
        return results

    def reindex_all(self, settings: RAGSettings | None = None, progress: Progress | None = None) -> list[IngestResult]:
        """Re-chunk and re-embed every stored document with the given settings."""
        settings = (settings or RAGSettings()).validate()
        with self._lock:
            records = self.store.list_documents()
            self.store.reset()
            return self._reindex_records(records, settings, progress)

    # ------------------------------------------------------------ management
    def list_documents(self) -> list[dict]:
        return self.store.list_documents()

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            record = self.store.documents.get(doc_id)
            if not record:
                return False
            self.store.delete_document(doc_id)
            self._remove_files(record)
            return True

    def clear_repository(self) -> int:
        with self._lock:
            count = len(self.store.documents)
            self.store.clear()
            for path in self.upload_dir.glob("*"):
                if path.is_file() and path.name != ".gitkeep":
                    path.unlink(missing_ok=True)
            return count

    def stats(self) -> dict:
        return {**self.store.stats(), **self.status()}

    # -------------------------------------------------------------- question
    def ask(self, question: str, settings: RAGSettings | None = None, history: list[str] | None = None) -> Answer:
        """Answer a question from the indexed documents, with citations."""
        started = time.time()
        question = (question or "").strip()
        if len(question) < MIN_QUESTION_CHARS or not any(ch.isalnum() for ch in question):
            return Answer(question, "Please enter a question of at least a few words.", "invalid")
        if len(question) > MAX_QUESTION_CHARS:
            return Answer(question[:80] + "...", f"That question is too long (limit {MAX_QUESTION_CHARS} characters).", "invalid")
        if self.store.vector_count == 0:
            return Answer(question, "The repository is empty. Upload documents (or load the sample set) first.", "empty")
        settings = (settings or RAGSettings()).validate()

        standalone = contextualize_question(question, history)
        retrieved = self.retriever.retrieve(standalone, settings.top_k, settings.similarity_threshold,
                                            settings.keyword_rerank, set(settings.doc_ids) or None)
        if not retrieved:
            return Answer(question, NOT_FOUND_MESSAGE, "not_found", standalone_question=standalone,
                          seconds=time.time() - started)

        warnings: list[str] = []
        self.ensure_generator()
        try:
            result = self.generator.generate(standalone, retrieved, detailed=settings.detailed_answers)
        except Exception as exc:  # model crashed: degrade to extractive answers instead of failing
            warnings.append(f"The language model failed ({type(exc).__name__}); showing extracted passages instead.")
            result = ExtractiveGenerator().generate(standalone, retrieved, detailed=settings.detailed_answers)
        used = retrieved[: max(result.chunks_used, 1)]
        model = getattr(result, "model", "") or getattr(self.generator, "name", "")

        if not result.found:
            return Answer(question, NOT_FOUND_MESSAGE, "not_found", False, self._sources(used, set()),
                          standalone, model, time.time() - started, truncated=result.truncated, warnings=warnings)

        cited, coverage = find_supporting_chunks(result.answer, [c.text for c in used])
        text = re.sub(r"\s*\[\d{1,2}\]", "", result.answer).strip()
        low = len(text.split()) >= 4 and coverage < 0.25
        return Answer(question, text, "answer", True, self._sources(used, set(cited)), standalone, model,
                      time.time() - started, low_confidence=low, truncated=result.truncated, warnings=warnings)

    @staticmethod
    def _sources(chunks: list[RetrievedChunk], cited: set[int]) -> list[Source]:
        return [Source(i, c.filename, c.page, c.chunk_index, c.text, round(c.score, 4), round(c.final_score, 4), i in cited)
                for i, c in enumerate(chunks, start=1)]
