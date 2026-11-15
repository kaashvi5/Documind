"""FAISS vector store with separate JSON metadata, document registry and atomic persistence."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

INDEX_FILE = "index.faiss"
META_FILE = "metadata.json"
FORMAT_VERSION = 1


class VectorStoreError(Exception):
    """Raised when the vector database cannot complete an operation."""


class VectorStore:
    """Exact cosine search (IndexFlatIP on normalised vectors) wrapped in IndexIDMap2.

    Chunk ids are stable integers, so deleting a document removes exactly its vectors.
    Metadata (filename, page, text, ...) lives in a plain dict persisted as JSON next to the index.
    """

    def __init__(self, directory: str | Path, dim: int, embedding_model: str = "", autosave: bool = True):
        self.directory = Path(directory)
        self.dim = int(dim)
        self.embedding_model = embedding_model
        self.autosave = autosave
        self.load_warning: str | None = None
        self.stale_documents: list[dict] = []  # documents of an index built with another model
        self._lock = threading.RLock()
        self._reset_state()
        self._load()

    # ------------------------------------------------------------------ state
    def _reset_state(self) -> None:
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
        self.chunks: dict[int, dict] = {}
        self.documents: dict[str, dict] = {}
        self.next_id = 1

    def reset(self, dim: int | None = None, embedding_model: str | None = None) -> None:
        """Drop everything (optionally switching dimension / model) and persist the empty state."""
        with self._lock:
            if dim is not None:
                self.dim = int(dim)
            if embedding_model is not None:
                self.embedding_model = embedding_model
            self._reset_state()
            self._save_if_needed()

    @property
    def vector_count(self) -> int:
        return int(self.index.ntotal)

    def has_document(self, doc_id: str) -> bool:
        return doc_id in self.documents

    def find_by_filename(self, filename: str) -> dict | None:
        for record in self.documents.values():
            if record["filename"] == filename:
                return record
        return None

    def list_documents(self) -> list[dict]:
        return sorted((dict(r) for r in self.documents.values()), key=lambda r: r["added_at"])

    # --------------------------------------------------------------- mutation
    def add_document(self, record: dict, chunks: list[dict], vectors: np.ndarray) -> None:
        """Store a document's chunks and vectors atomically (all or nothing)."""
        with self._lock:
            doc_id = record["doc_id"]
            if doc_id in self.documents:
                raise VectorStoreError("This document is already indexed.")
            vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape != (len(chunks), self.dim):
                raise VectorStoreError(
                    f"Embedding shape {vectors.shape} does not match {len(chunks)} chunks x {self.dim} dims.")
            if not chunks:
                raise VectorStoreError("Cannot index a document without chunks.")
            ids = np.arange(self.next_id, self.next_id + len(chunks), dtype=np.int64)
            try:
                self.index.add_with_ids(vectors, ids)
            except Exception as exc:
                raise VectorStoreError(f"FAISS rejected the vectors: {exc}") from exc
            for chunk_id, chunk in zip(ids.tolist(), chunks):
                self.chunks[chunk_id] = {**chunk, "id": chunk_id, "doc_id": doc_id, "filename": record["filename"]}
            self.next_id += len(chunks)
            self.documents[doc_id] = {**record, "chunks": len(chunks),
                                      "added_at": record.get("added_at") or datetime.now(timezone.utc).isoformat()}
            self._save_if_needed()

    def delete_document(self, doc_id: str) -> int:
        """Remove a document, its vectors and its metadata. Returns the number of chunks removed."""
        with self._lock:
            if doc_id not in self.documents:
                return 0
            ids = [cid for cid, meta in self.chunks.items() if meta["doc_id"] == doc_id]
            if ids:
                self.index.remove_ids(np.asarray(ids, dtype=np.int64))
            for cid in ids:
                del self.chunks[cid]
            del self.documents[doc_id]
            self._save_if_needed()
            return len(ids)

    def clear(self) -> None:
        self.reset()

    # ----------------------------------------------------------------- search
    def search(self, vector: np.ndarray, k: int, doc_ids: set[str] | None = None) -> list[tuple[dict, float]]:
        """Return up to k (metadata, cosine score) pairs, best first, optionally restricted to docs."""
        with self._lock:
            if self.index.ntotal == 0 or k <= 0:
                return []
            query = np.ascontiguousarray(np.asarray(vector, dtype=np.float32).reshape(1, -1))
            if query.shape[1] != self.dim:
                raise VectorStoreError(f"Query has {query.shape[1]} dims; the index expects {self.dim}.")
            fetch = self.index.ntotal if doc_ids else min(k, self.index.ntotal)
            scores, ids = self.index.search(query, int(fetch))
            hits: list[tuple[dict, float]] = []
            for score, cid in zip(scores[0], ids[0]):
                meta = self.chunks.get(int(cid)) if cid >= 0 else None
                if meta is None or (doc_ids and meta["doc_id"] not in doc_ids):
                    continue
                hits.append((dict(meta), float(score)))
                if len(hits) >= k:
                    break
            return hits

    # ------------------------------------------------------------ persistence
    def _save_if_needed(self) -> None:
        if self.autosave:
            self.save()

    def save(self) -> None:
        """Write index + metadata atomically (temp file, then rename)."""
        with self._lock:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                tmp_index = self.directory / (INDEX_FILE + ".tmp")
                tmp_meta = self.directory / (META_FILE + ".tmp")
                faiss.write_index(self.index, str(tmp_index))
                payload = {"version": FORMAT_VERSION, "dim": self.dim, "embedding_model": self.embedding_model,
                           "next_id": self.next_id, "documents": self.documents,
                           "chunks": {str(k): v for k, v in self.chunks.items()}}
                tmp_meta.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp_index, self.directory / INDEX_FILE)
                os.replace(tmp_meta, self.directory / META_FILE)
            except Exception as exc:
                raise VectorStoreError(f"Could not save the vector database: {exc}") from exc

    def _load(self) -> None:
        index_path, meta_path = self.directory / INDEX_FILE, self.directory / META_FILE
        if not (index_path.exists() and meta_path.exists()):
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("dim") != self.dim or meta.get("embedding_model") != self.embedding_model:
                self.stale_documents = list(meta.get("documents", {}).values())
                self.load_warning = (f"The saved index was built with '{meta.get('embedding_model')}'; "
                                     f"it will be rebuilt for '{self.embedding_model}'.")
                return
            index = faiss.read_index(str(index_path))
            if index.d != self.dim:
                raise ValueError("index dimension mismatch")
            self.index = index
            self.chunks = {int(k): v for k, v in meta["chunks"].items()}
            self.documents = dict(meta["documents"])
            self.next_id = int(meta["next_id"])
            self._reconcile()
        except Exception as exc:
            for path in (index_path, meta_path):
                if path.exists():
                    path.replace(path.with_suffix(path.suffix + ".corrupt"))
            self._reset_state()
            self.load_warning = f"The saved vector database was unreadable ({type(exc).__name__}) and was reset."

    def _reconcile(self) -> None:
        """Remove orphan vectors / metadata so the index and registry always agree."""
        in_index = {int(i) for i in faiss.vector_to_array(self.index.id_map)}
        orphans = in_index - set(self.chunks)
        if orphans:
            self.index.remove_ids(np.asarray(sorted(orphans), dtype=np.int64))
        for cid in set(self.chunks) - in_index:
            del self.chunks[cid]
        counts: dict[str, int] = {}
        for meta in self.chunks.values():
            counts[meta["doc_id"]] = counts.get(meta["doc_id"], 0) + 1
        self.documents = {d: {**r, "chunks": counts[d]} for d, r in self.documents.items() if d in counts}
        if orphans or len(self.chunks) != len(in_index):
            self.load_warning = "Stale records were found in the saved index and removed."

    def stats(self) -> dict:
        size = sum(p.stat().st_size for p in self.directory.glob("*") if p.is_file()) if self.directory.exists() else 0
        return {"documents": len(self.documents), "chunks": len(self.chunks), "vectors": self.vector_count,
                "characters": sum(int(r.get("chars", 0)) for r in self.documents.values()), "disk_bytes": size}
