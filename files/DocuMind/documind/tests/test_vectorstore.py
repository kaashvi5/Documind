import json

import numpy as np
import pytest

from rag.vectorstore import META_FILE, VectorStore, VectorStoreError

DIM = 8


def unit(*values):
    vec = np.zeros(DIM, dtype=np.float32)
    vec[: len(values)] = values
    return vec / np.linalg.norm(vec)


def add(store, doc_id, name, vectors):
    chunks = [{"page": i + 1, "chunk_index": i, "text": f"{name} chunk {i}"} for i in range(len(vectors))]
    store.add_document({"doc_id": doc_id, "filename": name, "added_at": f"2024-01-0{len(store.documents) + 1}"}, chunks, np.stack(vectors))


@pytest.fixture
def store(tmp_path):
    s = VectorStore(tmp_path / "db", DIM, "test-model")
    add(s, "d1", "a.txt", [unit(1, 0), unit(0.9, 0.1)])
    add(s, "d2", "b.txt", [unit(0, 1), unit(0.1, 0.9)])
    return s


def test_search_returns_best_first_with_metadata(store):
    hits = store.search(unit(1, 0.05), 3)
    assert hits[0][0]["filename"] == "a.txt" and hits[0][1] >= hits[1][1] >= hits[2][1]
    assert {"id", "doc_id", "page", "chunk_index", "text"} <= set(hits[0][0])


def test_document_filter(store):
    hits = store.search(unit(1, 0), 4, doc_ids={"d2"})
    assert hits and all(h[0]["doc_id"] == "d2" for h in hits)


def test_delete_removes_vectors_and_metadata(store):
    assert store.vector_count == 4
    assert store.delete_document("d1") == 2
    assert store.vector_count == 2 and len(store.chunks) == 2 and "d1" not in store.documents
    assert all(h[0]["doc_id"] == "d2" for h in store.search(unit(1, 0), 10))
    assert store.delete_document("d1") == 0


def test_clear(store):
    store.clear()
    assert store.vector_count == 0 and not store.documents and store.search(unit(1), 3) == []


def test_persistence_round_trip(store, tmp_path):
    reloaded = VectorStore(tmp_path / "db", DIM, "test-model")
    assert reloaded.vector_count == 4 and set(reloaded.documents) == {"d1", "d2"}
    assert reloaded.search(unit(0, 1), 1)[0][0]["filename"] == "b.txt"
    add(reloaded, "d3", "c.txt", [unit(0, 0, 1)])
    assert reloaded.next_id == 6  # ids keep growing: no id is ever reused


def test_duplicate_and_bad_shapes_are_rejected(store):
    with pytest.raises(VectorStoreError, match="already"):
        add(store, "d1", "a.txt", [unit(1)])
    with pytest.raises(VectorStoreError, match="shape"):
        store.add_document({"doc_id": "x", "filename": "x"}, [{"page": 1, "chunk_index": 0, "text": "t"}], np.zeros((2, DIM), dtype=np.float32))
    with pytest.raises(VectorStoreError, match="dims"):
        store.search(np.ones(3, dtype=np.float32), 2)


def test_stale_records_are_reconciled_on_load(store, tmp_path):
    meta_path = tmp_path / "db" / META_FILE
    meta = json.loads(meta_path.read_text())
    meta["chunks"].pop(next(iter(meta["chunks"])))  # metadata lost for one vector
    meta_path.write_text(json.dumps(meta))
    reloaded = VectorStore(tmp_path / "db", DIM, "test-model")
    assert reloaded.vector_count == 3 and len(reloaded.chunks) == 3 and reloaded.load_warning


def test_corrupted_files_reset_the_store(store, tmp_path):
    (tmp_path / "db" / "index.faiss").write_bytes(b"garbage")
    reloaded = VectorStore(tmp_path / "db", DIM, "test-model")
    assert reloaded.vector_count == 0 and "unreadable" in reloaded.load_warning


def test_model_change_exposes_stale_documents(store, tmp_path):
    other = VectorStore(tmp_path / "db", DIM, "another-model")
    assert other.vector_count == 0 and len(other.stale_documents) == 2
