import shutil

import pytest

from config import NOT_FOUND_MESSAGE, RAGSettings
from rag.pipeline import safe_filename


def test_multiple_formats_are_indexed(loaded_pipeline):
    docs = loaded_pipeline.list_documents()
    assert {d["file_type"] for d in docs} == {"md", "docx", "txt", "pdf"} and len(docs) == 5
    stats = loaded_pipeline.stats()
    assert stats["documents"] == 5 and stats["chunks"] == stats["vectors"] > 5 and stats["embeddings_generated"] > 0
    pdf = next(d for d in docs if d["file_type"] == "pdf")
    assert pdf["pages"] == 3 and pdf["chunks"] >= 3 and pdf["size_bytes"] > 0


def test_duplicate_content_is_not_indexed_twice(pipeline, sample_files, settings, tmp_path):
    first = pipeline.ingest_file(sample_files["03_renewable_energy_overview.txt"], settings)
    copy = tmp_path / "copy_of_energy.txt"
    shutil.copy(sample_files["03_renewable_energy_overview.txt"], copy)
    second = pipeline.ingest_file(copy, settings)
    assert first.status == "indexed" and second.status == "duplicate" and "03_renewable" in second.message
    assert pipeline.stats()["documents"] == 1 and pipeline.stats()["vectors"] == first.chunks


def test_same_filename_new_content_replaces_old_version(pipeline, tmp_path, settings):
    path = tmp_path / "notes.txt"
    path.write_text("The first version talks about apples and orchards in great detail every autumn.")
    old = pipeline.ingest_file(path, settings)
    path.write_text("The second version talks about bananas and plantations in great detail every winter.")
    new = pipeline.ingest_file(path, settings)
    assert new.status == "replaced" and new.doc_id != old.doc_id
    docs = pipeline.list_documents()
    assert len(docs) == 1 and docs[0]["doc_id"] == new.doc_id
    assert all("apples" not in c["text"] for c in pipeline.store.chunks.values())
    assert not (pipeline.upload_dir / old_stored(old, "notes.txt")).exists()


def old_stored(result, name):
    return f"{result.doc_id}_{safe_filename(name)}"


def test_bad_files_do_not_break_the_batch(pipeline, sample_files, settings, tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    exe = tmp_path / "virus.exe"
    exe.write_bytes(b"MZ")
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4 nothing useful")
    progress = []
    results = pipeline.ingest_files([empty, sample_files["05_rag_overview.md"], exe, broken], settings,
                                    progress=lambda f, t: progress.append(f))
    assert [r.status for r in results] == ["failed", "indexed", "failed", "failed"]
    assert all(r.message for r in results if r.status == "failed")
    assert pipeline.stats()["documents"] == 1 and progress[-1] == 1.0


def test_upload_limit_per_batch(pipeline, tmp_path, settings, monkeypatch):
    monkeypatch.setattr("rag.pipeline.MAX_FILES_PER_UPLOAD", 2)
    paths = []
    for i in range(3):
        p = tmp_path / f"f{i}.txt"
        p.write_text(f"Document {i} discusses entirely different subject number {i} with enough words to index.")
        paths.append(p)
    assert [r.status for r in pipeline.ingest_files(paths, settings)] == ["indexed", "indexed", "failed"]


def test_ask_returns_grounded_answer_with_citations(loaded_pipeline, settings):
    answer = loaded_pipeline.ask("How much is the home-office stipend?", settings)
    assert answer.kind == "answer" and answer.found and "$500" in answer.text
    top = answer.sources[0]
    assert top.filename == "02_remote_work_policy.docx" and top.cited and top.number == 1
    assert 0 < top.score <= 1 and "$500" in top.text and top.page is None
    assert answer.model and answer.seconds >= 0


def test_pdf_answers_cite_the_page(loaded_pipeline, settings):
    answer = loaded_pipeline.ask("Where does the Calvin cycle take place?", settings)
    assert answer.found and "stroma" in answer.text.lower()
    cited = [s for s in answer.sources if s.cited]
    assert cited[0].filename.endswith(".pdf") and cited[0].page == 2


def test_unsupported_question_gets_honest_not_found(loaded_pipeline):
    answer = loaded_pipeline.ask("Who won the football world cup in 2018?", RAGSettings())
    assert answer.kind == "not_found" and not answer.found and answer.text == NOT_FOUND_MESSAGE


def test_empty_repository_and_invalid_queries(make_pipeline, loaded_pipeline):
    assert make_pipeline("empty").ask("What is RAG?").kind == "empty"
    assert loaded_pipeline.ask("").kind == "invalid" and loaded_pipeline.ask("??").kind == "invalid"
    assert loaded_pipeline.ask("x" * 5000).kind == "invalid"


def test_follow_up_question_uses_history(loaded_pipeline, settings):
    answer = loaded_pipeline.ask("What about the internet?", settings, history=["How much is the home-office stipend?"])
    assert "regarding" in answer.standalone_question and answer.found and "$40" in answer.text


def test_document_filter_limits_sources(loaded_pipeline, settings):
    docs = {d["filename"]: d["doc_id"] for d in loaded_pipeline.list_documents()}
    only = RAGSettings(similarity_threshold=0.0, top_k=4, doc_ids=(docs["03_renewable_energy_overview.txt"],))
    answer = loaded_pipeline.ask("What is overfitting?", only)
    assert {s.filename for s in answer.sources} <= {"03_renewable_energy_overview.txt"}


def test_delete_removes_everything_for_that_document(loaded_pipeline, settings):
    doc = next(d for d in loaded_pipeline.list_documents() if d["file_type"] == "docx")
    before = loaded_pipeline.stats()["vectors"]
    assert loaded_pipeline.delete_document(doc["doc_id"]) and not loaded_pipeline.delete_document(doc["doc_id"])
    assert loaded_pipeline.stats()["vectors"] == before - doc["chunks"]
    assert all(c["doc_id"] != doc["doc_id"] for c in loaded_pipeline.store.chunks.values())
    assert not (loaded_pipeline.upload_dir / doc["stored_as"]).exists()
    assert loaded_pipeline.ask("How much is the home-office stipend?", settings).kind == "not_found"


def test_clear_repository(loaded_pipeline):
    assert loaded_pipeline.clear_repository() == 5
    assert loaded_pipeline.stats()["documents"] == 0 and loaded_pipeline.stats()["vectors"] == 0
    assert not [p for p in loaded_pipeline.upload_dir.iterdir() if p.name != ".gitkeep"]


def test_state_survives_restart(loaded_pipeline, make_pipeline, settings):
    restarted = make_pipeline()  # same data dir
    assert restarted.stats()["documents"] == 5 and restarted.stats()["vectors"] == loaded_pipeline.stats()["vectors"]
    assert "$500" in restarted.ask("How much is the home-office stipend?", settings).text
    assert restarted.ingest_file(next(restarted.upload_dir.iterdir()), settings).status == "duplicate"


def test_reindex_applies_new_chunk_settings(loaded_pipeline, settings):
    before = loaded_pipeline.stats()["chunks"]
    results = loaded_pipeline.reindex_all(RAGSettings(chunk_size=250, chunk_overlap=40, similarity_threshold=0.12))
    assert all(r.status == "indexed" for r in results)
    stats = loaded_pipeline.stats()
    assert stats["documents"] == 5 and stats["chunks"] > before and stats["chunks"] == stats["vectors"]
    assert "$500" in loaded_pipeline.ask("How much is the home-office stipend?", settings).text


def test_invalid_settings_are_rejected(pipeline, sample_files):
    with pytest.raises(ValueError):
        pipeline.ingest_file(sample_files["05_rag_overview.md"], RAGSettings(chunk_size=50))
    with pytest.raises(ValueError):
        RAGSettings(chunk_size=300, chunk_overlap=300).validate()


def test_model_failure_degrades_to_extractive_answers(loaded_pipeline, settings):
    class Broken:
        name, is_fallback, loaded = "broken-model", False, True

        def generate(self, *a, **k):
            raise RuntimeError("CUDA error")

    loaded_pipeline.generator = Broken()
    answer = loaded_pipeline.ask("How much is the home-office stipend?", settings)
    assert answer.found and "$500" in answer.text and any("failed" in w for w in answer.warnings)


def test_generator_load_failure_is_reported(pipeline):
    class Unloadable:
        name, is_fallback, loaded = "ghost-model", False, False

        def load(self):
            raise OSError("not downloadable")

    pipeline.generator = Unloadable()
    pipeline.ensure_generator()
    assert pipeline.generator.name == "extractive-fallback" and any("ghost-model" in w for w in pipeline.warnings)


def test_safe_filename_blocks_path_tricks():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("my report (final).pdf") == "my_report_final_.pdf"
