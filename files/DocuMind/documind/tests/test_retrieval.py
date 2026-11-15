import pytest

from rag.retrieval import Retriever, contextualize_question

QUESTIONS = [
    ("How much is the home-office stipend?", "02_remote_work_policy.docx"),
    ("Where does the Calvin cycle take place?", "04_photosynthesis_and_plant_biology.pdf"),
    ("What is overfitting?", "01_intro_to_machine_learning.md"),
    ("What is the capacity factor of offshore wind farms?", "03_renewable_energy_overview.txt"),
    ("What does a vector database such as FAISS do?", "05_rag_overview.md"),
]


@pytest.mark.parametrize("question,expected", QUESTIONS)
def test_relevant_document_is_ranked_first(loaded_pipeline, question, expected):
    results = loaded_pipeline.retriever.retrieve(question, top_k=3, threshold=0.1)
    assert results and results[0].filename == expected
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_source_metadata_is_preserved(loaded_pipeline):
    top = loaded_pipeline.retriever.retrieve("Where does the Calvin cycle take place?", top_k=1, threshold=0.1)[0]
    assert top.page == 2 and "stroma" in top.text and 0 < top.score <= 1 and top.doc_id and top.chunk_id > 0


def test_top_k_is_respected(loaded_pipeline):
    assert len(loaded_pipeline.retriever.retrieve("energy", top_k=2, threshold=0.0)) <= 2
    assert len(loaded_pipeline.retriever.retrieve("energy", top_k=6, threshold=0.0)) <= 6


def test_threshold_filters_irrelevant_chunks(loaded_pipeline):
    question = "Who won the football world cup in 2018?"
    assert loaded_pipeline.retriever.retrieve(question, top_k=4, threshold=0.5) == []
    assert all(r.score >= 0.3 for r in loaded_pipeline.retriever.retrieve("How much is the stipend?", top_k=4, threshold=0.3))


def test_rerank_blends_keyword_coverage(loaded_pipeline):
    plain = loaded_pipeline.retriever.retrieve("home office stipend amount", 3, 0.0, rerank=False)
    mixed = loaded_pipeline.retriever.retrieve("home office stipend amount", 3, 0.0, rerank=True)
    assert all(abs(r.final_score - r.score) < 1e-9 for r in plain)
    assert any(abs(r.final_score - r.score) > 1e-6 for r in mixed) and mixed[0].keyword_score > 0


def test_document_filter(loaded_pipeline):
    pdf = next(d for d in loaded_pipeline.list_documents() if d["filename"].endswith(".pdf"))
    results = loaded_pipeline.retriever.retrieve("energy", top_k=4, threshold=0.0, doc_ids={pdf["doc_id"]})
    assert results and {r.filename for r in results} == {pdf["filename"]}


def test_empty_query_and_empty_store(loaded_pipeline, make_pipeline):
    with pytest.raises(ValueError):
        loaded_pipeline.retriever.retrieve("   ")
    assert make_pipeline("empty").retriever.retrieve("anything", 3, 0.0) == []


def test_follow_up_questions_are_made_standalone():
    history = ["How much is the home-office stipend?"]
    assert "regarding: How much is the home-office stipend?" in contextualize_question("What about the internet?", history)
    assert "regarding" in contextualize_question("Is it taxable?", history)
    full = "Which days are the anchor days in the office?"
    assert contextualize_question(full, history) == full
    assert contextualize_question("What about it?", []) == "What about it?"
