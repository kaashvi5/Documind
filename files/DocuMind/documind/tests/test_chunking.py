import pytest

from rag.chunking import chunk_pages, chunk_text, split_sentences
from rag.loaders import Page

TEXT = " ".join(f"Sentence number {i} talks about topic {i % 4} in some detail." for i in range(60))


def test_chunks_respect_size_and_cover_all_text():
    chunks = chunk_text(TEXT, 300, 80)
    assert len(chunks) > 5 and all(len(c) <= 300 for c in chunks)
    joined = " ".join(chunks)
    assert all(f"Sentence number {i} " in joined for i in range(60))


def test_neighbouring_chunks_overlap():
    chunks = chunk_text(TEXT, 300, 100)
    for left, right in zip(chunks, chunks[1:]):
        assert left.split(". ")[-1][:20] in right  # last sentence of left reappears in right


def test_zero_overlap_has_no_repetition():
    chunks = chunk_text(TEXT, 300, 0)
    assert sum(len(c) for c in chunks) <= len(TEXT)
    assert len(set(chunks)) == len(chunks)


@pytest.mark.parametrize("size,overlap", [(50, 0), (300, 300), (300, -1), (200, 250)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_text(TEXT, size, overlap)


def test_long_sentence_is_hard_split():
    chunks = chunk_text("word " * 400 + "end.", 200, 20)
    assert len(chunks) > 5 and all(len(c) <= 200 for c in chunks)
    assert all(len(c) <= 200 for c in chunk_text("x" * 1000, 200, 20))


def test_empty_text_gives_no_chunks():
    assert chunk_text("", 300, 50) == [] and chunk_text("   \n\n ", 300, 50) == []


def test_sentence_splitter_keeps_paragraphs_apart():
    assert split_sentences("First one. Second one.\n\nThird one!") == ["First one.", "Second one.", "Third one!"]


def test_page_numbers_and_indices_are_preserved():
    pages = [Page(1, TEXT[:600]), Page(2, TEXT[600:1400]), Page(None, TEXT[1400:])]
    chunks = chunk_pages(pages, 250, 50)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert {c.page for c in chunks} == {1, 2, None}
    assert [c.page for c in chunks] == sorted((c.page for c in chunks), key=lambda p: (p is None, p or 0))


def test_tiny_fragments_are_dropped():
    assert chunk_pages([Page(1, "12"), Page(2, "-- ..")], 200, 20) == []
