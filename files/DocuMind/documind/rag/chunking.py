"""Sentence-aware overlapping chunking that preserves page numbers."""
from __future__ import annotations

import re
from dataclasses import dataclass

from rag.loaders import Page

_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[])")
MIN_CHUNK_ALNUM = 15


@dataclass
class Chunk:
    text: str
    page: int | None
    chunk_index: int  # position within the document (0-based)


def validate_params(chunk_size: int, overlap: int) -> None:
    if chunk_size < 100:
        raise ValueError("Chunk size must be at least 100 characters.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk overlap must be >= 0 and smaller than the chunk size.")


def split_sentences(text: str) -> list[str]:
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if paragraph:
            units.extend(s.strip() for s in _SENTENCE_END.split(paragraph) if s and s.strip())
    return units


def _hard_split(sentence: str, size: int) -> list[str]:
    """Split an over-long sentence on whitespace so no piece exceeds `size`."""
    pieces, current = [], ""
    for word in sentence.split():
        while len(word) > size:  # a single unbroken token longer than the limit
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:size])
            word = word[size:]
        if current and len(current) + 1 + len(word) > size:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    """Pack whole sentences into chunks of <= chunk_size characters.

    Each new chunk begins with the trailing sentences of the previous one (up to `overlap`
    characters) so that facts spanning a boundary stay retrievable.
    """
    validate_params(chunk_size, overlap)
    units: list[str] = []
    for sentence in split_sentences(text):
        units.extend(_hard_split(sentence, chunk_size) if len(sentence) > chunk_size else [sentence])

    chunks: list[str] = []
    current: list[str] = []
    fresh = 0  # sentences in `current` that were not part of the previous chunk
    for unit in units:
        if current and len(" ".join(current)) + 1 + len(unit) > chunk_size:
            chunks.append(" ".join(current))
            tail: list[str] = []
            for sentence in reversed(current):
                if len(" ".join([sentence, *tail])) > overlap:
                    break
                tail.insert(0, sentence)
            current, fresh = tail, 0
            while current and len(" ".join(current)) + 1 + len(unit) > chunk_size:
                current.pop(0)
        current.append(unit)
        fresh += 1
    if current and fresh:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(pages: list[Page], chunk_size: int = 600, overlap: int = 100) -> list[Chunk]:
    """Chunk every page separately (so a chunk never spans pages) and number the chunks."""
    validate_params(chunk_size, overlap)
    result: list[Chunk] = []
    for page in pages:
        for text in chunk_text(page.text, chunk_size, overlap):
            if sum(ch.isalnum() for ch in text) < MIN_CHUNK_ALNUM:
                continue  # skip page numbers, stray symbols, etc.
            result.append(Chunk(text=text, page=page.number, chunk_index=len(result)))
    return result
