"""Query-time retrieval: embed -> vector search -> similarity threshold -> keyword rerank."""
from __future__ import annotations

from dataclasses import dataclass

from rag.textutils import content_tokens

FOLLOW_UP_PRONOUNS = {"it", "its", "they", "them", "their", "this", "these", "those", "he", "she", "his", "her"}
FOLLOW_UP_STARTS = ("and ", "what about", "how about", "why", "also", "then", "what else")


@dataclass
class RetrievedChunk:
    chunk_id: int
    doc_id: str
    filename: str
    page: int | None
    chunk_index: int
    text: str
    score: float          # cosine similarity from the vector search
    keyword_score: float  # share of query terms present in the chunk
    final_score: float    # score used for ranking (after optional reranking)
    rank: int = 0


def contextualize_question(question: str, history: list[str] | None) -> str:
    """Make short follow-up questions self-contained using the previous user question."""
    question = question.strip()
    if not history:
        return question
    lowered = question.lower()
    words = set(lowered.replace("?", " ").split())
    is_follow_up = (
        bool(words & FOLLOW_UP_PRONOUNS)
        or lowered.startswith(FOLLOW_UP_STARTS)
        or len(content_tokens(question)) <= 2
    )
    if not is_follow_up:
        return question
    return f"{question} (regarding: {history[-1].strip()})"


class Retriever:
    def __init__(self, embedder, store):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int = 4, threshold: float = 0.25, rerank: bool = True,
                 doc_ids: set[str] | None = None, candidate_factor: int = 3) -> list[RetrievedChunk]:
        query = (query or "").strip()
        if not query:
            raise ValueError("The query is empty.")
        if self.store.vector_count == 0:
            return []
        vector = self.embedder.encode([query])[0]
        hits = self.store.search(vector, k=max(top_k, 1) * candidate_factor, doc_ids=set(doc_ids) if doc_ids else None)
        query_terms = set(content_tokens(query))
        results: list[RetrievedChunk] = []
        for meta, score in hits:
            if score < threshold:
                continue
            overlap = len(query_terms & set(content_tokens(meta["text"]))) / len(query_terms) if query_terms else 0.0
            final = 0.8 * score + 0.2 * overlap if rerank else score
            results.append(RetrievedChunk(meta["id"], meta["doc_id"], meta["filename"], meta.get("page"),
                                          meta["chunk_index"], meta["text"], score, overlap, final))
        results.sort(key=lambda r: r.final_score, reverse=True)
        results = results[:top_k]
        for position, item in enumerate(results, start=1):
            item.rank = position
        return results
