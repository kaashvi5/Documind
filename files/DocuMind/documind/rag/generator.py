"""Answer generation from retrieved context: FLAN-T5 (local, free) with an extractive fallback."""
from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from typing import Callable

from config import DEFAULT_GENERATION_MODEL, MAX_INPUT_TOKENS, get_device
from rag.chunking import split_sentences
from rag.textutils import content_tokens

_NOT_FOUND_PHRASES = {
    "unanswerable", "unknown", "no answer", "none", "na", "not mentioned", "not specified", "not stated",
    "i dont know", "i do not know", "cannot be answered", "no information",
}


@dataclass
class GenerationResult:
    answer: str
    found: bool
    chunks_used: int
    truncated: bool = False
    model: str = ""


def is_not_found(text: str) -> bool:
    cleaned = re.sub(r"[^a-z ]", "", (text or "").lower().replace("'", "")).strip()
    return not cleaned or cleaned in _NOT_FOUND_PHRASES or cleaned.startswith("unanswerable")


def build_prompt(question: str, texts: list[str], count_tokens: Callable[[str], int],
                 truncate: Callable[[str, int], str], max_input_tokens: int = MAX_INPUT_TOKENS,
                 detailed: bool = False) -> tuple[str, int, bool]:
    """Build a grounded prompt that fits the token budget.

    Returns (prompt, number_of_chunks_included, context_was_truncated). Chunks are added in rank
    order, so the best evidence is always kept; the instruction and question are never cut.
    """
    instruction = ("Read the context and answer the question using only the information in the context. "
                   'If the context does not contain the answer, reply "unanswerable".')
    ask = "Answer in a few complete sentences:" if detailed else "Answer in one or two complete sentences:"
    footer = f"\n\nQuestion: {question}\n{ask}"
    budget = max_input_tokens - count_tokens(f"{instruction}\n\nContext:\n{footer}") - 4
    lines: list[str] = []
    used, truncated = 0, False
    for number, text in enumerate(texts, start=1):
        label = f"[{number}] "
        room = budget - used - count_tokens(label) - 1
        if room < 30 and lines:
            truncated = True
            break
        room = max(room, 30)
        size = count_tokens(text)
        if size > room:
            text, size, truncated = truncate(text, room), room, True
        lines.append(label + text)
        used += count_tokens(label) + size + 1
        if truncated:
            break
    if len(lines) < len(texts):
        truncated = True
    return f"{instruction}\n\nContext:\n" + "\n".join(lines) + footer, len(lines), truncated


def find_supporting_chunks(answer: str, texts: list[str], max_sources: int = 3) -> tuple[list[int], float]:
    """Pick the chunks whose wording best supports the answer.

    Returns (1-based chunk numbers, best coverage). Coverage is the share of the answer's content
    words found in the best chunk; a low value signals an ungrounded answer.
    """
    explicit = [int(n) for n in re.findall(r"\[(\d{1,2})\]", answer) if 1 <= int(n) <= len(texts)]
    answer_terms = set(content_tokens(re.sub(r"\[\d{1,2}\]", " ", answer)))
    if not answer_terms:
        return (sorted(set(explicit)) or [1]), 1.0
    scores = [len(answer_terms & set(content_tokens(t))) / len(answer_terms) for t in texts]
    best = max(scores)
    if best == 0:
        return (sorted(set(explicit)) or [1]), 0.0
    ranked = sorted(range(len(texts)), key=lambda i: scores[i], reverse=True)
    picked = [i + 1 for i in ranked if scores[i] >= max(0.3, 0.6 * best)][:max_sources]
    return sorted(set(picked) | set(explicit)), best


class ExtractiveGenerator:
    """Model-free fallback: returns the most relevant sentences from the retrieved chunks.

    Always grounded (it only copies source text), but not fluent or abstractive.
    """

    name = "extractive-fallback"
    is_fallback = True
    device = "cpu"
    loaded = True

    def load(self) -> "ExtractiveGenerator":
        return self

    def generate(self, question: str, chunks, detailed: bool = False, max_new_tokens: int | None = None) -> GenerationResult:
        terms = set(content_tokens(question))
        sentences = [(ci, si, sent, set(content_tokens(sent)))
                     for ci, chunk in enumerate(chunks) for si, sent in enumerate(split_sentences(chunk.text))]
        # Rarer query words (in the retrieved text) count more than words that appear everywhere.
        weight = {t: math.log(1 + len(sentences) / (1 + sum(t in toks for *_, toks in sentences))) for t in terms}
        total = sum(weight.values())
        scored = []
        for ci, si, sent, toks in sentences:
            matched = sum(weight[t] for t in terms & toks)
            if total and matched > 0:
                scored.append((matched / total, -ci, -si, ci, si, sent))
        if not scored:
            return GenerationResult("", False, len(chunks), model=self.name)
        scored.sort(reverse=True)
        best = scored[0][0]
        if best < 0.34:
            return GenerationResult("", False, len(chunks), model=self.name)
        limit = 3 if detailed else 2
        picked, seen = [], set()
        for item in scored:
            if item[0] >= 0.5 * best and item[5] not in seen:
                seen.add(item[5])
                picked.append(item)
            if len(picked) >= limit:
                break
        picked.sort(key=lambda it: (it[3], it[4]))
        return GenerationResult(" ".join(p[5] for p in picked), True, len(chunks), model=self.name)


class FlanT5Generator:
    """Local instruction-tuned seq2seq model (FLAN-T5). Loaded lazily; GPU used when available."""

    is_fallback = False

    def __init__(self, model_name: str = DEFAULT_GENERATION_MODEL, device: str | None = None,
                 max_input_tokens: int = MAX_INPUT_TOKENS):
        self.name = model_name
        self.device = get_device(device)
        self.max_input_tokens = max_input_tokens
        self.loaded = False
        self._tok = None
        self._model = None
        self._lock = threading.RLock()

    def load(self) -> "FlanT5Generator":
        with self._lock:
            if self.loaded:
                return self
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.name)  # float32: T5 is unstable in fp16
            self._model.to(self.device).eval()
            self.loaded = True
        return self

    # token helpers -------------------------------------------------------------
    def _count(self, text: str) -> int:
        return len(self._tok(text, add_special_tokens=False).input_ids)

    def _truncate(self, text: str, tokens: int) -> str:
        return self._tok.decode(self._tok(text, add_special_tokens=False).input_ids[:tokens], skip_special_tokens=True)

    def _run(self, prompt: str, max_new_tokens: int, beams: int) -> str:
        import torch

        inputs = self._tok(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=beams,
                                          no_repeat_ngram_size=3, early_stopping=True, do_sample=False)
        return self._tok.decode(output[0], skip_special_tokens=True).strip()

    def generate(self, question: str, chunks, detailed: bool = False, max_new_tokens: int | None = None) -> GenerationResult:
        self.load()
        max_new = max_new_tokens or (256 if detailed else 128)
        texts = [c.text for c in chunks]
        prompt, used, truncated = build_prompt(question, texts, self._count, self._truncate,
                                               self.max_input_tokens, detailed)
        try:
            raw = self._run(prompt, max_new, beams=4)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            import torch

            torch.cuda.empty_cache()
            try:  # retry with greedy decoding (much less memory)
                raw = self._run(prompt, max_new, beams=1)
            except RuntimeError as exc2:
                if "out of memory" not in str(exc2).lower():
                    raise
                torch.cuda.empty_cache()
                self._model.to("cpu")  # last resort: continue on CPU
                self.device = "cpu"
                raw = self._run(prompt, max_new, beams=1)
        if is_not_found(raw):
            return GenerationResult("", False, used, truncated, self.name)
        return GenerationResult(raw, True, used, truncated, self.name)
