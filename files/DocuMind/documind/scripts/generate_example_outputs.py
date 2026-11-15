"""Run the sample questions through the pipeline and save the results to docs/example_outputs.md.

Uses the real models (MiniLM + FLAN-T5) when they can be loaded; otherwise the offline fallback
components, and the output file says which mode produced it.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import RAGSettings  # noqa: E402
from rag.embeddings import HashingEmbedder  # noqa: E402
from rag.generator import ExtractiveGenerator  # noqa: E402
from rag.pipeline import DocuMindPipeline  # noqa: E402

QUESTIONS = [
    ("How much is the home-office stipend?", None),
    ("What is the difference between supervised and unsupervised learning?", None),
    ("Where does the Calvin cycle take place?", None),
    ("What is the capacity factor of offshore wind farms?", None),
    ("Why does RAG reduce hallucinations?", None),
    ("What about the internet?", "How much is the home-office stipend?"),
    ("Who won the 2018 football world cup?", None),
]


def main(real: bool = False) -> None:
    data = Path(tempfile.mkdtemp())
    if real:
        pipeline = DocuMindPipeline(data)
        pipeline.load_models()
    else:
        pipeline = DocuMindPipeline(data, embedder=HashingEmbedder(), generator=ExtractiveGenerator())
    mode = ("real models: " + pipeline.embedder.name + " + " + pipeline.generator.name) if real and not pipeline.embedder.is_fallback \
        else "OFFLINE FALLBACK (hashing embedder + extractive answers), not the real models"
    settings = RAGSettings(similarity_threshold=0.25 if real else 0.12)
    pipeline.ingest_files(sorted((ROOT / "sample_docs").glob("*")), settings)
    lines = ["# Example outputs", "", f"Generated on the sample documents. Mode: **{mode}**.", ""]
    if "FALLBACK" in mode:
        lines += ["> These outputs come from the dependency-free fallback so they could be produced without downloading models.",
                  "> With FLAN-T5 and MiniLM (the default in Colab) answers are more fluent and retrieval handles paraphrased questions.", ""]
    for question, previous in QUESTIONS:
        answer = pipeline.ask(question, settings, history=[previous] if previous else None)
        lines += [f"## Q: {question}", "", f"**Answer ({answer.kind}):** {answer.text}", ""]
        for s in answer.sources[:3]:
            page = f"page {s.page}" if s.page else "page n/a"
            flag = "cited" if s.cited else "retrieved, not cited"
            lines.append(f"- [{s.number}] `{s.filename}`, {page}, similarity {s.score:.2f} ({flag}): {s.text[:160]}...")
        lines.append("")
    out = ROOT / "docs" / "example_outputs.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main(real="--real" in sys.argv)
