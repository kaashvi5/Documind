"""Shared UI pieces: theme, CSS and HTML builders (kept free of Gradio event logic)."""
from __future__ import annotations

import html
import re
from datetime import datetime

import gradio as gr

from config import RAGSettings
from rag.textutils import STOPWORDS, tokenize

esc = html.escape

SUGGESTIONS = [
    "How much is the home-office stipend?",
    "What is the difference between supervised and unsupervised learning?",
    "Where does the Calvin cycle take place?",
    "What are the main challenges of wind and solar power?",
]

FORCE_LIGHT_JS = "() => { document.documentElement.classList.remove('dark'); document.body.classList.remove('dark'); }"

CSS = """
:root{--dm-bg:#F5F6FB;--dm-card:#FFFFFF;--dm-ink:#1A1D3B;--dm-muted:#667085;--dm-line:#E3E6F2;
--dm-blue:#3F6BF4;--dm-violet:#7B5CF0;--dm-blue-soft:#EAF0FF;--dm-violet-soft:#F1ECFF;
--dm-ok:#0F7B55;--dm-ok-soft:#E6F6EF;--dm-warn:#9A5B06;--dm-warn-soft:#FFF4E0;--dm-bad:#B42318;--dm-bad-soft:#FDECEA}
.gradio-container{max-width:1360px!important;margin:0 auto!important;background:var(--dm-bg)!important;color:var(--dm-ink)}
footer{display:none!important}
.dm-layout{gap:20px!important;align-items:flex-start!important}
.dm-side{background:var(--dm-card);border:1px solid var(--dm-line);border-radius:20px;padding:18px!important;gap:12px!important}
@media (max-width:900px){.dm-layout{flex-direction:column!important}.dm-side{min-width:100%!important}}
.dm-logo{display:flex;gap:12px;align-items:center}
.dm-logo svg{flex:none}
.dm-logo b{font-size:20px;letter-spacing:-.02em;display:block}
.dm-logo span{font-size:12.5px;color:var(--dm-muted);line-height:1.35;display:block;max-width:230px}
.dm-h{font-weight:700;font-size:13.5px;margin:6px 0 0;color:var(--dm-ink)}
.dm-hero{background:var(--dm-card);border:1px solid var(--dm-line);border-radius:20px;padding:20px 24px;display:flex;
justify-content:space-between;gap:16px;align-items:flex-start;border-top:3px solid var(--dm-violet)}
.dm-hero h1{font-size:26px;letter-spacing:-.03em;margin:0 0 4px;font-weight:800}
.dm-hero p{margin:0;color:var(--dm-muted);font-size:14px;max-width:560px;line-height:1.5}
.dm-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:12px}
@media (max-width:700px){.dm-stats{grid-template-columns:repeat(2,minmax(0,1fr))}}
.dm-stat{background:var(--dm-card);border:1px solid var(--dm-line);border-radius:16px;padding:14px 16px}
.dm-stat small{display:block;color:var(--dm-muted);font-size:12px;margin-bottom:2px}
.dm-stat strong{font-size:24px;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.dm-stat em{display:block;font-style:normal;font-size:11.5px;color:var(--dm-muted);margin-top:2px;overflow-wrap:anywhere}
.dm-pill{display:inline-flex;align-items:center;gap:6px;border-radius:999px;padding:4px 11px;font-size:12.5px;font-weight:600}
.dm-pill::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor}
.dm-pill-ok{background:var(--dm-ok-soft);color:var(--dm-ok)}.dm-pill-busy{background:var(--dm-blue-soft);color:var(--dm-blue)}
.dm-pill-warn{background:var(--dm-warn-soft);color:var(--dm-warn)}
.dm-note{border-radius:12px;padding:9px 12px;font-size:13px;line-height:1.45;margin:6px 0}
.dm-note-info{background:var(--dm-blue-soft);color:#24408F}.dm-note-ok{background:var(--dm-ok-soft);color:var(--dm-ok)}
.dm-note-warn{background:var(--dm-warn-soft);color:var(--dm-warn)}.dm-note-error{background:var(--dm-bad-soft);color:var(--dm-bad)}
.dm-log{display:flex;flex-direction:column;gap:6px}
.dm-log-row{border-radius:12px;padding:8px 11px;font-size:12.5px;line-height:1.4;overflow-wrap:anywhere}
.dm-log-row b{display:block;font-size:13px}
.dm-doc{display:flex;gap:10px;align-items:center;padding:9px 0;border-bottom:1px solid var(--dm-line)}
.dm-doc:last-child{border-bottom:none}
.dm-ext{flex:none;width:38px;height:38px;border-radius:10px;display:grid;place-items:center;font-size:10.5px;font-weight:800}
.dm-ext-a{background:var(--dm-blue-soft);color:var(--dm-blue)}.dm-ext-b{background:var(--dm-violet-soft);color:var(--dm-violet)}
.dm-doc-main{min-width:0}.dm-doc-main b{display:block;font-size:13px;overflow-wrap:anywhere}
.dm-doc-main span{font-size:11.5px;color:var(--dm-muted)}
.dm-empty{color:var(--dm-muted);font-size:13px;line-height:1.5;padding:6px 0}
.dm-kv{display:grid;grid-template-columns:1fr auto;gap:5px 10px;font-size:12.5px}
.dm-kv span{color:var(--dm-muted)}.dm-kv b{font-variant-numeric:tabular-nums;text-align:right}
.dm-chatbot{border-radius:18px!important}
.dm-sugg-title{font-size:12.5px;color:var(--dm-muted);margin:4px 0}
.dm-chip-btn{text-align:left!important;white-space:normal!important;height:auto!important;font-size:12.5px!important}
.dm-answer-text{font-size:15px;line-height:1.6}
.dm-ref{display:inline-grid;place-items:center;min-width:20px;height:20px;border-radius:10px;background:var(--dm-violet);color:#fff;
font-size:11.5px;font-weight:700;margin-left:3px;vertical-align:middle;padding:0 5px}
.dm-meta{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--dm-muted);margin:8px 0 2px}
.dm-sources{margin-top:10px;display:flex;flex-direction:column;gap:7px}
.dm-sources-title{font-weight:700;font-size:13px}
.dm-src{border:1px solid var(--dm-line);border-radius:14px;background:#FBFBFE;overflow:hidden}
.dm-src[open]{border-color:#C9D3FA;background:#fff}
.dm-src summary{cursor:pointer;display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center;padding:9px 12px;list-style:none;font-size:13px}
.dm-src summary::-webkit-details-marker{display:none}
.dm-badge{display:inline-grid;place-items:center;width:24px;height:24px;border-radius:50%;font-size:12px;font-weight:700;
border:1.5px solid var(--dm-violet);color:var(--dm-violet);flex:none}
.dm-badge-on{background:var(--dm-violet);color:#fff}
.dm-src-name{font-weight:600;overflow-wrap:anywhere}
.dm-chip{background:var(--dm-blue-soft);color:var(--dm-blue);border-radius:8px;padding:2px 8px;font-size:11.5px;font-weight:600}
.dm-bar{display:inline-block;width:70px;height:6px;border-radius:3px;background:var(--dm-line);overflow:hidden}
.dm-bar i{display:block;height:100%;background:linear-gradient(90deg,var(--dm-blue),var(--dm-violet))}
.dm-score{font-size:12px;color:var(--dm-muted);font-variant-numeric:tabular-nums}
.dm-passage{padding:2px 14px 12px 14px;font-size:13px;line-height:1.6;color:#2D3152;border-top:1px dashed var(--dm-line)}
.dm-passage mark{background:#E6DEFF;color:inherit;border-radius:3px;padding:0 1px}
.dm-typing{display:inline-flex;gap:5px;padding:6px 2px}
.dm-typing span{width:8px;height:8px;border-radius:50%;background:var(--dm-violet);opacity:.35;animation:dmblink 1s infinite}
.dm-typing span:nth-child(2){animation-delay:.15s}.dm-typing span:nth-child(3){animation-delay:.3s}
@keyframes dmblink{0%,100%{opacity:.25}50%{opacity:1}}
@media (prefers-reduced-motion:reduce){.dm-typing span{animation:none;opacity:.7}}
.dm-flow{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:8px 0 18px}
.dm-flow span{background:var(--dm-card);border:1px solid var(--dm-line);border-radius:10px;padding:6px 11px;font-size:13px;font-weight:600}
.dm-flow i{color:var(--dm-muted);font-style:normal}
.dm-doc-text{font-size:14px;line-height:1.65;max-width:760px}
"""


def make_theme() -> gr.themes.Base:
    return gr.themes.Base(
        primary_hue="indigo", secondary_hue="violet", neutral_hue="slate", radius_size="lg",
        font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#F5F6FB",
        button_primary_background_fill="linear-gradient(135deg,#3F6BF4,#7B5CF0)",
        button_primary_background_fill_hover="linear-gradient(135deg,#3860E6,#6F4FE6)",
        button_primary_text_color="white",
    )


# ----------------------------------------------------------------------------- helpers
def human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num} B"


def settings_from_inputs(chunk_size=600, overlap=100, top_k=4, threshold=0.25, rerank=True,
                         detailed=False, ocr=False, doc_ids=()) -> RAGSettings:
    return RAGSettings(chunk_size=int(chunk_size), chunk_overlap=int(overlap), top_k=int(top_k),
                       similarity_threshold=float(threshold), keyword_rerank=bool(rerank),
                       detailed_answers=bool(detailed), use_ocr=bool(ocr), doc_ids=tuple(doc_ids or ())).validate()


def status_html(message: str, kind: str = "info") -> str:
    return f'<div class="dm-note dm-note-{kind}">{esc(message)}</div>'


def logo_html() -> str:
    mark = ('<svg width="44" height="44" viewBox="0 0 44 44" aria-hidden="true"><defs><linearGradient id="dmg" x1="0" y1="0" x2="1" y2="1">'
            '<stop offset="0" stop-color="#3F6BF4"/><stop offset="1" stop-color="#7B5CF0"/></linearGradient></defs>'
            '<rect width="44" height="44" rx="13" fill="url(#dmg)"/><path d="M13 11h13l6 6v16a2 2 0 0 1-2 2H13a2 2 0 0 1-2-2V13a2 2 0 0 1 2-2z" fill="#fff" fill-opacity=".95"/>'
            '<path d="M16 22h12M16 26h12M16 30h7" stroke="#5B5FE8" stroke-width="2" stroke-linecap="round"/></svg>')
    return (f'<div class="dm-logo">{mark}<div><b>DocuMind</b>'
            '<span>Ask questions across your documents. Every answer shows where it came from.</span></div></div>')


def _status_pill(stats: dict) -> str:
    if stats.get("busy"):
        return f'<span class="dm-pill dm-pill-busy">{esc(stats["busy"])}</span>'
    if stats.get("embedding_fallback") or stats.get("generation_fallback") or stats.get("warnings"):
        return '<span class="dm-pill dm-pill-warn">Ready with warnings</span>'
    return '<span class="dm-pill dm-pill-ok">Ready</span>'


def hero_html(stats: dict) -> str:
    models = f'{stats["embedding_model"].split("/")[-1]} + {stats["generation_model"].split("/")[-1]} on {stats["device"]}'
    cards = [("Documents", stats["documents"], ""), ("Chunks", stats["chunks"], ""),
             ("Vectors", stats["vectors"], "")]
    cards_html = "".join(f'<div class="dm-stat"><small>{t}</small><strong>{v:,}</strong></div>' for t, v, _ in cards)
    cards_html += f'<div class="dm-stat"><small>Status</small>{_status_pill(stats)}<em>{esc(models)}</em></div>'
    return ('<div class="dm-hero"><div><h1>Ask your documents</h1>'
            '<p>Upload files, ask in plain language, and get answers grounded in your own text, '
            'with the exact passages they came from.</p></div></div>'
            f'<div class="dm-stats">{cards_html}</div>')


def stats_html(stats: dict) -> str:
    rows = [("Documents", f'{stats["documents"]:,}'), ("Chunks", f'{stats["chunks"]:,}'),
            ("Vectors in FAISS", f'{stats["vectors"]:,}'), ("Characters indexed", f'{stats["characters"]:,}'),
            ("Index size on disk", human_size(stats["disk_bytes"])),
            ("Embeddings generated", f'{stats["embeddings_generated"]:,}'), ("Served from cache", f'{stats["cache_hits"]:,}')]
    return '<div class="dm-kv">' + "".join(f"<span>{k}</span><b>{v}</b>" for k, v in rows) + "</div>"


def library_html(docs: list[dict]) -> str:
    if not docs:
        return '<div class="dm-empty">No documents yet. Upload files above, or choose Load samples to try DocuMind right away.</div>'
    items = []
    for d in docs:
        ext = d.get("file_type", "").upper()[:4]
        tone = "a" if d.get("file_type") in ("pdf", "docx") else "b"
        pages = f'{d["pages"]} pages, ' if d.get("pages") else ""
        try:
            added = datetime.fromisoformat(d["added_at"]).strftime("%d %b %H:%M")
        except Exception:
            added = ""
        items.append(f'<div class="dm-doc"><div class="dm-ext dm-ext-{tone}">{esc(ext)}</div><div class="dm-doc-main">'
                     f'<b>{esc(d["filename"])}</b><span>{pages}{d["chunks"]} chunks, {human_size(d["size_bytes"])}'
                     f'{", added " + added if added else ""}</span></div></div>')
    return "".join(items)


def upload_log_html(results) -> str:
    if not results:
        return status_html("No files were processed.", "warn")
    palette = {"indexed": ("ok", "Indexed"), "replaced": ("ok", "Updated"), "duplicate": ("warn", "Skipped"), "failed": ("error", "Failed")}
    rows = []
    for r in results:
        kind, label = palette.get(r.status, ("info", r.status))
        detail = r.message
        if r.status in ("indexed", "replaced"):
            detail = f"{r.chunks} chunks from {r.chars:,} characters in {r.seconds:.1f} s. " + " ".join(r.warnings)
        rows.append(f'<div class="dm-log-row dm-note-{kind}"><b>{label}: {esc(r.filename)}</b>{esc(detail.strip())}</div>')
    return '<div class="dm-log">' + "".join(rows) + "</div>"


def model_status_html(status: dict) -> str:
    rows = [("Embedding model", status["embedding_model"] + (" (fallback)" if status["embedding_fallback"] else "")),
            ("Generation model", status["generation_model"] + (" (fallback)" if status["generation_fallback"] else "")
             + ("" if status["generator_loaded"] else " (loads on first question)")),
            ("Device", status["device"])]
    body = '<div class="dm-kv">' + "".join(f"<span>{k}</span><b>{esc(v)}</b>" for k, v in rows) + "</div>"
    warnings = "".join(status_html(w, "warn") for w in status["warnings"])
    return body + warnings


TYPING_HTML = '<div class="dm-typing" aria-label="DocuMind is thinking"><span></span><span></span><span></span></div>'


def error_html(message: str) -> str:
    return status_html(message, "error")


def _highlight(text: str, question: str) -> str:
    safe = esc(text)
    terms = {t.replace("'", "") for t in tokenize(question) if t not in STOPWORDS and len(t) > 2}
    terms -= {"amp", "quot", "lt", "gt"}
    if not terms:
        return safe
    pattern = r"(?i)(?<![&#\w])(" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")"
    return re.sub(pattern, r"<mark>\1</mark>", safe)


def _source_html(source, question: str) -> str:
    page = f"Page {source.page}" if source.page else "Page n/a"
    width = max(0, min(100, round(source.score * 100)))
    badge = "dm-badge dm-badge-on" if source.cited else "dm-badge"
    title = "Used in the answer" if source.cited else "Retrieved but not used in the answer"
    return (f'<details class="dm-src"><summary><span class="{badge}" title="{title}">{source.number}</span>'
            f'<span class="dm-src-name">{esc(source.filename)}</span><span class="dm-chip">{page}</span>'
            f'<span class="dm-bar" title="Cosine similarity"><i style="width:{width}%"></i></span>'
            f'<span class="dm-score">similarity {source.score:.2f}</span></summary>'
            f'<div class="dm-passage">{_highlight(source.text, question)}</div></details>')


def render_answer(answer) -> str:
    """HTML for one assistant message: answer text, notes, metadata and expandable sources."""
    if answer.kind in ("invalid", "empty"):
        return status_html(answer.text, "warn")
    parts: list[str] = []
    if answer.kind == "answer":
        pills = "".join(f'<span class="dm-ref" title="Source {s.number}: {esc(s.filename)}">{s.number}</span>'
                        for s in answer.sources if s.cited)
        parts.append(f'<div class="dm-answer-text">{esc(answer.text).replace(chr(10), "<br>")} {pills}</div>')
        if answer.low_confidence:
            parts.append(status_html("Low confidence: this answer's wording barely overlaps with the sources. Check them below.", "warn"))
    else:
        parts.append(status_html(answer.text, "warn"))
    for note in answer.warnings:
        parts.append(status_html(note, "warn"))
    if answer.truncated:
        parts.append(status_html("Some retrieved text was shortened to fit the model's input limit.", "info"))
    if answer.sources:
        title = "Sources" if answer.kind == "answer" else "Closest passages (not used)"
        parts.append(f'<div class="dm-sources"><div class="dm-sources-title">{title}</div>'
                     + "".join(_source_html(s, answer.standalone_question or answer.question) for s in answer.sources) + "</div>")
    meta = [f"{len(answer.sources)} source{'s' if len(answer.sources) != 1 else ''}", f"{answer.seconds:.1f} s"]
    if answer.model:
        meta.append(answer.model.split("/")[-1])
    parts.append('<div class="dm-meta">' + "".join(f"<span>{esc(m)}</span>" for m in meta) + "</div>")
    return "".join(parts)


def how_it_works_html() -> str:
    def flow(*steps: str) -> str:
        return '<div class="dm-flow">' + "<i>then</i>".join(f"<span>{s}</span>" for s in steps) + "</div>"
    return ('<div class="dm-doc-text"><div class="dm-h">When you index documents</div>'
            + flow("Upload", "Extract text", "Clean", "Chunk with overlap", "Add metadata", "Embed", "Store in FAISS")
            + '<div class="dm-h">When you ask a question</div>'
            + flow("Embed the question", "Search FAISS", "Apply threshold", "Rerank by keywords", "Build context", "Generate answer", "Show citations")
            + "<p><b>Embeddings</b> turn text into numbers so that passages with similar meaning end up close together. "
              "<b>Chunks</b> are short passages; neighbouring chunks overlap so facts at a boundary are not lost. "
              "<b>FAISS</b> finds the closest chunks quickly. The language model only sees those chunks, and is told to answer from them alone "
              "or say that the answer is not in your documents.</p>"
              "<p>Each source shows its similarity score (higher means closer; 1.00 is identical). Filled numbers mark passages that support the answer.</p></div>")
