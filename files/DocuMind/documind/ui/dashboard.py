"""Sidebar (upload, library, statistics), settings tab and their event handlers."""
from __future__ import annotations

import gradio as gr

from config import (ALLOWED_EXTENSIONS, EMBEDDING_MODELS, GENERATION_MODELS, MAX_FILE_MB, SAMPLE_DIR)
from rag.loaders import ocr_available
from ui.components import (hero_html, library_html, logo_html, model_status_html, settings_from_inputs, stats_html,
                           status_html, upload_log_html)

SYNC_KEYS = ("hero", "stats", "library", "delete_dd", "filter_dd", "model_status")


def snapshot(pipeline) -> tuple:
    """Fresh values for every component that mirrors repository state."""
    docs = pipeline.list_documents()
    stats = pipeline.stats()
    choices = [(d["filename"], d["doc_id"]) for d in docs]
    return (hero_html(stats), stats_html(stats), library_html(docs), gr.Dropdown(choices=choices, value=None),
            gr.Dropdown(choices=choices, value=[]), model_status_html(pipeline.status()))


def sync_outputs(c: dict) -> list:
    return [c[k] for k in SYNC_KEYS]


def build_sidebar(pipeline) -> dict:
    c: dict = {}
    stats, docs = pipeline.stats(), pipeline.list_documents()
    with gr.Column(scale=1, min_width=330, elem_classes="dm-side"):
        gr.HTML(logo_html())
        gr.HTML('<div class="dm-h">Upload documents</div>')
        c["files"] = gr.File(label=f"PDF, DOCX, TXT or Markdown, up to {MAX_FILE_MB} MB each", file_count="multiple",
                             file_types=list(ALLOWED_EXTENSIONS), type="filepath", height=130)
        with gr.Row():
            c["index_btn"] = gr.Button("Index documents", variant="primary")
            c["sample_btn"] = gr.Button("Load samples", variant="secondary")
        c["upload_log"] = gr.HTML(status_html("Choose files, then select Index documents.", "info"))
        gr.HTML('<div class="dm-h">Document library</div>')
        c["library"] = gr.HTML(library_html(docs))
        c["delete_dd"] = gr.Dropdown(choices=[(d["filename"], d["doc_id"]) for d in docs], value=None,
                                     label="Document to remove")
        c["delete_btn"] = gr.Button("Delete document", size="sm")
        gr.HTML('<div class="dm-h">Repository statistics</div>')
        c["stats"] = gr.HTML(stats_html(stats))
        with gr.Accordion("Clear repository", open=False):
            c["clear_confirm"] = gr.Checkbox(label="I want to remove all documents and vectors")
            c["clear_btn"] = gr.Button("Clear repository", variant="stop", size="sm")
    return c


def build_overview(pipeline) -> dict:
    return {"hero": gr.HTML(hero_html(pipeline.stats()))}


def build_settings(pipeline) -> dict:
    c: dict = {}
    status = pipeline.status()
    gr.HTML('<div class="dm-h">Chunking</div>')
    with gr.Row():
        c["chunk_size"] = gr.Slider(200, 2000, value=600, step=50, label="Chunk size (characters)",
                                    info="Applies to documents indexed from now on.")
        c["overlap"] = gr.Slider(0, 500, value=100, step=10, label="Chunk overlap (characters)",
                                 info="Characters shared by neighbouring chunks.")
    c["reindex_btn"] = gr.Button("Re-index all documents with these settings", size="sm")
    c["reindex_log"] = gr.HTML("")
    gr.HTML('<div class="dm-h">Retrieval and answers</div>')
    with gr.Row():
        c["top_k"] = gr.Slider(1, 10, value=4, step=1, label="Top-K chunks", info="How many passages the model sees.")
        c["threshold"] = gr.Slider(0, 0.9, value=0.25, step=0.01, label="Similarity threshold",
                                   info="Passages below this cosine similarity are ignored.")
    c["rerank"] = gr.Checkbox(value=True, label="Keyword reranking", info="Blend keyword coverage into the ranking.")
    c["detailed"] = gr.Checkbox(value=False, label="Detailed answers", info="Longer answers; slightly slower.")
    c["ocr"] = gr.Checkbox(value=False, label="OCR scanned PDFs", interactive=ocr_available(),
                           info="Needs Tesseract." if ocr_available() else "Unavailable: Tesseract is not installed.")
    gr.HTML('<div class="dm-h">Models</div>')
    with gr.Row():
        c["gen_model"] = gr.Dropdown(choices=[(f"{d}", n) for n, d in GENERATION_MODELS.items()], value=None,
                                     label="Generation model", allow_custom_value=True)
        c["emb_model"] = gr.Dropdown(choices=[(f"{d}", n) for n, d in EMBEDDING_MODELS.items()], value=None,
                                     label="Embedding model", allow_custom_value=True)
    c["load_models_btn"] = gr.Button("Load selected models", size="sm")
    c["model_status"] = gr.HTML(model_status_html(status))
    return c


def wire_dashboard(pipeline, c: dict) -> None:
    outs = sync_outputs(c)

    def _paths(files) -> list[str]:
        return [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in (files or [])]

    def _run_ingest(paths, chunk_size, overlap, ocr, progress):
        try:
            settings = settings_from_inputs(chunk_size=chunk_size, overlap=overlap, ocr=ocr)
        except ValueError as exc:
            yield (status_html(str(exc), "error"), gr.update(), *snapshot(pipeline))
            return
        pipeline.busy = f"Indexing {len(paths)} file{'s' if len(paths) != 1 else ''}"
        yield (status_html(pipeline.busy + "...", "info"), gr.update(), *snapshot(pipeline))
        try:
            results = pipeline.ingest_files(paths, settings, progress=lambda f, t: progress(f, desc=t))
            log = upload_log_html(results)
        except Exception as exc:  # never leave the UI stuck in the busy state
            log = status_html(f"Indexing failed: {type(exc).__name__}: {exc}", "error")
        finally:
            pipeline.busy = ""
        yield (log, gr.File(value=None), *snapshot(pipeline))

    def on_index(files, chunk_size, overlap, ocr, progress=gr.Progress()):
        paths = _paths(files)
        if not paths:
            yield (status_html("Choose at least one file first.", "warn"), gr.update(), *snapshot(pipeline))
            return
        yield from _run_ingest(paths, chunk_size, overlap, ocr, progress)

    def on_samples(chunk_size, overlap, ocr, progress=gr.Progress()):
        paths = sorted(str(p) for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in ALLOWED_EXTENSIONS)
        if not paths:
            yield (status_html("The sample_docs folder is empty.", "warn"), gr.update(), *snapshot(pipeline))
            return
        yield from _run_ingest(paths, chunk_size, overlap, ocr, progress)

    index_inputs = [c["files"], c["chunk_size"], c["overlap"], c["ocr"]]
    c["index_btn"].click(on_index, index_inputs, [c["upload_log"], c["files"], *outs], api_name="index_documents")
    c["sample_btn"].click(on_samples, index_inputs[1:], [c["upload_log"], c["files"], *outs], api_name="load_samples")

    def on_delete(doc_id):
        if not doc_id:
            return (status_html("Select a document to remove first.", "warn"), *snapshot(pipeline))
        name = pipeline.store.documents.get(doc_id, {}).get("filename", "document")
        ok = pipeline.delete_document(doc_id)
        message = status_html(f"Removed {name} and its vectors.", "ok") if ok else status_html("That document no longer exists.", "warn")
        return (message, *snapshot(pipeline))

    c["delete_btn"].click(on_delete, [c["delete_dd"]], [c["upload_log"], *outs], api_name="delete_document")

    def on_clear(confirm):
        if not confirm:
            return (status_html("Tick the confirmation box to clear the repository.", "warn"), gr.update(), *snapshot(pipeline))
        count = pipeline.clear_repository()
        return (status_html(f"Cleared the repository ({count} document{'s' if count != 1 else ''} removed).", "ok"),
                gr.Checkbox(value=False), *snapshot(pipeline))

    c["clear_btn"].click(on_clear, [c["clear_confirm"]], [c["upload_log"], c["clear_confirm"], *outs], api_name="clear_repository")

    def on_reindex(chunk_size, overlap, ocr, progress=gr.Progress()):
        try:
            settings = settings_from_inputs(chunk_size=chunk_size, overlap=overlap, ocr=ocr)
        except ValueError as exc:
            return (status_html(str(exc), "error"), *snapshot(pipeline))
        if not pipeline.list_documents():
            return (status_html("There are no documents to re-index.", "warn"), *snapshot(pipeline))
        pipeline.busy = "Re-indexing"
        try:
            results = pipeline.reindex_all(settings, progress=lambda f, t: progress(f, desc=t))
            log = upload_log_html(results)
        finally:
            pipeline.busy = ""
        return (log, *snapshot(pipeline))

    c["reindex_btn"].click(on_reindex, [c["chunk_size"], c["overlap"], c["ocr"]], [c["reindex_log"], *outs], api_name="reindex")

    def on_models(gen_model, emb_model, chunk_size, overlap, progress=gr.Progress()):
        messages = []
        pipeline.busy = "Loading models"
        try:
            if emb_model:
                progress(0.2, desc="Loading embedding model")
                settings = settings_from_inputs(chunk_size=chunk_size, overlap=overlap)
                messages.append(pipeline.set_embedding_model(emb_model, settings))
            if gen_model:
                progress(0.6, desc="Loading generation model")
                messages.append(pipeline.set_generation_model(gen_model))
        except Exception as exc:
            messages.append(f"Model switch failed: {type(exc).__name__}: {exc}")
        finally:
            pipeline.busy = ""
        if not messages:
            messages = ["Pick a model first."]
        return (*snapshot(pipeline)[:-1], model_status_html(pipeline.status()) + "".join(status_html(m, "info") for m in messages))

    c["load_models_btn"].click(on_models, [c["gen_model"], c["emb_model"], c["chunk_size"], c["overlap"]], outs, api_name="load_models")
