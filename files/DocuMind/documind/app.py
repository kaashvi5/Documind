"""DocuMind entry point.  Run locally with `python app.py` (add --share for a public link)."""
from __future__ import annotations

import argparse
import inspect

import gradio as gr

from config import DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATION_MODEL
from rag.pipeline import DocuMindPipeline
from ui.chat import build_chat, wire_chat
from ui.components import CSS, FORCE_LIGHT_JS, how_it_works_html, make_theme
from ui.dashboard import build_overview, build_settings, build_sidebar, snapshot, sync_outputs, wire_dashboard

# Gradio 6 takes theme/css/js in launch(); Gradio 5 takes them in Blocks().
_STYLE_IN_LAUNCH = "css" in inspect.signature(gr.Blocks.launch).parameters


def build_app(pipeline: DocuMindPipeline | None = None) -> gr.Blocks:
    pipeline = pipeline or DocuMindPipeline()
    style = {} if _STYLE_IN_LAUNCH else {"theme": make_theme(), "css": CSS, "js": FORCE_LIGHT_JS}
    with gr.Blocks(title="DocuMind", **style) as demo:
        c: dict = {}
        with gr.Row(elem_classes="dm-layout", equal_height=False):
            c.update(build_sidebar(pipeline))
            with gr.Column(scale=3, min_width=480):
                c.update(build_overview(pipeline))
                with gr.Tabs():
                    with gr.Tab("Chat"):
                        c.update(build_chat(pipeline))
                    with gr.Tab("Settings"):
                        c.update(build_settings(pipeline))
                    with gr.Tab("How it works"):
                        gr.HTML(how_it_works_html())
        wire_dashboard(pipeline, c)
        wire_chat(pipeline, c)
        demo.load(lambda: snapshot(pipeline), None, sync_outputs(c), api_name=False)
    demo.pipeline = pipeline
    return demo


def launch_app(demo: gr.Blocks, share: bool = False, **kwargs):
    style = {"theme": make_theme(), "css": CSS, "js": FORCE_LIGHT_JS} if _STYLE_IN_LAUNCH else {}
    demo.queue(max_size=32)
    return demo.launch(share=share, **style, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="DocuMind: RAG document assistant")
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--generation-model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--no-preload", action="store_true", help="load the LLM on the first question")
    args = parser.parse_args()
    pipeline = DocuMindPipeline(embedding_model=args.embedding_model, generation_model=args.generation_model)
    if not args.no_preload:
        for warning in pipeline.load_models():
            print("WARNING:", warning)
    launch_app(build_app(pipeline), share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
