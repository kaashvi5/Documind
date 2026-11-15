"""Chat workspace: conversation, input, suggested prompts and document filter."""
from __future__ import annotations

import inspect

import gradio as gr

from ui.components import SUGGESTIONS, TYPING_HTML, error_html, render_answer, settings_from_inputs

_CHATBOT_KWARGS = {"type": "messages"} if "type" in inspect.signature(gr.Chatbot.__init__).parameters else {}
PLACEHOLDER = ("<div style='text-align:center;padding:32px 12px'><b style='font-size:17px'>Ask anything about your documents</b>"
               "<p style='color:#667085;margin-top:6px'>Index some files first, or load the samples, then ask a question below.</p></div>")


def message_text(content) -> str:
    """Plain text of a chat message (Gradio 6 passes lists of content blocks, Gradio 5 plain strings)."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", "")) if content.get("type", "text") == "text" else ""
    if isinstance(content, (list, tuple)):
        return " ".join(t for t in (message_text(part) for part in content) if t)
    return ""


def build_chat(pipeline) -> dict:
    c: dict = {}
    c["chatbot"] = gr.Chatbot(value=[], height=520, show_label=False, placeholder=PLACEHOLDER,
                              elem_classes="dm-chatbot", sanitize_html=True, **_CHATBOT_KWARGS)
    with gr.Row():
        c["msg"] = gr.Textbox(placeholder="Ask a question about your documents", show_label=False, container=False,
                              lines=1, max_lines=4, scale=8)
        c["send"] = gr.Button("Send", variant="primary", scale=1, min_width=90)
    with gr.Row():
        c["filter_dd"] = gr.Dropdown(choices=[], value=[], multiselect=True, scale=3,
                                     label="Search only in these documents (leave empty to search all)")
        c["clear_chat"] = gr.Button("Clear conversation", size="sm", scale=1, min_width=150)
    gr.HTML('<div class="dm-sugg-title">Suggested questions (they match the sample documents)</div>')
    with gr.Row():
        c["suggestions"] = [gr.Button(q, size="sm", elem_classes="dm-chip-btn") for q in SUGGESTIONS]
    return c


def wire_chat(pipeline, c: dict) -> None:
    def respond(message, history, top_k, threshold, rerank, detailed, doc_ids):
        history = list(history or [])
        message = (message or "").strip()
        if not message:
            gr.Warning("Type a question first.")
            yield history, ""
            return
        user = {"role": "user", "content": message}
        yield history + [user, {"role": "assistant", "content": TYPING_HTML}], ""
        try:
            settings = settings_from_inputs(top_k=top_k, threshold=threshold, rerank=rerank, detailed=detailed,
                                            doc_ids=doc_ids)
            previous = [message_text(m.get("content")) for m in history if m.get("role") == "user"]
            previous = [q for q in previous if q.strip()]
            body = render_answer(pipeline.ask(message, settings, previous))
        except Exception as exc:  # surface any failure in the chat instead of a silent crash
            body = error_html(f"Something went wrong: {type(exc).__name__}: {exc}")
        yield history + [user, {"role": "assistant", "content": body}], ""

    inputs = [c["msg"], c["chatbot"], c["top_k"], c["threshold"], c["rerank"], c["detailed"], c["filter_dd"]]
    outputs = [c["chatbot"], c["msg"]]
    c["msg"].submit(respond, inputs, outputs, api_name="ask")
    c["send"].click(respond, inputs, outputs, api_name=False)
    c["clear_chat"].click(lambda: ([], ""), None, outputs, api_name=False)

    def prefill(text):
        def setter():
            return text
        return setter

    for button, question in zip(c["suggestions"], SUGGESTIONS):
        button.click(prefill(question), None, c["msg"], api_name=False).then(respond, inputs, outputs, api_name=False)
