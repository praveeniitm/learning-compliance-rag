"""Gradio demo. Run: python app/app.py  (http://127.0.0.1:7860)

Binds to localhost and has no authentication. Put it behind SSO / an authenticating proxy before
exposing it on a network.
"""

import shutil

import gradio as gr

from lcrag.config import DATA_DIR
from lcrag.graph import build_graph
from lcrag.store import make_retriever, sync_index

APP = build_graph(make_retriever(sync_index()[0]))
SAMPLE = DATA_DIR / "samples" / "RN-LMS-2026-09.md"
EXAMPLES = ["Is the annual HIPAA refresher a legal requirement or a Northwind rule?",
            "A Fresno supervisor says forklift re-evaluations are every 24 months now. Is that right?",
            "What does 29 CFR 1910.178(l)(4)(iii) require?",
            "What was the new-hire completion window before the January 2026 policy change?",
            "What is the passing score for the EHS-FL-201 forklift practical evaluation?",
            "When was the harassment back-fill for finding F-4 run?"]


def ask(question):
    out = APP.invoke({"question": question})
    r = out["result"]
    status = f"**Status:** {out['status']}" + (" (revised once after the grounding check)" if out["attempts"] > 1 else "")
    if out.get("unsupported"):
        status += "\n\n**Unsupported statements flagged by the grader:** " + "; ".join(out["unsupported"])
    answer = r.answer + (f"\n\n**Clarifying question:** {r.clarifying_question}" if r.clarifying_question else "")
    sources = "\n\n---\n\n".join(f"**[S{i}]** {d.page_content}" for i, d in enumerate(out["docs"], 1))
    return status, answer, sources


def add_sample():
    (DATA_DIR / "incoming").mkdir(exist_ok=True)
    shutil.copy(SAMPLE, DATA_DIR / "incoming" / SAMPLE.name)
    return resync()


def resync():
    global APP
    vs, stats = sync_index()  # only new or changed chunks are embedded
    APP = build_graph(make_retriever(vs))
    return f"Index synced: {stats}"


with gr.Blocks(title="Compliance Training Knowledge Assistant") as demo:
    gr.Markdown("# Compliance Training Knowledge Assistant\nGrounded answers over OSHA/HIPAA regulations, "
                "California law and (fictional) Northwind policies, SOPs and course catalog.")
    q = gr.Textbox(label="Question", lines=2)
    gr.Examples(EXAMPLES, q)
    status, answer = gr.Markdown(), gr.Markdown()
    with gr.Accordion("Retrieved sources", open=False):
        sources = gr.Markdown()
    gr.Button("Ask", variant="primary").click(ask, q, [status, answer, sources])
    q.submit(ask, q, [status, answer, sources])
    with gr.Accordion("Index freshness", open=False):
        log = gr.Markdown()
        with gr.Row():
            gr.Button("Add September release notes").click(add_sample, outputs=log)
            gr.Button("Re-sync index").click(resync, outputs=log)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1")
