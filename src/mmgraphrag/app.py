"""Gradio interface: ask a question, see the route, the query, the answer and the cited images."""

from __future__ import annotations

from typing import Any

import gradio as gr

from mmgraphrag.agent import answer
from mmgraphrag.config import SETTINGS
from mmgraphrag.store import run_read_query

EXAMPLES = [
    "How many images show both a person and a dog?",
    "Which objects appear most often together with a car?",
    "Find photos of people playing a sport on a field",
    "Show me a cosy living room",
]


def _files(ids: list[int]) -> list[str]:
    if not ids:
        return []
    rows = run_read_query(f"MATCH (i:Image) WHERE i.id IN {ids[:12]} RETURN i.file AS file")
    return [str(SETTINGS.images_dir / r["file"]) for r in rows]


def ask(question: str) -> tuple[str, str, list[Any]]:
    if not question.strip():
        return "", "", []
    result = answer(question)
    header = f"**Route:** {result.route}"
    if result.error:
        header += f"  \n**Error:** {result.error}"
    query = result.cypher or "(semantic search in ChromaDB, no Cypher)"
    return f"{header}\n\n{result.text}", query, _files(result.images)


def build() -> gr.Blocks:
    with gr.Blocks(title="Multimodal Graph RAG") as demo:
        gr.Markdown(
            "# Multimodal Graph RAG\nLocal models only: qwen3-vl describes the images, "
            "Neo4j and ChromaDB index them, an LLM routes and answers with citations."
        )
        question = gr.Textbox(label="Question", placeholder=EXAMPLES[0])
        go = gr.Button("Ask", variant="primary")
        out = gr.Markdown()
        query = gr.Code(label="Query", language="sql")
        gallery = gr.Gallery(label="Cited images", columns=6, height=260)
        gr.Examples(EXAMPLES, inputs=question)
        go.click(ask, question, [out, query, gallery])
        question.submit(ask, question, [out, query, gallery])
    return demo
