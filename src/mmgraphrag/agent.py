"""Question answering: route, retrieve (Cypher or vectors), synthesise with citations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from mmgraphrag.config import SETTINGS
from mmgraphrag.ollama import chat
from mmgraphrag.store import SCHEMA, run_read_query, semantic_search

Route = Literal["graph", "semantic"]

ROUTER_PROMPT = """Classify the question about a photo collection.
graph: counting, listing or comparing objects and scenes ("how many images show a dog",
       "which objects appear most often with a car", "images with both a cat and a sofa").
semantic: describing or finding images by their content or atmosphere ("find a sunny beach",
       "photos of people cooking", "what does the market look like").
Answer with one word: graph or semantic.
Question: {question}"""

CYPHER_PROMPT = """Write one Cypher query for Neo4j that answers the question.
Graph schema:
{schema}
Object names are lower-case singular COCO category names (e.g. "dog", "cell phone").
Rules: read-only query; count images with count(DISTINCT i); when listing images, return i.id
and i.file and LIMIT 20. Return only the query, no explanation, no code fence.
Question: {question}"""

ANSWER_PROMPT = """Answer the question from the evidence only, in one or two sentences.
Cite the images you rely on as [image <id>]. If the evidence does not answer it, say so.
Question: {question}
Evidence:
{evidence}"""

# Defence in depth: the query also runs in a read transaction.
_WRITE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|CALL|FOREACH)\b", re.I
)


class UnsafeQuery(ValueError):
    pass


def route(question: str) -> Route:
    reply = chat(
        SETTINGS.llm_model, [{"role": "user", "content": ROUTER_PROMPT.format(question=question)}]
    )
    return "graph" if "graph" in reply.lower() else "semantic"


def clean_cypher(text: str) -> str:
    """Strip code fences and trailing prose; refuse anything that could write."""
    fenced = re.search(r"```(?:cypher)?\s*(.*?)```", text, re.S | re.I)
    query = (fenced.group(1) if fenced else text).strip().rstrip(";")
    if _WRITE.search(query):
        raise UnsafeQuery(f"refused non read-only query: {query[:120]}")
    if not re.match(r"^\s*(MATCH|OPTIONAL\s+MATCH|WITH|RETURN|UNWIND)\b", query, re.I):
        raise UnsafeQuery(f"not a query: {query[:120]}")
    return query


def first_number(rows: list[dict[str, Any]]) -> int | None:
    for row in rows:
        for value in row.values():
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


@dataclass
class Answer:
    question: str
    route: Route
    text: str
    cypher: str | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    images: list[int] = field(default_factory=list)
    error: str | None = None


def generate_cypher(question: str) -> str:
    reply = chat(
        SETTINGS.llm_model,
        [{"role": "user", "content": CYPHER_PROMPT.format(schema=SCHEMA, question=question)}],
    )
    return clean_cypher(reply)


def graph_evidence(cypher: str, rows: list[dict[str, Any]]) -> str:
    """Query and result in plain words. Given a bare `{'count(DISTINCT i)': 7}`, the model
    answered that the evidence did not say how many images there were."""
    lines = [f"This query was run on the image graph:\n{cypher}", "Result:"]
    for row in rows[:20]:
        lines.append("; ".join(f"{k} = {v}" for k, v in row.items()))
    if not rows:
        lines.append("(no rows: nothing in the graph matches)")
    return "\n".join(lines)


def image_ids(rows: list[dict[str, Any]], limit: int = 12) -> list[int]:
    """Image ids found in the result, whether returned one per row or collected in a list."""
    ids: list[int] = []
    for row in rows:
        for key, value in row.items():
            if not key.lower().endswith("id") and not key.lower().endswith("ids"):
                continue
            for v in value if isinstance(value, list) else [value]:
                if isinstance(v, int) and v not in ids:
                    ids.append(v)
    return ids[:limit]


def examples_query(cypher: str, limit: int = 6) -> str | None:
    """The same MATCH, returning image ids instead of the aggregate.

    Asking the LLM to return both a count and example ids made it write UNIONs and undefined
    variables (5 of 30 queries failed, against 0 with the count alone), so the citation query
    is derived by code from the validated query instead.
    """
    if not re.search(r"\(\s*i\s*:\s*Image\b", cypher):
        return None
    last = max(m.start() for m in re.finditer(r"\bRETURN\b", cypher, re.I))
    return clean_cypher(cypher[:last] + f"RETURN DISTINCT i.id AS id LIMIT {limit}")


def example_images(cypher: str) -> list[int]:
    query = examples_query(cypher)
    if query is None:
        return []
    try:
        return image_ids(run_read_query(query))
    except Exception:  # the aggregate query may not keep `i` in scope: no citation then
        return []


def answer(question: str, forced_route: Route | None = None) -> Answer:
    chosen = forced_route or route(question)
    if chosen == "graph":
        try:
            cypher = generate_cypher(question)
            rows = run_read_query(cypher)
        except Exception as exc:
            return Answer(question, chosen, "The graph query failed.", error=str(exc))
        evidence = graph_evidence(cypher, rows)
        images = image_ids(rows) or example_images(cypher)
        text = chat(
            SETTINGS.llm_model,
            [
                {
                    "role": "user",
                    "content": ANSWER_PROMPT.format(question=question, evidence=evidence),
                }
            ],
        )
        return Answer(question, chosen, text, cypher, rows, images)
    hits = semantic_search(question)
    evidence = "\n".join(f"[image {h['image_id']}] {h['document']}" for h in hits)
    text = chat(
        SETTINGS.llm_model,
        [{"role": "user", "content": ANSWER_PROMPT.format(question=question, evidence=evidence)}],
    )
    cited = [int(m) for m in re.findall(r"image (\d+)", text)]
    return Answer(question, chosen, text, images=cited or [h["image_id"] for h in hits])
