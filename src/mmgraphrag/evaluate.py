"""Measure the system against COCO ground truth.

1. Extraction: precision / recall of the objects the VLM lists, against COCO annotations,
   within the 80 COCO categories (other words the VLM uses are outside the label space).
2. Routing: share of questions sent to the right index.
3. Counting questions, answered through the graph: the LLM-written Cypher is compared with
   (a) the same count computed by a reference query on the graph (did the LLM translate the
   question correctly?) and (b) the COCO count (end to end, VLM errors included).
4. Semantic search: precision@5, an image counting as relevant if COCO annotates the target
   object on it.
"""

from __future__ import annotations

import itertools
import json
import random
from collections import Counter
from typing import Any

from mmgraphrag.agent import first_number, generate_cypher, route
from mmgraphrag.config import SETTINGS
from mmgraphrag.extract import load_extractions
from mmgraphrag.store import run_read_query, semantic_search

SEMANTIC_QUERIES = [
    ("a cat resting indoors", "cat"),
    ("people playing a sport on a field", "sports ball"),
    ("a plate of food on a table", "dining table"),
    ("a busy city street with vehicles", "car"),
    ("a train at a station", "train"),
    ("someone working on a laptop", "laptop"),
    ("an aeroplane in the sky or on a runway", "airplane"),
    ("a bedroom with a bed", "bed"),
    ("a person riding a horse", "horse"),
    ("a bathroom with a toilet and a sink", "toilet"),
]


def extraction_scores(
    preds: dict[int, set[str]], truth: dict[int, set[str]], categories: set[str]
) -> dict[str, Any]:
    tp = fp = fn = 0
    missed: Counter[str] = Counter()
    for image_id, gt in truth.items():
        pred = preds.get(image_id, set()) & categories
        tp += len(pred & gt)
        fp += len(pred - gt)
        fn += len(gt - pred)
        missed.update(gt - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "most_missed": missed.most_common(5),
    }


def counting_questions(
    truth: dict[int, set[str]], n_single: int = 15, n_pairs: int = 15, seed: int = 0
) -> list[dict[str, Any]]:
    """Seeded counting questions with their COCO answer."""
    rng = random.Random(seed)
    freq = Counter(c for objs in truth.values() for c in objs)
    singles = sorted(c for c, n in freq.items() if n >= 5)
    pairs = Counter(p for objs in truth.values() for p in itertools.combinations(sorted(objs), 2))
    frequent_pairs = sorted(p for p, n in pairs.items() if n >= 3)
    questions = []
    for c in rng.sample(singles, min(n_single, len(singles))):
        questions.append(
            {"question": f"How many images show a {c}?", "objects": [c], "coco": freq[c]}
        )
    for a, b in rng.sample(frequent_pairs, min(n_pairs, len(frequent_pairs))):
        questions.append(
            {
                "question": f"How many images show both a {a} and a {b}?",
                "objects": [a, b],
                "coco": pairs[(a, b)],
            }
        )
    return questions


def graph_count(objects: list[str]) -> int:
    rows = run_read_query(
        "MATCH (i:Image) WHERE all(n IN $names WHERE (i)-[:CONTAINS]->(:Object {name: n})) "
        "RETURN count(i) AS n",
        names=objects,
    )
    return int(rows[0]["n"])


def evaluate(gt: dict[int, dict[str, Any]], ids: list[int], categories: list[str]) -> dict:
    records = {r["image_id"]: r for r in load_extractions()}
    truth = {i: set(gt[i]["objects"]) for i in ids if i in records}
    preds = {i: set(records[i]["objects"]) for i in truth}
    report: dict[str, Any] = {
        "images": len(truth),
        "vlm": SETTINGS.vlm_model,
        "prompt": SETTINGS.prompt_version,
        "llm": SETTINGS.llm_model,
        "extraction": extraction_scores(preds, truth, set(categories)),
    }

    qa = []
    for q in counting_questions(truth):
        chosen = route(q["question"])
        try:
            cypher = generate_cypher(q["question"])
            got = first_number(run_read_query(cypher))
        except Exception as exc:
            cypher, got = f"ERROR {type(exc).__name__}: {exc}", None
        qa.append(
            {
                **q,
                "route": chosen,
                "cypher": cypher,
                "answer": got,
                "graph": graph_count(q["objects"]),
            }
        )
    semantic = []
    for text, target in SEMANTIC_QUERIES:
        hits = semantic_search(text, k=5)
        relevant = sum(target in truth.get(h["image_id"], set()) for h in hits)
        semantic.append(
            {"query": text, "target": target, "route": route(text), "precision_at_5": relevant / 5}
        )

    n = len(qa)
    report["counting"] = {
        "questions": n,
        "routed_to_graph": sum(q["route"] == "graph" for q in qa) / n,
        "cypher_matches_graph": sum(q["answer"] == q["graph"] for q in qa) / n,
        "answer_matches_coco": sum(q["answer"] == q["coco"] for q in qa) / n,
        "graph_matches_coco": sum(q["graph"] == q["coco"] for q in qa) / n,
    }
    report["semantic"] = {
        "queries": len(semantic),
        "routed_to_semantic": sum(s["route"] == "semantic" for s in semantic) / len(semantic),
        "mean_precision_at_5": sum(s["precision_at_5"] for s in semantic) / len(semantic),
    }
    out = SETTINGS.data_dir.parent / "results"
    out.mkdir(exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(report, indent=2))
    (out / "counting_details.json").write_text(json.dumps(qa, indent=2))
    (out / "semantic_details.json").write_text(json.dumps(semantic, indent=2))
    return report
