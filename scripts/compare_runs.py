"""Compare extraction runs on the same images, without Neo4j or an LLM.

The graph is built from the extraction records, so the graph answer to "how many images show
X (and Y)?" is the number of records listing X (and Y). This script computes that count for
each run on the images every run described, against COCO, using the evaluation questions.

    python scripts/compare_runs.py data/runs/test/qwen3-vl_8b-v1 data/runs/test/<new run>
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from mmgraphrag.coco import load_ground_truth, select_images
from mmgraphrag.config import SETTINGS
from mmgraphrag.evaluate import counting_questions, extraction_scores
from mmgraphrag.extract import load_extractions


def main(runs: list[Path]) -> dict:
    gt, categories = load_ground_truth(SETTINGS.data_dir)
    ids = select_images(gt, SETTINGS.n_images)
    records = {r: {x["image_id"]: x for x in load_extractions(r)} for r in runs}
    common = [i for i in ids if all(i in rec for rec in records.values())]
    truth = {i: set(gt[i]["objects"]) for i in common}
    questions = counting_questions(truth)
    out: dict = {"images": len(common), "questions": len(questions), "runs": {}}
    for run, rec in records.items():
        preds = {i: set(rec[i]["objects"]) for i in common}
        errors, relative = [], []
        for q in questions:
            got = sum(all(o in preds[i] for o in q["objects"]) for i in common)
            errors.append(got - q["coco"])
            relative.append(abs(got - q["coco"]) / q["coco"])
        out["runs"][run.name] = {
            "extraction": extraction_scores(preds, truth, set(categories)),
            "counts_exact": sum(e == 0 for e in errors),
            "counts_within_1": sum(abs(e) <= 1 for e in errors),
            "median_relative_error": round(statistics.median(relative), 3),
            "too_low": sum(e < 0 for e in errors),
            "too_high": sum(e > 0 for e in errors),
        }
    return out


if __name__ == "__main__":
    print(json.dumps(main([Path(p) for p in sys.argv[1:]]), indent=2))
