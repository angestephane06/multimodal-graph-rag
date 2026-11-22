"""python -m mmgraphrag {download,extract,index,evaluate,app,dev}"""

from __future__ import annotations

import argparse
import json

from mmgraphrag.config import SETTINGS


def main() -> int:
    parser = argparse.ArgumentParser(prog="mmgraphrag")
    parser.add_argument(
        "command", choices=["download", "extract", "index", "evaluate", "app", "dev"]
    )
    args = parser.parse_args()

    from mmgraphrag.coco import (
        download_images,
        load_ground_truth,
        select_dev_images,
        select_images,
    )

    gt, categories = load_ground_truth(SETTINGS.data_dir)
    ids = select_images(gt, SETTINGS.n_images)
    if args.command == "download":
        print(f"{download_images(SETTINGS.data_dir, ids, gt)} images downloaded")
    elif args.command == "extract":
        from mmgraphrag.extract import extract_all

        files = {i: str(gt[i]["file"]) for i in ids}
        print(json.dumps(extract_all(ids, files, categories)))
    elif args.command == "index":
        from mmgraphrag.extract import load_extractions
        from mmgraphrag.store import load_graph, load_vectors

        records = load_extractions()
        load_graph(records)
        load_vectors(records)
        print(f"{len(records)} images indexed in Neo4j and ChromaDB")
    elif args.command == "evaluate":
        from mmgraphrag.evaluate import evaluate

        print(json.dumps(evaluate(gt, ids, categories), indent=2))
    elif args.command == "dev":
        # Prompt tuning happens here, on images disjoint from the evaluation set.
        from mmgraphrag.evaluate import extraction_scores
        from mmgraphrag.extract import extract_all, load_extractions

        dev = select_dev_images(gt, ids)
        download_images(SETTINGS.data_dir, dev, gt)
        run = SETTINGS.run_dir("dev")
        files = {i: str(gt[i]["file"]) for i in dev}
        print(json.dumps(extract_all(dev, files, categories, run)))
        records = {r["image_id"]: r for r in load_extractions(run)}
        truth = {i: set(gt[i]["objects"]) for i in dev if i in records}
        preds = {i: set(records[i]["objects"]) for i in truth}
        scores = {"images": len(truth), **extraction_scores(preds, truth, set(categories))}
        (run.parent / f"{run.name}.json").write_text(json.dumps(scores, indent=2))
        print(json.dumps(scores, indent=2))
    elif args.command == "app":
        from mmgraphrag.app import build

        build().launch(server_name="127.0.0.1", server_port=7860)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
