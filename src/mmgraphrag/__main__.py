"""python -m mmgraphrag {download,extract,index,evaluate,app}"""

from __future__ import annotations

import argparse
import json

from mmgraphrag.config import SETTINGS


def main() -> int:
    parser = argparse.ArgumentParser(prog="mmgraphrag")
    parser.add_argument("command", choices=["download", "extract", "index", "evaluate", "app"])
    args = parser.parse_args()

    from mmgraphrag.coco import download_images, load_ground_truth, select_images

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

        records = load_extractions(SETTINGS.data_dir)
        load_graph(records)
        load_vectors(records)
        print(f"{len(records)} images indexed in Neo4j and ChromaDB")
    elif args.command == "evaluate":
        from mmgraphrag.evaluate import evaluate

        print(json.dumps(evaluate(gt, ids, categories), indent=2))
    elif args.command == "app":
        from mmgraphrag.app import build

        build().launch(server_name="127.0.0.1", server_port=7860)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
