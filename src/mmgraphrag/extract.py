"""Describe each image with the vision-language model: caption, scene, objects.

Results are cached one JSON file per image, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from mmgraphrag.config import SETTINGS
from mmgraphrag.ollama import describe_image

PROMPT = """Describe this photo for a search index. Answer with a JSON object only:
{{"caption": "<one factual sentence>",
  "scene": "<one or two words: e.g. kitchen, street, beach, living room, field>",
  "objects": ["<object>", ...]}}
List every distinct kind of object you can actually see, once each. When an object matches one
of these category names, use that exact name: {categories}. Do not guess objects you cannot see."""


def normalise_objects(objects: object, categories: list[str]) -> list[str]:
    """Lower-case, deduplicate, and map simple plural or variant forms to category names."""
    known = {c.lower(): c for c in categories}
    out: list[str] = []
    if not isinstance(objects, list):
        return out
    for item in objects:
        name = str(item).strip().lower()
        if not name:
            continue
        for candidate in (name, name.rstrip("s"), name.removesuffix("es")):
            if candidate in known:
                name = known[candidate]
                break
        if name not in out:
            out.append(name)
    return out


def extract_all(ids: list[int], files: dict[int, str], categories: list[str]) -> dict[str, Any]:
    out_dir = SETTINGS.extractions_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = PROMPT.format(categories=", ".join(categories))
    done, failed, start = 0, 0, time.time()
    for image_id in ids:
        target = out_dir / f"{image_id}.json"
        if target.exists():
            continue
        try:
            raw = describe_image(SETTINGS.images_dir / files[image_id], prompt)
            record = {
                "image_id": image_id,
                "file": files[image_id],
                "caption": str(raw.get("caption", "")).strip(),
                "scene": str(raw.get("scene", "")).strip().lower(),
                "objects": normalise_objects(raw.get("objects", []), categories),
                "model": SETTINGS.vlm_model,
            }
        except Exception as exc:  # recorded, the image is retried on the next run
            failed += 1
            print(f"image {image_id}: {type(exc).__name__}: {exc}")
            continue
        target.write_text(json.dumps(record, indent=1))
        done += 1
        if done % 25 == 0:
            rate = (time.time() - start) / done
            print(f"{done} images described, {rate:.1f} s/image", flush=True)
    elapsed = round(time.time() - start, 1)
    return {"described": done, "failed": failed, "seconds": elapsed}


def load_extractions(data_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((data_dir / "extractions").glob("*.json"))]
