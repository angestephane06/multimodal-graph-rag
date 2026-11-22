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

PROMPT_V1 = """Describe this photo for a search index. Answer with a JSON object only:
{{"caption": "<one factual sentence>",
  "scene": "<one or two words: e.g. kitchen, street, beach, living room, field>",
  "objects": ["<object>", ...]}}
List every distinct kind of object you can actually see, once each. When an object matches one
of these category names, use that exact name: {categories}. Do not guess objects you cannot see."""

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V1, "v3": PROMPT_V1}

# v2 and v3 = the same description, then a second look: the model checks the category names it
# listed against the image. Listing every category name primes the model to "see" some of them.
# v2 checks all names in one question; on busy scenes the model recites most of the category
# list and confirms it all at once. v3 asks one yes/no question per name instead.
VERIFY_PROMPT = """Look at the photo again. For each name below, answer true only if at least one
such object is clearly visible in the photo, otherwise false. Answer with a JSON object only,
mapping each name to true or false: {names}"""

VERIFY_ONE_PROMPT = """Is there at least one {name} clearly visible in this photo? Answer with a
JSON object only: {{"visible": true}} or {{"visible": false}}."""


def verify_objects(
    image_path: Path, objects: list[str], categories: list[str], one_by_one: bool = False
) -> list[str]:
    """Keep the category objects the model confirms; free-form names are kept as they are."""
    known = set(categories)
    to_check = [o for o in objects if o in known]
    if not to_check:
        return objects
    if one_by_one:
        rejected = {
            o
            for o in to_check
            if describe_image(image_path, VERIFY_ONE_PROMPT.format(name=o)).get("visible") is False
        }
    else:
        answer = describe_image(image_path, VERIFY_PROMPT.format(names=json.dumps(to_check)))
        rejected = {o for o in to_check if answer.get(o) is False}
    return [o for o in objects if o not in rejected]


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


def extract_all(
    ids: list[int], files: dict[int, str], categories: list[str], out_dir: Path | None = None
) -> dict[str, Any]:
    out_dir = out_dir or SETTINGS.extractions_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = PROMPTS[SETTINGS.prompt_version].format(categories=", ".join(categories))
    done, failed, start = 0, 0, time.time()
    for image_id in ids:
        target = out_dir / f"{image_id}.json"
        if target.exists():
            continue
        try:
            path = SETTINGS.images_dir / files[image_id]
            raw = describe_image(path, prompt)
            candidates = normalise_objects(raw.get("objects", []), categories)
            objects = candidates
            if SETTINGS.prompt_version in ("v2", "v3"):
                one_by_one = SETTINGS.prompt_version == "v3"
                objects = verify_objects(path, candidates, categories, one_by_one)
            record = {
                "image_id": image_id,
                "file": files[image_id],
                "caption": str(raw.get("caption", "")).strip(),
                "scene": str(raw.get("scene", "")).strip().lower(),
                "objects": objects,
                "candidates": candidates,
                "model": SETTINGS.vlm_model,
                "prompt": SETTINGS.prompt_version,
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


def load_extractions(run_dir: Path | None = None) -> list[dict[str, Any]]:
    run_dir = run_dir or SETTINGS.extractions_dir
    return [json.loads(p.read_text()) for p in sorted(run_dir.glob("*.json"))]
