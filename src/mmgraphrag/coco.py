"""COCO 2017 validation subset: image selection, download, ground-truth objects.

Only the annotations (CC BY 4.0) and a seeded list of image ids drive the dataset; the images
themselves are downloaded from the official COCO server and never committed.
"""

from __future__ import annotations

import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

ANNOTATIONS = "coco/annotations/instances_val2017.json"
IMAGE_URL = "http://images.cocodataset.org/val2017/{file}"


def load_ground_truth(data_dir: Path) -> tuple[dict[int, dict[str, object]], list[str]]:
    """image id -> {file, objects (set of category names)}, and the 80 category names."""
    raw = json.loads((data_dir / ANNOTATIONS).read_text())
    names = {c["id"]: c["name"] for c in raw["categories"]}
    objects: dict[int, set[str]] = defaultdict(set)
    for ann in raw["annotations"]:
        objects[ann["image_id"]].add(names[ann["category_id"]])
    images = {
        im["id"]: {"file": im["file_name"], "objects": objects[im["id"]]}
        for im in raw["images"]
        if objects[im["id"]]
    }
    return images, sorted(names.values())


def select_images(images: dict[int, dict[str, object]], n: int, seed: int = 0) -> list[int]:
    ids = sorted(images)
    return sorted(random.Random(seed).sample(ids, n))


def select_dev_images(
    images: dict[int, dict[str, object]], exclude: list[int], n: int = 100, seed: int = 1
) -> list[int]:
    """Development images, disjoint from the evaluation images, used to tune the prompt."""
    taken = set(exclude)
    ids = sorted(i for i in images if i not in taken)
    return sorted(random.Random(seed).sample(ids, n))


def download_images(data_dir: Path, ids: list[int], images: dict[int, dict[str, object]]) -> int:
    out = data_dir / "images"
    out.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for image_id in ids:
        file = str(images[image_id]["file"])
        target = out / file
        if not target.exists():
            urllib.request.urlretrieve(IMAGE_URL.format(file=file), target)
            fetched += 1
    return fetched
