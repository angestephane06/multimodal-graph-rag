"""Minimal Ollama client: chat (text or image), JSON mode, embeddings."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

import httpx

from mmgraphrag.config import SETTINGS


def chat(
    model: str,
    messages: list[dict[str, Any]],
    json_mode: bool = False,
    temperature: float = 0.0,
    timeout: float = 300.0,
    num_predict: int | None = None,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "seed": 0, "num_ctx": 8192},
    }
    if num_predict is not None:
        payload["options"]["num_predict"] = num_predict
    if json_mode:
        payload["format"] = "json"
    r = httpx.post(f"{SETTINGS.ollama_url}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return str(r.json()["message"]["content"])


def parse_json_lenient(text: str) -> dict[str, Any]:
    """Parse the model's JSON; if the answer was cut (a looping object list hits the length cap),
    keep the caption, scene and the complete object names written before the cut."""
    try:
        return dict(json.loads(text))
    except json.JSONDecodeError:
        out: dict[str, Any] = {}
        for key in ("caption", "scene"):
            m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text)
            if m:
                out[key] = m.group(1)
        m = re.search(r'"objects"\s*:\s*\[(.*)', text, re.S)
        if not m and not out:
            raise
        out["objects"] = re.findall(r'"([^"]+)"\s*[,\]]', m.group(1)) if m else []
        return out


def describe_image(path: Path, prompt: str) -> dict[str, Any]:
    """Ask the vision model about one image and parse its JSON answer."""
    image = base64.b64encode(path.read_bytes()).decode()
    text = chat(
        SETTINGS.vlm_model,
        [{"role": "user", "content": prompt, "images": [image]}],
        json_mode=True,
        timeout=600.0,
        # qwen3-vl reasons before answering even with think=False; the cap counts both.
        num_predict=SETTINGS.vlm_max_tokens,
    )
    return parse_json_lenient(text)


def embed(texts: list[str]) -> list[list[float]]:
    r = httpx.post(
        f"{SETTINGS.ollama_url}/api/embed",
        json={"model": SETTINGS.embed_model, "input": texts},
        timeout=300.0,
    )
    r.raise_for_status()
    return [list(map(float, v)) for v in r.json()["embeddings"]]
