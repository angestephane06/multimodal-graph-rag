"""Minimal Ollama client: chat (text or image), JSON mode, embeddings."""

from __future__ import annotations

import base64
import json
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
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "seed": 0, "num_ctx": 8192},
    }
    if json_mode:
        payload["format"] = "json"
    r = httpx.post(f"{SETTINGS.ollama_url}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return str(r.json()["message"]["content"])


def describe_image(path: Path, prompt: str) -> dict[str, Any]:
    image = base64.b64encode(path.read_bytes()).decode()
    text = chat(
        SETTINGS.vlm_model, [{"role": "user", "content": prompt, "images": [image]}], json_mode=True
    )
    return dict(json.loads(text))


def embed(texts: list[str]) -> list[list[float]]:
    r = httpx.post(
        f"{SETTINGS.ollama_url}/api/embed",
        json={"model": SETTINGS.embed_model, "input": texts},
        timeout=300.0,
    )
    r.raise_for_status()
    return [list(map(float, v)) for v in r.json()["embeddings"]]
