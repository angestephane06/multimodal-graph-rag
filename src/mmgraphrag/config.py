"""Settings, read from the environment (or a local .env file, never committed)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT / "data"
    n_images: int = int(os.environ.get("N_IMAGES", "500"))
    ollama_url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    vlm_model: str = os.environ.get("VLM_MODEL", "qwen3-vl:30b")
    prompt_version: str = os.environ.get("PROMPT_VERSION", "v1")
    vlm_max_tokens: int = int(os.environ.get("VLM_MAX_TOKENS", "8192"))
    llm_model: str = os.environ.get("LLM_MODEL", "mistral-small3.2:latest")
    embed_model: str = os.environ.get("EMBED_MODEL", "bge-m3")
    neo4j_uri: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password: str = os.environ.get("NEO4J_PASSWORD", "")

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    def run_dir(self, split: str = "test") -> Path:
        """One folder of extractions per split, vision model and prompt version."""
        slug = self.vlm_model.replace(":", "_").replace("/", "_")
        return self.data_dir / "runs" / split / f"{slug}-{self.prompt_version}"

    @property
    def extractions_dir(self) -> Path:
        return self.run_dir("test")


SETTINGS = Settings()
