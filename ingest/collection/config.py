"""Loading and validation for data-collection YAML configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_MODELS = ("qwen3-vl:4b", "qwen2.5vl:7b", "gemma3:4b")


@dataclass(frozen=True)
class CollectionConfig:
    raw: dict[str, Any]

    @property
    def season(self) -> int:
        return int(self.raw["season"])

    @property
    def game(self) -> str:
        return str(self.raw["game"])

    @property
    def collections_root(self) -> Path:
        return Path(self.raw["storage"]["collections"])

    @property
    def sampling_fps(self) -> float:
        return float(self.raw["sampling"]["fps"])

    @property
    def split_group_by(self) -> str:
        """Field used to keep related frames in a single dataset split."""
        return str(self.raw.get("split", {}).get("group_by", "event"))

    @property
    def jpeg_quality(self) -> int:
        return int(self.raw["sampling"].get("jpeg_quality", 85))

    @property
    def classes(self) -> list[str]:
        return [str(item["name"]) for item in self.raw["classes"]]

    @property
    def models(self) -> tuple[str, ...]:
        values = self.raw.get("ollama", {}).get("models", DEFAULT_MODELS)
        return tuple(str(value) for value in values)

    @property
    def ollama_url(self) -> str:
        return str(self.raw.get("ollama", {}).get("url", "http://127.0.0.1:11434"))

    @property
    def iou_threshold(self) -> float:
        return float(self.raw.get("ollama", {}).get("iou_threshold", 0.40))

    @property
    def keep_alive(self) -> str:
        """How long Ollama keeps each model's weights resident between frames.

        Defaults to "0" -- unload immediately -- which is the safe choice on a machine that
        cannot hold all three model sets at once. Where there is room (roughly 12 GB of VRAM for
        the default trio), setting something like "10m" avoids reloading a model for every
        single frame and cuts a run several-fold. It changes speed only, never output.
        """
        return str(self.raw.get("ollama", {}).get("keep_alive", "0"))

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.raw, sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


def load_config(path: str | Path) -> CollectionConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    required = ("season", "game", "storage", "sampling", "split", "classes")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Missing configuration keys: {', '.join(missing)}")
    fps = float(data["sampling"].get("fps", 0))
    if fps <= 0:
        raise ValueError("sampling.fps must be greater than zero")
    classes = data.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("classes must contain at least one class")
    names = [item.get("name") for item in classes if isinstance(item, dict)]
    if len(names) != len(classes) or len(set(names)) != len(names):
        raise ValueError("Every class needs a unique name")
    threshold = float(data.get("ollama", {}).get("iou_threshold", 0.4))
    if not 0 < threshold <= 1:
        raise ValueError("ollama.iou_threshold must be in (0, 1]")
    return CollectionConfig(data)
