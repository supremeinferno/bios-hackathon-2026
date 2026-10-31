"""Stage 2 worker manipulator profile (FORGE): what manipulators the team says the camera sees.

Declared in config/worker_profiles.yaml, never inferred from video, and holds no personal or
medical characteristics beyond a neutral description used to set expectations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

Kind = Literal["hand", "prosthetic", "forearm", "tool_end", "none"]


class Manipulator(BaseModel):
    id: Literal["left", "right"]
    kind: Kind = "hand"
    gloved: bool = False


class WorkerProfile(BaseModel):
    manipulators: list[Manipulator] = Field(default_factory=lambda: [Manipulator(id="left"), Manipulator(id="right")])
    source: str = "default"

    @property
    def expected(self) -> list[str]:
        """Sides we expect a hand-like manipulator on (hand landmark models can find these)."""
        return [m.id for m in self.manipulators if m.kind in ("hand", "prosthetic")]

    @property
    def any_gloved(self) -> bool:
        return any(m.gloved for m in self.manipulators)

    def allows(self, hand: str) -> bool:
        ids = {m.id for m in self.manipulators if m.kind != "none"}
        return {"left": "left" in ids, "right": "right" in ids, "both": {"left", "right"} <= ids}.get(hand, True)


def load_profile(path: Path, clip: str) -> WorkerProfile:
    if not path.is_file():
        return WorkerProfile()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sources = data.get("sources") or {}
    keys = [k for k in sources if clip.startswith(k)]
    if keys:
        key = max(keys, key=len)
        return WorkerProfile(**sources[key], source=key)
    return WorkerProfile(**(data.get("default") or {}), source="default")
