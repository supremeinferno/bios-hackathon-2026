"""Writers for the competition-facing files and our richer internal format."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from egovision.schema import ACTIONS_CSV_COLUMNS, Action, FrameHands


def _hand_dict(h: Any) -> dict[str, Any]:
    d = h.model_dump()
    d["bbox"] = [round(v, 1) for v in d["bbox"]]
    d["kp2d"] = [[round(x, 1), round(y, 1)] for x, y in d["kp2d"]]
    d["score"] = round(d["score"], 3)
    return d


def write_hands_jsonl(frames: list[FrameHands], path: Path) -> None:
    """PS1 suggested format: one JSON object per frame (PDF p.3), plus `id` and `interp`."""
    with open(path, "w", encoding="utf-8") as f:
        for fh in frames:
            f.write(json.dumps({"frame": fh.frame, "t": round(fh.t, 3),
                                "hands": [_hand_dict(h) for h in fh.hands]}) + "\n")


def read_hands_jsonl(path: Path) -> list[FrameHands]:
    with open(path, encoding="utf-8") as f:
        return [FrameHands.model_validate_json(line) for line in f if line.strip()]


def write_actions_csv(actions: list[Action], path: Path) -> None:
    """PS3 suggested format exactly: clip, start_s, end_s, verb, object, hand (PDF p.4)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(ACTIONS_CSV_COLUMNS)
        for a in actions:
            w.writerow(a.csv_row())


def write_actions_json(actions: list[Action], path: Path) -> None:
    path.write_text(json.dumps([a.model_dump() for a in actions], indent=1), encoding="utf-8")


def read_actions_json(path: Path) -> list[Action]:
    return [Action.model_validate(d) for d in json.loads(path.read_text(encoding="utf-8"))]
