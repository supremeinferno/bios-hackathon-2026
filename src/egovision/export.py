"""Writers for the competition-facing files and our richer internal format.

Per clip, in OUT/<clip>/:
  actions.csv   PS3 suggested format, exactly (PDF p.4)            <- the PS3 deliverable
  hands.jsonl   PS1 suggested format (PDF p.3) + id/interp          <- hand tracking (judged in every PS)
  contacts.csv  PS2 suggested format (PDF p.4)                      <- grab/release timeline used by PS3
  result.json   rich internal record: meta, keyframe objects+contacts, actions with confidence/source
  features.npz  keyframe crop embeddings per side (for few-shot exemplars)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from egovision.schema import (
    ACTIONS_CSV_COLUMNS,
    CONTACTS_CSV_COLUMNS,
    Action,
    ContactSegment,
    FrameHands,
    FrameObjects,
    Hand,
    VideoMeta,
)


def _hand_dict(h: Hand, width: int | None = None, height: int | None = None) -> dict[str, Any]:
    d = h.model_dump()
    if width and height:  # FORGE Stage 4: wrist position in camera-origin coordinates
        from egovision.signals import to_camera_xy

        d["xy_cam"] = [round(v, 4) for v in to_camera_xy(h.kp2d[0][0], h.kp2d[0][1], width, height)]
    d["bbox"] = [round(v, 1) for v in d["bbox"]]
    d["kp2d"] = [[round(x, 1), round(y, 1)] for x, y in d["kp2d"]]
    d["score"] = round(d["score"], 3)
    return d


def write_hands_jsonl(frames: list[FrameHands], path: Path, width: int | None = None,
                      height: int | None = None) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for fh in frames:
            f.write(json.dumps({"frame": fh.frame, "t": round(fh.t, 3),
                                "hands": [_hand_dict(h, width, height) for h in fh.hands]}) + "\n")


def read_hands_jsonl(path: Path) -> list[FrameHands]:
    with open(path, encoding="utf-8") as f:
        return [FrameHands.model_validate_json(line) for line in f if line.strip()]


def write_actions_csv(actions: list[Action], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(ACTIONS_CSV_COLUMNS)
        for a in actions:
            w.writerow(a.csv_row())


def read_actions_csv(path: Path) -> list[Action]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = set(ACTIONS_CSV_COLUMNS) - set(rows[0].keys()) if rows else set()
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return [Action(clip=r["clip"], start_s=float(r["start_s"]), end_s=float(r["end_s"]), verb=r["verb"].strip(),
                   object=r["object"].strip(), hand=r["hand"].strip()) for r in rows]  # type: ignore[arg-type]


def write_signals_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Stage 4 signals at ~10 Hz in camera-origin coordinates (frame: camera)."""
    from egovision.signals import SIGNAL_COLUMNS

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SIGNAL_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_signals_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_contacts_csv(contacts: list[ContactSegment], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CONTACTS_CSV_COLUMNS)
        for c in contacts:
            w.writerow(c.csv_row())


def write_actions_json(actions: list[Action], path: Path) -> None:
    path.write_text(json.dumps([a.model_dump() for a in actions], indent=1), encoding="utf-8")


def read_actions_json(path: Path) -> list[Action]:
    return [Action.model_validate(d) for d in json.loads(path.read_text(encoding="utf-8"))]


def load_result(out_dir: Path) -> dict[str, Any]:
    """Reload a finished run (for the UI and make_demo): meta, hands, keyframe objects, contacts, actions."""
    r = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    run = out_dir / "run.json"
    return {
        "meta": VideoMeta.model_validate(r["video"]),
        "hands": read_hands_jsonl(out_dir / "hands.jsonl"),
        "objects": [FrameObjects.model_validate(k) for k in r["keyframes"]],
        "contacts": [ContactSegment.model_validate(c) for c in r["contacts"]],
        "actions": [Action.model_validate(a) for a in r["actions"]],
        "run": json.loads(run.read_text(encoding="utf-8")) if run.is_file() else {},
        "sanitized": r.get("files", {}).get("sanitized_video"),
    }


def write_outputs(out_dir: Path, meta: VideoMeta, hands: list[FrameHands], objects: list[FrameObjects],
                  contacts: list[ContactSegment], actions: list[Action], keyframes: list[Any],
                  sanitized: Path | None = None) -> None:
    write_hands_jsonl(hands, out_dir / "hands.jsonl", meta.width, meta.height)
    write_actions_csv(actions, out_dir / "actions.csv")
    write_contacts_csv(contacts, out_dir / "contacts.csv")
    write_actions_json(actions, out_dir / "actions.json")
    result = {
        "clip": meta.name,
        "video": meta.model_dump(),
        "files": {"hands": "hands.jsonl", "actions": "actions.csv", "contacts": "contacts.csv",
                  "sanitized_video": sanitized.as_posix() if sanitized else None},
        "keyframes": [fo.model_dump() for fo in objects],
        "contacts": [c.model_dump() for c in contacts],
        "actions": [a.model_dump() for a in actions],
    }
    (out_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    embs = {s: [kf.sides[s].emb if s in kf.sides and kf.sides[s].emb is not None else None for kf in keyframes]
            for s in ("left", "right")}
    dim = next((e.shape[0] for v in embs.values() for e in v if e is not None), 0)
    if dim:
        arrs = {s: np.stack([e if e is not None else np.full(dim, np.nan, np.float32) for e in v]).astype(np.float16)
                for s, v in embs.items()}
        np.savez_compressed(out_dir / "features.npz", t=np.array([kf.t for kf in keyframes]), **arrs)
