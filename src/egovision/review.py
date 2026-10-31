"""Stage 9 human-in-the-loop review (FORGE). Reviewer decisions are an append-only layer in
OUT/<clip>/review.json; model output (actions.json / result.json) is never overwritten.

Decisions: approve | reject | edit (verb/object/hand/start_s/end_s) | split (at t) | merge (with the
next action of the same hand). `apply_review` replays the log over the model actions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from egovision.schema import Action

Op = Literal["approve", "reject", "edit", "split", "merge", "approve_trim"]
EDITABLE = ("verb", "object", "hand", "start_s", "end_s")
STATUS_RANK = {"rejected": 0, "unverified": 1, "corrected": 2, None: 3, "verified": 4}


class Decision(BaseModel):
    action_id: str | None = None  # None for clip-level decisions (approve_trim)
    op: Op
    fields: dict[str, Any] = Field(default_factory=dict)
    at_s: float | None = None  # split time
    by: str = "reviewer"
    ts: float = Field(default_factory=time.time)


def load_log(out_dir: Path) -> list[Decision]:
    p = out_dir / "review.json"
    if not p.is_file():
        return []
    return [Decision.model_validate(d) for d in json.loads(p.read_text(encoding="utf-8"))["decisions"]]


def append(out_dir: Path, d: Decision) -> None:
    log = load_log(out_dir)
    log.append(d)
    tmp = out_dir / ".review.json.tmp"
    tmp.write_text(json.dumps({"decisions": [x.model_dump() for x in log]}, indent=1), encoding="utf-8")
    tmp.replace(out_dir / "review.json")


def queue(actions: list[Action], log: list[Decision]) -> list[Action]:
    """Unreviewed actions first, most uncertain first: status, then flags, then low confidence."""
    reviewed = {d.action_id for d in log if d.action_id}
    return sorted(actions, key=lambda a: (a.id in reviewed, STATUS_RANK.get(a.status, 3), -len(a.flags),
                                          a.confidence if a.confidence is not None else 0.5, a.start_s))


def apply_review(actions: list[Action], log: list[Decision]) -> list[Action]:
    acts = {a.id: a for a in actions if a.id}
    order = [a.id for a in sorted(actions, key=lambda x: (x.start_s, x.hand)) if a.id]
    for d in log:
        a = acts.get(d.action_id or "")
        if a is None:
            continue
        rv = {"op": d.op, "by": d.by, "ts": d.ts}
        if d.op == "approve":
            acts[a.id] = a.model_copy(update={"review": {**rv, "status": "approved"}})
        elif d.op == "reject":
            acts[a.id] = a.model_copy(update={"review": {**rv, "status": "rejected"}})
        elif d.op == "edit":
            upd = {k: v for k, v in d.fields.items() if k in EDITABLE}
            acts[a.id] = Action.model_validate({**a.model_dump(), **upd, "review": {**rv, "status": "edited",
                                                                                  "changed": sorted(upd)}})
        elif d.op == "split" and d.at_s is not None and a.start_s < d.at_s < a.end_s:
            left = a.model_copy(update={"end_s": d.at_s, "review": {**rv, "status": "edited"}})
            right = a.model_copy(update={"id": f"{a.id}b", "start_s": d.at_s, "review": {**rv, "status": "edited"}})
            acts[a.id] = left
            acts[right.id] = right
            order.insert(order.index(a.id) + 1, right.id)
        elif d.op == "merge":
            same = [i for i in order[order.index(a.id) + 1:] if acts[i].hand == a.hand]
            if same:
                b = acts.pop(same[0])
                order.remove(b.id)
                acts[a.id] = a.model_copy(update={"end_s": max(a.end_s, b.end_s), "review": {**rv, "status": "edited"}})
    return sorted(acts.values(), key=lambda x: (x.start_s, x.hand))


def reviewed_export(actions: list[Action]) -> list[Action]:
    """What goes into actions_reviewed.csv: everything except actions the reviewer rejected."""
    return [a for a in actions if not (a.review and a.review.get("status") == "rejected")]


def progress(actions: list[Action], log: list[Decision]) -> dict[str, Any]:
    reviewed = {d.action_id for d in log if d.action_id}
    return {"total": len(actions), "reviewed": sum(a.id in reviewed for a in actions),
            "trim_approved": any(d.op == "approve_trim" for d in log)}
