"""Turn one validated analyser reply into stored events.

Rules (all deterministic, so a batch can be replayed from its saved raw reply):
- Timestamps are snapped to the frames sampled in this batch; a move larger than `snap_tolerance_s` is logged
  as a correction. Events outside the batch (by more than 1.5 sample intervals) are rejected.
- An event that continues an open event (by `continues_event_id`, or the same worker + action + object seen from
  the first frame of the batch) extends that event instead of creating a new one.
- Two events of the same worker, action and object that overlap within one batch are merged.
- Open events that are not continued are closed as completed; on the last batch, still-running events become
  incomplete.
- verification_status: low_confidence (confidence < min_confidence) > unverified (seen in one frame only) >
  corrected (the backend changed something) > verified.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from egovision.steps.postprocess import canonical_object
from egovision.studio.db import Database
from egovision.studio.models import BatchReply, EventRecord, ReplyEvent, ReplyWorker
from egovision.studio.settings import StudioSettings


@dataclass
class MergeResult:
    created: list[EventRecord] = field(default_factory=list)
    updated: list[EventRecord] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    workers_changed: bool = False
    notes: str = ""


def verification(ev: EventRecord, s: StudioSettings) -> str:
    if ev.confidence < s.min_confidence:
        return "low_confidence"
    if ev.frames_seen < 2:
        return "unverified"
    if ev.corrections:
        return "corrected"
    return "verified"


def _errors(e: ValidationError) -> list[str]:
    return [f"{'.'.join(str(p) for p in err['loc']) or 'item'}: {err['msg']}" for err in e.errors()][:5]


def _snap(v: float, ts: list[float], every: float) -> tuple[float | None, float]:
    """Nearest sampled timestamp and the distance to it; None when far outside the batch."""
    near = min(ts, key=lambda t: abs(t - v))
    if v < ts[0] - 1.5 * every or v > ts[-1] + 1.5 * every:
        return None, abs(near - v)
    return near, abs(near - v)


def _same_task(a: EventRecord, worker: str, action: str, obj: str) -> bool:
    return a.worker_id == worker and a.action == action and canonical_object(a.object) == canonical_object(obj)


def apply_reply(db: Database, s: StudioSettings, video: dict[str, Any], batch_index: int, frame_ts: list[float],
                reply: dict[str, Any], open_events: dict[str, EventRecord], end_s: float, last_batch: bool,
                source: str = "") -> MergeResult:
    """Validate and store one reply. `open_events` (event_id -> event still in progress) is updated in place.

    Raises ValidationError if the reply envelope itself is malformed (the batch is then marked failed).
    """
    vid = video["video_id"]
    every = float(video["sample_every_s"])
    sop = video.get("sop") or []
    env = BatchReply.model_validate(reply)
    res = MergeResult(notes=env.notes[:500])
    now = time.time()

    for w in env.workers:
        try:
            cw = ReplyWorker.model_validate(w)
        except ValidationError as e:
            res.rejected.append({"kind": "worker", "item": w, "errors": _errors(e)})
            continue
        db.upsert_worker(vid, cw.worker_id, cw.description, cw.is_camera_wearer)
        res.workers_changed = True
    known = {w["worker_id"] for w in db.list_workers(vid)}

    continued: set[str] = set()
    batch_events: list[EventRecord] = []  # created or continued in this batch, for duplicate merging
    for item in env.events:
        try:
            ce = ReplyEvent.model_validate(item)
        except ValidationError as e:
            res.rejected.append({"kind": "event", "item": item, "errors": _errors(e)})
            continue
        start, d_start = _snap(ce.start_ts, frame_ts, every)
        end, d_end = _snap(ce.end_ts, frame_ts, every)
        if start is None or end is None:
            res.rejected.append({"kind": "event", "item": item, "errors": [
                f"timestamps {ce.start_ts:.1f}-{ce.end_ts:.1f} s are outside this batch "
                f"({frame_ts[0]:.1f}-{frame_ts[-1]:.1f} s)"]})
            continue
        if end < start:
            end = start
        corrections: list[str] = []
        if ce.worker_id not in known:
            db.upsert_worker(vid, ce.worker_id, None, None)
            known.add(ce.worker_id)
            res.workers_changed = True
            corrections.append(f"worker {ce.worker_id} was not in the roster; added")
        sop_step = ce.sop_step
        if sop_step is not None and (not sop or sop_step > len(sop)):
            corrections.append(f"sop_step {sop_step} does not exist; cleared")
            sop_step = None
        n_seen = sum(start - 1e-6 <= t <= end + 1e-6 for t in frame_ts)
        end_ts = min(end_s, round(end + every, 3))
        completion = {"completed": "completed", "in_progress": "in_progress", "interrupted": "incomplete"}[ce.status]

        # ---- continuation of an open event -----------------------------------------------------
        target = open_events.get(ce.continues_event_id or "")
        if target is not None and (target.event_id in continued or not _same_task(target, ce.worker_id, ce.action,
                                                                                    ce.object)):
            corrections.append(f"continues_event_id {ce.continues_event_id} ignored (different task or already continued)")
            target = None
        elif ce.continues_event_id and target is None:
            corrections.append(f"continues_event_id {ce.continues_event_id} is not an open event; started a new event")
        if target is None and not ce.continues_event_id and start <= frame_ts[0] + 1e-6:
            target = next((o for o in open_events.values() if o.event_id not in continued and o.last_batch < batch_index
                           and _same_task(o, ce.worker_id, ce.action, ce.object)), None)
            if target is not None:
                corrections.append(f"linked to open event {target.event_id} (same worker, action and object)")
        if target is None:  # same task reported twice in this batch -> merge
            target = next((b for b in batch_events if _same_task(b, ce.worker_id, ce.action, ce.object)
                           and start <= b.end_ts + 1e-6), None)
            if target is not None:
                corrections.append(f"merged with overlapping {target.event_id} from the same batch")

        if target is not None:
            n_old = target.frames_seen
            ev = target.model_copy(update={
                "end_ts": max(target.end_ts, end_ts),
                "frames_seen": n_old + n_seen,
                "confidence": round((target.confidence * n_old + ce.confidence * n_seen) / max(n_old + n_seen, 1), 3),
                "tool": target.tool or ce.tool,
                "sop_step": target.sop_step or sop_step,
                "completion_status": completion,
                "evidence": target.evidence or ce.evidence,
                "corrections": target.corrections + corrections,
                "last_batch": batch_index, "updated_at": now})
            ev.duration_s = round(ev.end_ts - ev.start_ts, 3)
            ev.verification_status = verification(ev, s)  # type: ignore[assignment]
            continued.add(ev.event_id)
            db.save_event(ev)
            open_events.pop(ev.event_id, None)
            batch_events = [b for b in batch_events if b.event_id != ev.event_id] + [ev]
            if not any(c.event_id == ev.event_id for c in res.created):
                res.updated = [u for u in res.updated if u.event_id != ev.event_id] + [ev]
            else:
                res.created = [ev if c.event_id == ev.event_id else c for c in res.created]
            if completion == "in_progress":
                open_events[ev.event_id] = ev
            continue

        if d_start > s.snap_tolerance_s:
            corrections.append(f"start {ce.start_ts:.2f} s moved to the sampled frame at {start:.2f} s")
        if d_end > s.snap_tolerance_s:
            corrections.append(f"end {ce.end_ts:.2f} s moved to the sampled frame at {end:.2f} s")
        seq = db.next_seq(vid)
        ev = EventRecord(event_id=f"E{seq:04d}", video_id=vid, seq=seq, worker_id=ce.worker_id, action=ce.action,
                         object=ce.object, tool=ce.tool, start_ts=start, end_ts=end_ts,
                         duration_s=round(end_ts - start, 3), confidence=ce.confidence, verification_status="verified",
                         completion_status=completion, frames_seen=max(n_seen, 1), sop_step=sop_step,  # type: ignore[arg-type]
                         evidence=ce.evidence, corrections=corrections, source=source, first_batch=batch_index,
                         last_batch=batch_index, created_at=now, updated_at=now)
        ev.verification_status = verification(ev, s)  # type: ignore[assignment]
        db.save_event(ev)
        res.created.append(ev)
        batch_events.append(ev)
        if completion == "in_progress":
            open_events[ev.event_id] = ev

    # Open events from earlier batches that were not seen again ended at the batch boundary.
    for eid in [e for e in open_events if e not in continued and all(c.event_id != e for c in res.created)]:
        ev = open_events.pop(eid).model_copy(update={"completion_status": "completed", "updated_at": now})
        db.save_event(ev)
        res.updated.append(ev)
    if last_batch:
        finish_open(db, open_events, res)
    for ev in res.created + res.updated:
        db.upsert_worker(vid, ev.worker_id, None, None, ev.start_ts, ev.end_ts)
    return res


def finish_open(db: Database, open_events: dict[str, EventRecord], res: MergeResult | None = None) -> list[EventRecord]:
    """The analysed range ended while these tasks were still running: they are incomplete."""
    done = []
    for eid in list(open_events):
        ev = open_events.pop(eid).model_copy(update={"completion_status": "incomplete", "updated_at": time.time()})
        db.save_event(ev)
        done.append(ev)
        if res is not None:
            if any(c.event_id == eid for c in res.created):
                res.created = [ev if c.event_id == eid else c for c in res.created]
            else:
                res.updated = [u for u in res.updated if u.event_id != eid] + [ev]
    return done
