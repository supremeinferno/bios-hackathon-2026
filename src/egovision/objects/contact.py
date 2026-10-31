"""Hand-object association and contact timelines.

Rule (PDF step 5, our starting point): a hand touches an object when >= `contact_min_tips`
fingertips lie inside the object's box padded by `contact_margin` x hand size. If several
objects qualify, portable beats fixed (holding a tool while leaning on the bench), then more
fingertips, then higher detector score, then the smaller box.
"""

from __future__ import annotations

from egovision.config import ObjectsConfig
from egovision.hands import FINGERTIPS
from egovision.schema import Contact, ContactSegment, FrameObjects, Hand, ObjectDet


def _tips_inside(hand: Hand, box: list[float], pad: float) -> int:
    x1, y1, x2, y2 = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
    return sum(x1 < hand.kp2d[i][0] < x2 and y1 < hand.kp2d[i][1] < y2 for i in FINGERTIPS)


def associate(hands: list[Hand], objects: list[ObjectDet], cfg: ObjectsConfig) -> list[Contact]:
    contacts = []
    for h in hands:
        if h.id is None:
            continue
        size = max(h.bbox[2] - h.bbox[0], h.bbox[3] - h.bbox[1])
        pad = cfg.contact_margin * size
        best: tuple | None = None
        best_obj: ObjectDet | None = None
        best_tips = 0
        for o in objects:
            n = _tips_inside(h, o.bbox, pad)
            if n < cfg.contact_min_tips:
                continue
            area = (o.bbox[2] - o.bbox[0]) * (o.bbox[3] - o.bbox[1])
            key = (o.kind == "portable", n, o.score, -area)
            if best is None or key > best:
                best, best_obj, best_tips = key, o, n
        if best_obj is None:
            contacts.append(Contact(hand_id=h.id, side=h.side, state="none"))
        else:
            contacts.append(Contact(hand_id=h.id, side=h.side, state=best_obj.kind, object=best_obj.label,
                                    object_bbox=best_obj.bbox, object_score=best_obj.score, n_tips=best_tips))
    return contacts


def nearest_object(hand: Hand, objects: list[ObjectDet], max_dist: float) -> ObjectDet | None:
    """Closest object to the fingertips when the hand touches nothing: portable objects first, fixed ones only
    if no portable object is within `max_dist` hand sizes (a fixed conveyor fills the frame and would always
    win otherwise). Distance = fingertip centroid to the box edge (0 inside), in hand sizes."""
    size = max(hand.bbox[2] - hand.bbox[0], hand.bbox[3] - hand.bbox[1], 1.0)
    cx = sum(hand.kp2d[i][0] for i in FINGERTIPS) / len(FINGERTIPS)
    cy = sum(hand.kp2d[i][1] for i in FINGERTIPS) / len(FINGERTIPS)

    def dist(o: ObjectDet) -> float:
        dx = max(o.bbox[0] - cx, 0.0, cx - o.bbox[2])
        dy = max(o.bbox[1] - cy, 0.0, cy - o.bbox[3])
        return (dx * dx + dy * dy) ** 0.5 / size

    for kind in ("portable", "fixed"):
        cands = [(dist(o), -o.score, o) for o in objects if o.kind == kind]
        cands = [c for c in cands if c[0] <= max_dist]
        if cands:
            return min(cands, key=lambda c: (c[0], c[1]))[2]
    return None


def contact_segments(frames: list[FrameObjects], clip: str, step_s: float, min_s: float) -> list[ContactSegment]:
    """Per-side runs of (state, object) at keyframe resolution; runs shorter than `min_s` are absorbed."""
    out: list[ContactSegment] = []
    for side in ("left", "right"):
        runs: list[list] = []  # [state, object, t_start, t_end]
        for fo in frames:
            c = next((c for c in fo.contacts if c.side == side), None)
            key = (c.state, c.object or "") if c else ("none", "")
            t0, t1 = fo.t - step_s / 2, fo.t + step_s / 2
            if runs and (runs[-1][0], runs[-1][1]) == key and t0 - runs[-1][3] < step_s:
                runs[-1][3] = t1
            else:
                runs.append([key[0], key[1], max(0.0, t0), t1])
        merged: list[list] = []
        for r in runs:
            if merged and r[3] - r[2] < min_s:
                merged[-1][3] = r[3]  # too short: noise, extend the previous run
            elif merged and (merged[-1][0], merged[-1][1]) == (r[0], r[1]):
                merged[-1][3] = r[3]
            else:
                merged.append(list(r))
        out += [ContactSegment(clip=clip, t_start=round(s, 2), t_end=round(e, 2), hand=side,  # type: ignore[arg-type]
                               state=st, object_name=obj)
                for st, obj, s, e in merged if st != "none" and e - s >= min_s]
    return sorted(out, key=lambda c: (c.t_start, c.hand))
