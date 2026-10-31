"""Temporal post-processing of per-frame hands: identity assignment and gap filling.

M1 baseline: identity = side (one left track, one right track; duplicates of a side keep the
higher score). Real tracking, swap fixing and smoothing arrive in M3.
"""

from __future__ import annotations

from egovision.schema import FrameHands, Hand

SIDE_ID = {"left": 0, "right": 1}


def assign_side_ids(frames: list[FrameHands]) -> list[FrameHands]:
    out = []
    for fh in frames:
        best: dict[str, Hand] = {}
        for h in fh.hands:
            if h.side not in best or h.score > best[h.side].score:
                best[h.side] = h
        hands = [h.model_copy(update={"id": SIDE_ID[s]}) for s, h in sorted(best.items())]
        out.append(FrameHands(frame=fh.frame, t=fh.t, hands=hands))
    return out


def _lerp(a: list[float], b: list[float], w: float) -> list[float]:
    return [x + (y - x) * w for x, y in zip(a, b, strict=True)]


def fill_gaps(processed: list[FrameHands], n_frames: int, fps: float, max_gap: int) -> list[FrameHands]:
    """Return one FrameHands per source frame in [0, n_frames).

    Frames between two processed frames get a hand (marked interp=True) only when the same id is
    present on both sides and the gap is <= max_gap source frames. Otherwise they stay empty:
    we never invent a hand we did not see on both sides.
    """
    by_frame = {fh.frame: fh for fh in processed}
    keys = sorted(by_frame)
    out: list[FrameHands] = []
    k = 0
    for i in range(n_frames):
        if i in by_frame:
            out.append(by_frame[i])
            continue
        while k + 1 < len(keys) and keys[k + 1] < i:
            k += 1
        prev_i = keys[k] if keys and keys[k] < i else None
        next_i = keys[k + 1] if k + 1 < len(keys) and keys[k + 1] > i else None
        hands: list[Hand] = []
        if prev_i is not None and next_i is not None and next_i - prev_i <= max_gap:
            w = (i - prev_i) / (next_i - prev_i)
            nxt = {h.id: h for h in by_frame[next_i].hands if h.id is not None}
            for a in by_frame[prev_i].hands:
                b = nxt.get(a.id)
                if b is None:
                    continue
                hands.append(Hand(
                    id=a.id, side=a.side, score=min(a.score, b.score), interp=True,
                    bbox=_lerp(a.bbox, b.bbox, w),
                    kp2d=[_lerp(p, q, w) for p, q in zip(a.kp2d, b.kp2d, strict=True)],
                ))
        out.append(FrameHands(frame=i, t=round(i / fps, 4), hands=hands))
    return out
