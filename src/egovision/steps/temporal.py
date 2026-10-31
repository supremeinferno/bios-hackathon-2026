"""Temporal decoding and timeline rules shared by step backends."""

from __future__ import annotations

import numpy as np

from egovision.schema import Action, ContactSegment


def log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def viterbi(emissions: np.ndarray, switch_penalty: float) -> np.ndarray:
    """Best state path for (K, S) log-emissions with a constant cost per label change."""
    k_len, s_len = emissions.shape
    if k_len == 0:
        return np.zeros(0, dtype=int)
    score = emissions[0].copy()
    back = np.zeros((k_len, s_len), dtype=int)
    trans = -switch_penalty * (1 - np.eye(s_len))
    for k in range(1, k_len):
        cand = score[:, None] + trans  # from x to
        back[k] = cand.argmax(axis=0)
        score = cand.max(axis=0) + emissions[k]
    path = np.zeros(k_len, dtype=int)
    path[-1] = int(score.argmax())
    for k in range(k_len - 1, 0, -1):
        path[k - 1] = back[k, path[k]]
    return path


def runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """[(label, first_index, last_index)] for consecutive equal labels."""
    out: list[tuple[int, int, int]] = []
    for i, lab in enumerate(labels.tolist()):
        if out and out[-1][0] == lab:
            out[-1] = (lab, out[-1][1], i)
        else:
            out.append((lab, i, i))
    return out


def drop_short(rs: list[tuple[int, int, int]], times: np.ndarray, step: float, min_s: float,
               idle: int) -> list[tuple[int, int, int]]:
    """Relabel non-idle runs shorter than min_s as idle, then re-merge."""
    labels = np.empty(len(times), dtype=int)
    for lab, a, b in rs:
        short = (times[b] - times[a] + step) < min_s
        labels[a:b + 1] = idle if (short and lab != idle) else lab
    return runs(labels)


def subtract(a: Action, cuts: list[tuple[float, float]], min_s: float) -> list[Action]:
    """Pieces of `a` outside the given intervals, each at least min_s long."""
    pieces = [(a.start_s, a.end_s)]
    for c0, c1 in cuts:
        nxt = []
        for s, e in pieces:
            if c1 <= s or c0 >= e:
                nxt.append((s, e))
                continue
            if c0 > s:
                nxt.append((s, c0))
            if c1 < e:
                nxt.append((c1, e))
        pieces = nxt
    return [a.model_copy(update={"start_s": round(s, 2), "end_s": round(e, 2)}) for s, e in pieces if e - s >= min_s]


def carve_pick_place(actions: list[Action], contacts: list[ContactSegment], length: float,
                     min_s: float) -> list[Action]:
    """Grab -> 'pick <obj>' at contact start, release -> 'place <obj>' at contact end (portable only)."""
    if length <= 0:
        return actions
    events: list[Action] = []
    for c in contacts:
        if c.state != "portable":
            continue
        dur = c.t_end - c.t_start
        p_end = c.t_start + min(length, dur)
        events.append(Action(clip=c.clip, start_s=round(c.t_start, 2), end_s=round(p_end, 2), verb="pick",
                             object=c.object_name, hand=c.hand, confidence=None, source="contact"))
        if dur > 2 * length:
            events.append(Action(clip=c.clip, start_s=round(c.t_end - length, 2), end_s=round(c.t_end, 2),
                                 verb="place", object=c.object_name, hand=c.hand, confidence=None, source="contact"))
    out: list[Action] = []
    for a in actions:
        cuts = [(e.start_s, e.end_s) for e in events if e.hand == a.hand]
        out += subtract(a, cuts, min_s)
    return out + events


def temporal_iou(a: Action, b: Action) -> float:
    inter = max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))
    union = max(a.end_s, b.end_s) - min(a.start_s, b.start_s)
    return inter / union if union > 0 else 0.0


def merge_both(actions: list[Action], iou_min: float) -> list[Action]:
    """Left and right segments doing the same verb on the same object become one 'both' action."""
    left = [a for a in actions if a.hand == "left"]
    right = [a for a in actions if a.hand == "right"]
    rest = [a for a in actions if a.hand not in ("left", "right")]
    used_r: set[int] = set()
    out: list[Action] = []
    for a in left:
        match = None
        for j, b in enumerate(right):
            if j not in used_r and b.verb == a.verb and b.object == a.object and temporal_iou(a, b) >= iou_min:
                match = j
                break
        if match is None:
            out.append(a)
            continue
        b = right[match]
        used_r.add(match)
        conf = [c for c in (a.confidence, b.confidence) if c is not None]
        out.append(a.model_copy(update={"hand": "both", "start_s": min(a.start_s, b.start_s),
                                        "end_s": max(a.end_s, b.end_s),
                                        "confidence": round(sum(conf) / len(conf), 3) if conf else None}))
    out += [b for j, b in enumerate(right) if j not in used_r] + rest
    return sorted(out, key=lambda a: (a.start_s, a.hand))
