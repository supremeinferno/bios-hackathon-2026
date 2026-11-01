"""Metric definitions used by `egovision.evaluate` and `scripts/eval_report.py`.

Every metric here is defined once, with its convention written next to it, so experiments are compared
with the same yardstick. Nothing here knows where labels came from; callers must state provenance
(labels/README.md: the step labels are unverified drafts, so their scores are agreement, not accuracy).

Actions
  frame grid      10 Hz, one label per hand lane (left / right); 'both' fills both lanes; gaps are 'idle'.
                  Confusion matrix over verbs + idle, pooled over lanes (and clips when aggregated).
  frame metrics   accuracy over all frames, accuracy over frames where GT or prediction is not idle
                  (the historical `frame_acc`), per-class precision / recall / F1, macro-F1 over verbs that
                  occur in GT or prediction, weighted-F1 over verbs weighted by GT support. Idle is a class
                  in the matrix (idle confusions count as misses / false alarms) but not in the averages.
  segment match   one-to-one Hungarian matching of GT and predicted segments on temporal IoU (>= 0.25),
                  label-agnostic, so verb / object / hand / boundary errors are measured on the same pairs.
                  Unmatched GT = missed, unmatched prediction = spurious.
  segment F1@k    per lane, same verb, IoU >= k (MS-TCN convention; unchanged from the first version).
Hands (need human GT in labels/hands/<clip>.hands.jsonl; see docs/ANNOTATION_PLAN.md)
  detection       per frame, Hungarian on box IoU >= 0.5 -> TP / FP / FN -> precision, recall, F1
  handedness      share of matched hands with the right side
  landmarks       mean keypoint error in px and normalised by GT box size, PCK@0.1 / 0.2; DETECTED and
                  INTERPOLATED predictions are reported separately and never pooled
  tracking        ID switches (CLEAR-MOT: a GT track matched to a different predicted id than last time)
Label-free hand diagnostics (no GT; describe behaviour, do not measure accuracy) are in `hand_diagnostics`.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from egovision.schema import Action, FrameHands

IDLE = "idle"
GRID_HZ = 10.0
MATCH_IOU = 0.25


# ---- shared helpers ---------------------------------------------------------------------------------
def seg_iou(a: Action, b: Action) -> float:
    inter = max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))
    union = max(a.end_s, b.end_s) - min(a.start_s, b.start_s)
    return inter / union if union > 0 else 0.0


def lanes(hand: str) -> tuple[str, ...]:
    """Hand lanes an action occupies. 'both', 'none' and 'unknown' occupy both lanes (historical rule)."""
    return {"left": ("left",), "right": ("right",)}.get(hand, ("left", "right"))


def prf(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    return {"precision": _pct(p), "recall": _pct(r), "f1": _pct(f), "tp": tp, "fp": fp, "fn": fn}


def _pct(x: float | None) -> float | None:
    return None if x is None else round(100 * x, 1)


# ---- frame grid ------------------------------------------------------------------------------------
def frame_grid(actions: Sequence[Action], lane: str, duration: float, hz: float = GRID_HZ) -> list[str]:
    """Per-lane label sequence; later (start-sorted) actions overwrite earlier ones where they overlap."""
    n = int(math.ceil(duration * hz))
    lab = [IDLE] * n
    for a in sorted((a for a in actions if lane in lanes(a.hand)), key=lambda a: a.start_s):
        for i in range(int(a.start_s * hz), min(n, int(math.ceil(a.end_s * hz)))):
            lab[i] = a.verb
    return lab


def confusion(pairs: Iterable[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """{gt_label: {pred_label: count}} from (gt, pred) pairs."""
    m: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for g, p in pairs:
        m[g][p] += 1
    return {g: dict(r) for g, r in m.items()}


def merge_confusions(ms: Iterable[dict[str, dict[str, int]]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for m in ms:
        for g, row in m.items():
            for p, c in row.items():
                out[g][p] += c
    return {g: dict(r) for g, r in out.items()}


def frame_pairs(pred: Sequence[Action], gt: Sequence[Action], duration: float) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for lane in ("left", "right"):
        out += list(zip(frame_grid(gt, lane, duration), frame_grid(pred, lane, duration), strict=True))
    return out


def classification_from_confusion(cm: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Accuracy, per-class P/R/F1, macro- and weighted-F1 (verbs only; idle is a class but not averaged)."""
    labels = sorted(set(cm) | {p for r in cm.values() for p in r})
    total = sum(c for r in cm.values() for c in r.values())
    correct = sum(cm.get(lab, {}).get(lab, 0) for lab in labels)
    active = sum(c for g, r in cm.items() for p, c in r.items() if not (g == IDLE and p == IDLE))
    per: dict[str, Any] = {}
    for lab in labels:
        tp = cm.get(lab, {}).get(lab, 0)
        fn = sum(cm.get(lab, {}).values()) - tp
        fp = sum(r.get(lab, 0) for g, r in cm.items() if g != lab)
        per[lab] = {**prf(tp, fp, fn), "support": tp + fn}
    verbs = [lab for lab in labels if lab != IDLE]
    f1s = [(per[v]["f1"] or 0.0) for v in verbs]
    sup = [per[v]["support"] for v in verbs]
    return {
        "frames": total,
        "accuracy_all": _pct(correct / total) if total else None,
        "accuracy_active": _pct((correct - cm.get(IDLE, {}).get(IDLE, 0)) / active) if active else None,
        "macro_f1": round(float(np.mean(f1s)), 1) if f1s else None,
        "weighted_f1": round(float(np.average(f1s, weights=sup)), 1) if sum(sup) else None,
        "per_class": per,
    }


# ---- segment matching --------------------------------------------------------------------------------
def match_segments(pred: Sequence[Action], gt: Sequence[Action], min_iou: float = MATCH_IOU) -> list[tuple[int, int, float]]:
    """One-to-one (pred_idx, gt_idx, iou) maximising total IoU; ties broken toward hand-compatible pairs."""
    if not pred or not gt:
        return []
    iou = np.array([[seg_iou(p, g) for g in gt] for p in pred])
    compat = np.array([[bool(set(lanes(p.hand)) & set(lanes(g.hand))) for g in gt] for p in pred], dtype=float)
    score = np.where(iou >= min_iou, iou + 1e-3 * compat, 0.0)
    rows, cols = linear_sum_assignment(-score)
    return [(int(r), int(c), float(iou[r, c])) for r, c in zip(rows, cols, strict=True) if iou[r, c] >= min_iou]


MISSED, SPURIOUS = "<missed>", "<spurious>"


def segment_confusion(pred: Sequence[Action], gt: Sequence[Action], min_iou: float = MATCH_IOU) -> dict[str, dict[str, int]]:
    """Verb confusion on matched pairs; unmatched GT -> column <missed>, unmatched pred -> row <spurious>."""
    m = match_segments(pred, gt, min_iou)
    pairs = [(gt[g].verb, pred[p].verb) for p, g, _ in m]
    mp, mg = {p for p, _, _ in m}, {g for _, g, _ in m}
    pairs += [(g.verb, MISSED) for j, g in enumerate(gt) if j not in mg]
    pairs += [(SPURIOUS, p.verb) for i, p in enumerate(pred) if i not in mp]
    return confusion(pairs)


def segmental_counts(pred: Sequence[Action], gt: Sequence[Action], tau: float, verb: str | None = None) -> tuple[int, int, int]:
    """Greedy (by confidence) same-verb matching at IoU >= tau within one lane. verb=None: all verbs."""
    if verb is not None:
        pred = [a for a in pred if a.verb == verb]
        gt = [a for a in gt if a.verb == verb]
    used: set[int] = set()
    tp = 0
    for p in sorted(pred, key=lambda a: -(a.confidence or 0)):
        best, best_j = 0.0, None
        for j, g in enumerate(gt):
            if j in used or g.verb != p.verb:
                continue
            iou = seg_iou(p, g)
            if iou > best:
                best, best_j = iou, j
        if best_j is not None and best >= tau:
            tp += 1
            used.add(best_j)
    return tp, len(pred) - tp, len(gt) - tp


def lane_actions(actions: Sequence[Action], lane: str) -> list[Action]:
    return sorted([a for a in actions if lane in lanes(a.hand)], key=lambda a: a.start_s)


def per_class_segmental(pred: Sequence[Action], gt: Sequence[Action], tau: float) -> dict[str, list[int]]:
    """{verb: [tp, fp, fn]} at IoU >= tau, per lane, pooled."""
    out: dict[str, list[int]] = {}
    verbs = {a.verb for a in [*pred, *gt]}
    for v in sorted(verbs):
        c = [0, 0, 0]
        for lane in ("left", "right"):
            tp, fp, fn = segmental_counts(lane_actions(pred, lane), lane_actions(gt, lane), tau, v)
            c[0] += tp
            c[1] += fp
            c[2] += fn
        out[v] = c
    return out


def hand_agnostic_counts(pred: Sequence[Action], gt: Sequence[Action], tau: float) -> tuple[int, int, int]:
    """Segment F1 counts ignoring hands entirely (each action counted once, same verb required)."""
    return segmental_counts(list(pred), list(gt), tau)


# ---- verification ------------------------------------------------------------------------------------
ACCEPTED = ("verified", "corrected")


def verification_counts(pred: Sequence[Action], gt: Sequence[Action], min_iou: float = MATCH_IOU,
                        same_verb: bool = False) -> dict[str, dict[str, int]]:
    """{status: {'supported': n, 'unsupported': n}}. Supported = overlaps some GT segment with IoU >= min_iou
    (any hand; same verb too if `same_verb`). Statuses: verified / corrected / unverified / rejected / none."""
    out: dict[str, dict[str, int]] = defaultdict(lambda: {"supported": 0, "unsupported": 0})
    for p in pred:
        ok = any(seg_iou(p, g) >= min_iou and (not same_verb or g.verb == p.verb) for g in gt)
        out[p.status or "none"]["supported" if ok else "unsupported"] += 1
    return dict(out)


def verification_rates(counts: dict[str, dict[str, int]]) -> dict[str, Any]:
    sup = sum(v["supported"] for v in counts.values())
    uns = sum(v["unsupported"] for v in counts.values())
    rej_s = counts.get("rejected", {}).get("supported", 0)
    rej_u = counts.get("rejected", {}).get("unsupported", 0)
    flag_s = rej_s + counts.get("unverified", {}).get("supported", 0)
    acc_u = sum(counts.get(s, {}).get("unsupported", 0) for s in ACCEPTED)
    return {
        "supported": sup, "unsupported": uns,
        # rejected although a labelled action is there (the costly error: work silently disappears)
        "false_rejection_rate": _pct(rej_s / sup) if sup else None,
        # rejected or unverified although a labelled action is there (sent to review needlessly)
        "false_flag_rate": _pct(flag_s / sup) if sup else None,
        # accepted (verified/corrected) although no labelled action is there
        "false_acceptance_rate": _pct(acc_u / uns) if uns else None,
        "rejection_precision": _pct(rej_u / (rej_s + rej_u)) if rej_s + rej_u else None,
    }


# ---- hands with GT -------------------------------------------------------------------------------------
def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def eval_hands_gt(pred: Sequence[FrameHands], gt: Sequence[dict[str, Any]], min_iou: float = 0.5) -> dict[str, Any]:
    """Score predicted hands on human-annotated frames.

    gt rows: {"frame": int, "hands": [{"id": str|int, "side": "left"|"right", "bbox": [x1,y1,x2,y2],
              "kp2d": [[x,y]*21] | null}]}; frames listed with an empty "hands" list are true negatives.
    Predictions on frames that are not annotated are ignored. Detected and interpolated predictions are
    scored separately (`by_source`); the headline detection numbers count every prediction the pipeline
    outputs, because that is what a user receives.
    """
    by_frame = {fh.frame: fh for fh in pred}
    tp = fp = fn = 0
    side_ok = 0
    src_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # source -> [tp, fp]
    kp_err: dict[str, list[float]] = defaultdict(list)  # source -> normalised errors per hand
    kp_px: dict[str, list[float]] = defaultdict(list)
    pck: dict[str, list[tuple[float, float]]] = defaultdict(list)
    last_pred_id: dict[Any, Any] = {}
    id_switches = 0
    for row in sorted(gt, key=lambda r: r["frame"]):
        ph = by_frame[row["frame"]].hands if row["frame"] in by_frame else []
        gh = row.get("hands") or []
        if ph and gh:
            iou = np.array([[box_iou(p.bbox, g["bbox"]) for g in gh] for p in ph])
            r, c = linear_sum_assignment(-iou)
            pairs = [(i, j) for i, j in zip(r, c, strict=True) if iou[i, j] >= min_iou]
        else:
            pairs = []
        tp += len(pairs)
        fp += len(ph) - len(pairs)
        fn += len(gh) - len(pairs)
        matched_p = {i for i, _ in pairs}
        for i, p in enumerate(ph):
            src_counts[p.source][0 if i in matched_p else 1] += 1
        for i, j in pairs:
            p, g = ph[i], gh[j]
            side_ok += p.side == g["side"]
            if g.get("kp2d"):
                gk, pk = np.asarray(g["kp2d"], float), np.asarray(p.kp2d, float)
                size = max(g["bbox"][2] - g["bbox"][0], g["bbox"][3] - g["bbox"][1], 1.0)
                d = np.linalg.norm(gk - pk, axis=1)
                kp_px[p.source].append(float(d.mean()))
                kp_err[p.source].append(float(d.mean() / size))
                pck[p.source].append((float((d / size < 0.1).mean()), float((d / size < 0.2).mean())))
            gid = g.get("id")
            if gid is not None and p.id is not None:
                if gid in last_pred_id and last_pred_id[gid] != p.id:
                    id_switches += 1
                last_pred_id[gid] = p.id
    return {
        "frames": len(gt), "detection": prf(tp, fp, fn),
        "handedness_acc": _pct(side_ok / tp) if tp else None,
        "precision_by_source": {s: _pct(v[0] / (v[0] + v[1])) for s, v in src_counts.items() if sum(v)},
        "landmarks_by_source": {s: {"hands": len(kp_err[s]), "mean_px": round(float(np.mean(kp_px[s])), 1),
                                    "mean_norm": round(float(np.mean(kp_err[s])), 3),
                                    "pck@0.1": _pct(float(np.mean([a for a, _ in pck[s]]))),
                                    "pck@0.2": _pct(float(np.mean([b for _, b in pck[s]])))}
                                for s in kp_err if kp_err[s]},
        "id_switches": id_switches,
    }


# ---- label-free hand diagnostics -------------------------------------------------------------------
def hand_diagnostics(frames: Sequence[FrameHands], fps: float, width: int, stride: int = 2) -> dict[str, Any]:
    """Behaviour of the hand stage without ground truth. NOT accuracy: a higher detection share can mean
    more false hands, so pair any change with a visual spot check (scripts/hand_audit.py).

    Only frames the hand model ran on (frame % stride == 0) count for detection shares; interpolated hands
    never count as detections.
    """
    fr = [fh for fh in frames if fh.frame % stride == 0]
    n = max(len(fr), 1)
    det = [[h for h in fh.hands if h.source != "interpolated"] for fh in fr]
    dup_side = sum(1 for hs in det if len(hs) != len({h.side for h in hs}))
    crossed = 0
    for hs in det:
        lw = [h.kp2d[0][0] for h in hs if h.side == "left"]
        rw = [h.kp2d[0][0] for h in hs if h.side == "right"]
        crossed += bool(lw and rw and min(lw) > max(rw) + 0.05 * width)
    on_half = [(h.kp2d[0][0] < width / 2) == (h.side == "left") for hs in det for h in hs]
    life: dict[int, list[float]] = defaultdict(list)
    for fh in frames:
        for h in fh.hands:
            if h.id is not None:
                life[h.id].append(fh.t)
    lengths = [max(ts) - min(ts) for ts in life.values()]
    dur_min = max(len(frames) / fps / 60, 1e-9)
    # jitter: wrist second difference over three consecutive processed detections of one id, / hand size
    by_id: dict[int, list[tuple[int, np.ndarray, float]]] = defaultdict(list)
    for fh in fr:
        for h in fh.hands:
            if h.id is not None and h.source != "interpolated":
                size = max(h.bbox[2] - h.bbox[0], h.bbox[3] - h.bbox[1], 1.0)
                by_id[h.id].append((fh.frame, np.asarray(h.kp2d[0]), size))
    acc = [float(np.linalg.norm(a - 2 * b + c) / s)
           for seq in by_id.values()
           for (f0, a, _), (f1, b, s), (f2, c, _) in zip(seq, seq[1:], seq[2:], strict=False)
           if f1 - f0 == stride and f2 - f1 == stride]
    sources = Counter(h.source for fh in frames for h in fh.hands)
    return {
        "frames": len(fr),
        "with_hand_pct": _pct(sum(bool(hs) for hs in det) / n),
        "two_hands_pct": _pct(sum(len(hs) >= 2 for hs in det) / n),
        "same_side_duplicate_pct": _pct(dup_side / n),
        "left_right_crossed_pct": _pct(crossed / n),
        "side_matches_image_half_pct": _pct(float(np.mean(on_half))) if on_half else None,
        "track_ids": len(life),
        "tracks_per_min": round(len(life) / dur_min, 1),
        "median_track_s": round(float(np.median(lengths)), 2) if lengths else None,
        "short_tracks_lt_0_5s": sum(x < 0.5 for x in lengths),
        "wrist_jitter_median": round(float(np.median(acc)), 4) if acc else None,
        "hand_sources": dict(sources),
    }
