"""Evaluation against our own, disclosed labels (there is no official PS3 metric; SPEC.md §3, A1).

Steps (labels/<clip>.actions.csv, same columns as actions.csv):
  * frame-wise verb accuracy and macro-F1 on a 10 Hz grid, per hand lane (left / right; 'both'
    fills both lanes, gaps are 'idle')
  * segmental F1@{0.10,0.25,0.50} per lane (MS-TCN convention: same label + IoU >= tau)
  * edit score (normalised Levenshtein over each lane's segment-label sequence)
  * on GT segments matched to a predicted segment by time (IoU >= 0.25, any label, any hand):
    verb / object / hand accuracy and mean boundary error
Hands (labels/hands_judged.csv; PDF p.4 "label hands on 100 frames yourself"):
  * recall, precision, side accuracy and keypoint-OK rate from per-frame judgements
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from egovision.schema import Action

TAUS = (0.10, 0.25, 0.50)
IDLE = "idle"
GRID_HZ = 10.0


# ---- steps ------------------------------------------------------------------------------------
def lane_of(a: Action) -> tuple[str, ...]:
    return {"left": ("left",), "right": ("right",)}.get(a.hand, ("left", "right"))


def lane_segments(actions: list[Action], lane: str) -> list[Action]:
    return sorted([a for a in actions if lane in lane_of(a)], key=lambda a: a.start_s)


def frame_labels(actions: list[Action], lane: str, duration: float) -> list[str]:
    n = int(np.ceil(duration * GRID_HZ))
    lab = [IDLE] * n
    for a in lane_segments(actions, lane):
        for i in range(int(a.start_s * GRID_HZ), min(n, int(np.ceil(a.end_s * GRID_HZ)))):
            lab[i] = a.verb
    return lab


def _seg_iou(a: Action, b: Action) -> float:
    inter = max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))
    union = max(a.end_s, b.end_s) - min(a.start_s, b.start_s)
    return inter / union if union > 0 else 0.0


def segmental_counts(pred: list[Action], gt: list[Action], tau: float) -> tuple[int, int, int]:
    used = set()
    tp = 0
    for p in sorted(pred, key=lambda a: -(a.confidence or 0)):
        best, best_j = 0.0, None
        for j, g in enumerate(gt):
            if j in used or g.verb != p.verb:
                continue
            iou = _seg_iou(p, g)
            if iou > best:
                best, best_j = iou, j
        if best_j is not None and best >= tau:
            tp += 1
            used.add(best_j)
    return tp, len(pred) - tp, len(gt) - tp


def edit_score(pred: list[str], gt: list[str]) -> float:
    m, n = len(pred), len(gt)
    if max(m, n) == 0:
        return 100.0
    d = np.zeros((m + 1, n + 1))
    d[:, 0], d[0, :] = np.arange(m + 1), np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (pred[i - 1] != gt[j - 1]))
    return float((1 - d[m, n] / max(m, n)) * 100)


def _norm_obj(o: str) -> set[str]:
    return set(re.findall(r"[a-z]+", o.lower())) - {"a", "an", "the", "of", "piece", "small", "large"}


def object_match(p: str, g: str) -> bool:
    """Lenient: any shared content word ('aluminium profile' ~ 'profile')."""
    if g in ("", "unknown"):
        return p in ("", "unknown")
    return bool(_norm_obj(p) & _norm_obj(g))


def eval_steps(pred: list[Action], gt: list[Action], duration: float) -> dict[str, Any]:
    res: dict[str, Any] = {"n_pred": len(pred), "n_gt": len(gt)}
    correct = total = 0
    verbs_seen: set[str] = set()
    per_verb = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn on frames
    edits = []
    counts = {tau: [0, 0, 0] for tau in TAUS}
    for lane in ("left", "right"):
        pl, gl = frame_labels(pred, lane, duration), frame_labels(gt, lane, duration)
        for p, g in zip(pl, gl, strict=True):
            if p == IDLE and g == IDLE:
                continue  # only score time where someone (pred or GT) says something happens
            total += 1
            correct += p == g
            verbs_seen |= {p, g}
            if p == g:
                per_verb[g][0] += 1
            else:
                per_verb[p][1] += 1
                per_verb[g][2] += 1
        ps, gs = lane_segments(pred, lane), lane_segments(gt, lane)
        edits.append(edit_score([a.verb for a in ps], [a.verb for a in gs]))
        for tau in TAUS:
            tp, fp, fn = segmental_counts(ps, gs, tau)
            counts[tau][0] += tp
            counts[tau][1] += fp
            counts[tau][2] += fn
    res["frame_acc"] = round(100 * correct / total, 1) if total else None
    f1s = []
    for v in sorted(verbs_seen - {IDLE}):
        tp, fp, fn = per_verb[v]
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    res["frame_macro_f1"] = round(100 * float(np.mean(f1s)), 1) if f1s else None
    for tau, (tp, fp, fn) in counts.items():
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        res[f"F1@{int(tau * 100)}"] = round(100 * (2 * prec * rec / (prec + rec) if prec + rec else 0.0), 1)
    res["edit"] = round(float(np.mean(edits)), 1)
    # attribute accuracy on time-matched GT segments
    v_ok = o_ok = h_ok = matched = 0
    bound = []
    for g in gt:
        cands = [(p, _seg_iou(p, g)) for p in pred]
        p, iou = max(cands, key=lambda x: x[1]) if cands else (None, 0.0)
        if p is None or iou < 0.25:
            continue
        matched += 1
        v_ok += p.verb == g.verb
        o_ok += object_match(p.object, g.object)
        h_ok += p.hand == g.hand
        bound += [abs(p.start_s - g.start_s), abs(p.end_s - g.end_s)]
    res["gt_segments_matched_pct"] = round(100 * matched / len(gt), 1) if gt else None
    res["verb_acc_matched"] = round(100 * v_ok / matched, 1) if matched else None
    res["object_acc_matched"] = round(100 * o_ok / matched, 1) if matched else None
    res["hand_acc_matched"] = round(100 * h_ok / matched, 1) if matched else None
    res["boundary_mae_s"] = round(float(np.mean(bound)), 2) if bound else None
    return res


# ---- hands (judged frames) ------------------------------------------------------------------------
HAND_JUDGE_COLUMNS = ["clip", "frame", "gt_hands", "correct", "false_pos", "side_errors", "kp_bad", "notes"]


def eval_hands_judged(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("gt_hands", "").strip() != ""]
    if not rows:
        return {"frames": 0}

    def agg(rs: list[dict[str, str]]) -> dict[str, Any]:
        s = {k: sum(int(r[k] or 0) for r in rs) for k in ("gt_hands", "correct", "false_pos", "side_errors", "kp_bad")}
        return {
            "frames": len(rs), "gt_hands": s["gt_hands"],
            "recall": round(100 * s["correct"] / s["gt_hands"], 1) if s["gt_hands"] else None,
            "precision": round(100 * s["correct"] / (s["correct"] + s["false_pos"]), 1)
            if s["correct"] + s["false_pos"] else None,
            "side_acc": round(100 * (1 - s["side_errors"] / s["correct"]), 1) if s["correct"] else None,
            "kp_ok": round(100 * (1 - s["kp_bad"] / s["correct"]), 1) if s["correct"] else None,
        }

    by_src: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_src[r["clip"].rsplit("-", 1)[0]].append(r)
    return {"all": agg(rows), "by_source": {k: agg(v) for k, v in sorted(by_src.items())}}


# ---- driver ---------------------------------------------------------------------------------------
def clip_window(actions: list[Action], until: float) -> list[Action]:
    return [a.model_copy(update={"end_s": min(a.end_s, until)}) for a in actions if a.start_s < until - 0.05]


def evaluate_dir(pred_root: Path, labels_dir: Path) -> dict[str, Any]:
    from egovision.export import read_actions_csv

    report: dict[str, Any] = {"steps": {}, "hands": None}
    for gt_file in sorted(labels_dir.glob("*.actions.csv")):
        clip = gt_file.name.removesuffix(".actions.csv")
        pred_file = pred_root / clip / "actions.csv"
        if not pred_file.is_file():
            report["steps"][clip] = {"error": f"no prediction at {pred_file}"}
            continue
        gt = read_actions_csv(gt_file)
        pred = read_actions_csv(pred_file)
        duration = max([a.end_s for a in gt + pred] + [0.0])
        run = pred_root / clip / "run.json"
        if run.is_file():
            duration = json.loads(run.read_text())["processed_until_s"]
        meta = labels_dir / f"{clip}.meta.json"
        if meta.is_file():  # only score the window that was actually labelled
            duration = min(duration, float(json.loads(meta.read_text())["labelled_until_s"]))
        pred = clip_window(pred, duration)
        gt = clip_window(gt, duration)
        report["steps"][clip] = eval_steps(pred, gt, duration)
    if report["steps"]:
        keys = [k for k in next(iter(report["steps"].values())) if k not in ("n_pred", "n_gt", "error")]
        vals = [r for r in report["steps"].values() if "error" not in r]
        report["steps_mean"] = {k: round(float(np.mean([r[k] for r in vals if r.get(k) is not None])), 1)
                                for k in keys if any(r.get(k) is not None for r in vals)}
    judged = labels_dir / "hands_judged.csv"
    if judged.is_file():
        report["hands"] = eval_hands_judged(judged)
    return report


def format_report(rep: dict[str, Any]) -> str:
    lines = []
    if rep.get("steps"):
        cols = ["frame_acc", "frame_macro_f1", "F1@10", "F1@25", "F1@50", "edit", "verb_acc_matched",
                "object_acc_matched", "hand_acc_matched", "boundary_mae_s", "gt_segments_matched_pct"]
        lines.append("| clip | " + " | ".join(cols) + " |")
        lines.append("|---" * (len(cols) + 1) + "|")
        for clip, r in rep["steps"].items():
            lines.append(f"| {clip} | " + " | ".join(str(r.get(c, "-")) for c in cols) + " |")
        if "steps_mean" in rep:
            lines.append("| **mean** | " + " | ".join(str(rep["steps_mean"].get(c, "-")) for c in cols) + " |")
    if rep.get("hands") and rep["hands"].get("all"):
        h = rep["hands"]
        lines.append("")
        lines.append("| hands (judged) | frames | gt hands | recall | precision | side acc | kp ok |")
        lines.append("|---|---|---|---|---|---|---|")
        for name, r in [("all", h["all"]), *h["by_source"].items()]:
            lines.append(f"| {name} | {r['frames']} | {r['gt_hands']} | {r['recall']} | {r['precision']} | "
                         f"{r['side_acc']} | {r['kp_ok']} |")
    return "\n".join(lines) if lines else "no labels found"
