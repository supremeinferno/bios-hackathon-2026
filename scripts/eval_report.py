"""Full evaluation report for one experiment folder (OUT/<clip>/ from `egovision run`).

    python scripts/eval_report.py --exp outputs_exp/E0_baseline --name E0_baseline --md docs/eval/E0_baseline.md

Scores every labelled clip, keeping the splits apart (labels/ = held-out, labels/dev = tuning), and adds
label-free hand diagnostics for every clip in the folder. Writes <exp>/report.json and a Markdown report.

PROVENANCE: the step labels are drafts made by Claude Code and NOT checked by a person (labels/README.md).
Every action number is agreement with those drafts, not accuracy. Hand numbers have no ground truth at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from egovision.evaluate import eval_steps, eval_verification
from egovision.export import read_actions_csv, read_actions_json, read_hands_jsonl
from egovision.metrics import (
    IDLE,
    MISSED,
    SPURIOUS,
    classification_from_confusion,
    hand_diagnostics,
    merge_confusions,
    verification_rates,
)
from egovision.schema import Action

SPLITS = {"held-out": Path("labels"), "dev": Path("labels/dev")}
HEAD = ["frame_acc", "frame_macro_f1", "F1@10", "F1@25", "F1@50", "edit", "verb_acc_matched", "object_acc_matched",
        "hand_acc_matched", "boundary_mae_s", "gt_segments_matched_pct"]


def clip_window(actions: list[Action], until: float) -> list[Action]:
    return [a.model_copy(update={"end_s": min(a.end_s, until)}) for a in actions if a.start_s < until - 0.05]


def labelled_until(labels: Path, clip: str, run_dir: Path) -> float:
    dur = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["processed_until_s"]
    meta = labels / f"{clip}.meta.json"
    if meta.is_file():
        dur = min(dur, float(json.loads(meta.read_text(encoding="utf-8"))["labelled_until_s"]))
    return dur


def score_split(exp: Path, labels: Path) -> dict[str, Any]:
    per: dict[str, Any] = {}
    for gt_file in sorted(labels.glob("*.actions.csv")):
        clip = gt_file.name.removesuffix(".actions.csv")
        run = exp / clip
        if not (run / "actions.csv").is_file():
            per[clip] = {"error": f"no prediction in {run}"}
            continue
        dur = labelled_until(labels, clip, run)
        gt = clip_window(read_actions_csv(gt_file), dur)
        pred = clip_window(read_actions_csv(run / "actions.csv"), dur)
        r = eval_steps(pred, gt, dur)
        r["duration_s"] = dur
        if (run / "actions.json").is_file():
            st = clip_window(read_actions_json(run / "actions.json"), dur)
            if any(a.status for a in st):
                r["verification"] = eval_verification(st, gt)
        per[clip] = r
    ok = [r for r in per.values() if "error" not in r]
    if not ok:
        return {"clips": per}
    fcm = merge_confusions(r["detail"]["frame_confusion"] for r in ok)
    scm = merge_confusions(r["detail"]["segment_confusion"] for r in ok)
    pooled: dict[str, Any] = {"frame": classification_from_confusion(fcm), "frame_confusion": fcm,
                              "segment_confusion": scm}
    for k in ("10", "25", "50"):
        tp, fp, fn = (sum(r["detail"][f"counts@{k}"][i] for r in ok) for i in range(3))
        pooled[f"F1@{k}_any_hand"] = round(100 * 2 * tp / (2 * tp + fp + fn), 1) if tp + fp + fn else 0.0
    vc: dict[str, dict[str, int]] = {}
    for r in ok:
        for st, c in r.get("verification", {}).get("counts_any_verb", {}).items():
            d = vc.setdefault(st, {"supported": 0, "unsupported": 0})
            d["supported"] += c["supported"]
            d["unsupported"] += c["unsupported"]
    if vc:
        pooled["verification_counts"] = vc
        pooled["verification_rates"] = verification_rates(vc)
    mean = {k: round(sum(r[k] for r in ok if r.get(k) is not None) / max(1, sum(r.get(k) is not None for r in ok)), 1)
            for k in HEAD}
    return {"clips": per, "pooled": pooled, "mean_of_clips": mean}


def hands_block(exp: Path) -> dict[str, Any]:
    out = {}
    for run in sorted(p for p in exp.iterdir() if (p / "hands.jsonl").is_file()):
        info = json.loads((run / "run.json").read_text(encoding="utf-8"))
        v = info["video"]
        stride = info["config"]["ingest"]["stride"]
        out[run.name] = hand_diagnostics(read_hands_jsonl(run / "hands.jsonl"), v["fps"], v["width"], stride)
        out[run.name]["hands_fps"] = info["timings"].get("hands_fps")
    return out


# ---- markdown -----------------------------------------------------------------------------------------
def md_table(header: list[str], rows: list[list[Any]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *["| " + " | ".join("-" if c is None else str(c) for c in r) + " |" for r in rows]]


def md_confusion(cm: dict[str, dict[str, int]], title: str) -> list[str]:
    rows_l = sorted(cm, key=lambda x: (x in (IDLE, SPURIOUS), x))
    cols_l = sorted({p for r in cm.values() for p in r} | set(rows_l) - {SPURIOUS}, key=lambda x: (x in (IDLE, MISSED), x))
    lines = [f"**{title}** (rows = label, columns = prediction)", ""]
    lines += md_table(["label \\ pred", *cols_l],
                      [[g, *[cm.get(g, {}).get(p, 0) or "·" for p in cols_l]] for g in rows_l])
    return lines + [""]


def to_markdown(name: str, rep: dict[str, Any]) -> str:
    L = [f"# Evaluation report: {name}", "",
         "> **Provenance.** Step labels are drafts by Claude Code, **not checked by a person** "
         "(labels/README.md). Action numbers below are *agreement with unverified drafts*, not accuracy. "
         "There is no hand ground truth; the hand section is label-free behaviour only.", ""]
    for split, s in rep["splits"].items():
        if not s.get("clips"):
            continue
        L += [f"## Split: {split}", "", "### Headline metrics per clip (original definitions)", ""]
        L += md_table(["clip", "n_pred", "n_gt", *HEAD],
                      [[c, r.get("n_pred"), r.get("n_gt"), *[r.get(k) for k in HEAD]] for c, r in s["clips"].items()])
        if "mean_of_clips" in s:
            L += md_table(["", "", "", *HEAD], [["**mean of clips**", "", "", *[s["mean_of_clips"][k] for k in HEAD]]])[2:]
        L += ["", "### Classification on the 10 Hz frame grid (pooled over lanes and clips)", ""]
        rows = []
        for c, r in [*s["clips"].items(), ("**pooled**", {"detail": {"frame": s.get("pooled", {}).get("frame", {})}})]:
            f = r.get("detail", {}).get("frame", {})
            rows.append([c, f.get("frames"), f.get("accuracy_all"), f.get("accuracy_active"), f.get("macro_f1"),
                         f.get("weighted_f1")])
        L += md_table(["clip", "frames (both lanes)", "accuracy (all)", "accuracy (active)", "macro-F1 (verbs)",
                       "weighted-F1 (verbs)"], rows)
        L += ["", "### One-to-one matched segments (Hungarian, IoU >= 0.25, label-agnostic)", ""]
        keys = ["n", "gt_recall", "pred_precision", "verb_acc", "hand_acc", "object_exact", "object_canonical",
                "object_lenient", "start_mae_s", "end_mae_s", "mean_iou"]
        L += md_table(["clip", *keys, "F1@10/25/50 any hand"],
                      [[c, *[r["detail"]["matched_1to1"].get(k) for k in keys],
                        "/".join(str(r["detail"][f"F1@{t}_any_hand"]) for t in ("10", "25", "50"))]
                       for c, r in s["clips"].items() if "detail" in r])
        if "pooled" in s:
            p = s["pooled"]
            L += ["", "### Per-class (frame grid, pooled)", ""]
            per = p["frame"]["per_class"]
            L += md_table(["class", "support (frames)", "precision", "recall", "F1"],
                          [[k, v["support"], v["precision"], v["recall"], v["f1"]]
                           for k, v in sorted(per.items(), key=lambda kv: -kv[1]["support"])])
            L += ["", "### Per-class segment detection (same verb, IoU >= 0.25, per lane, pooled)", ""]
            seg: dict[str, list[int]] = {}
            for r in s["clips"].values():
                for v, c in r.get("detail", {}).get("segment_per_class@25", {}).items():
                    acc = seg.setdefault(v, [0, 0, 0])
                    for i in range(3):
                        acc[i] += c[i]
            L += md_table(["verb", "tp", "fp", "fn", "precision", "recall"],
                          [[v, *c, round(100 * c[0] / (c[0] + c[1]), 1) if c[0] + c[1] else None,
                            round(100 * c[0] / (c[0] + c[2]), 1) if c[0] + c[2] else None]
                           for v, c in sorted(seg.items(), key=lambda kv: -(kv[1][0] + kv[1][2]))])
            L += ["", *md_confusion(p["frame_confusion"], "Frame confusion (10 Hz frames, both lanes)")]
            L += md_confusion(p["segment_confusion"], "Segment confusion (one-to-one matched; <missed> / <spurious>)")
            if "verification_rates" in p:
                L += ["### Stage 7 verification vs labels (supported = IoU >= 0.25 with any labelled segment)", ""]
                vc = p["verification_counts"]
                L += md_table(["status", "supported", "unsupported"],
                              [[k, v["supported"], v["unsupported"]] for k, v in sorted(vc.items())])
                L += [""] + md_table(list(p["verification_rates"]), [list(p["verification_rates"].values())])
            L += ["", "### Object pairs on matched segments (label → prediction, canonical form)", ""]
            for c, r in s["clips"].items():
                if "detail" in r:
                    L.append(f"- {c}: " + "; ".join(f"{g} → {pr} ×{n}" for (g, pr), n in r["detail"]["object_pairs"]))
        L.append("")
    L += ["## Hands (label-free diagnostics; no ground truth)", ""]
    hk = ["frames", "with_hand_pct", "two_hands_pct", "same_side_duplicate_pct", "left_right_crossed_pct",
          "side_matches_image_half_pct", "track_ids", "tracks_per_min", "median_track_s", "short_tracks_lt_0_5s",
          "wrist_jitter_median", "hands_fps"]
    L += md_table(["clip", *hk], [[c, *[d.get(k) for k in hk]] for c, d in rep["hands"].items()])
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--md", type=Path, default=None)
    a = ap.parse_args()
    rep = {"experiment": a.name or a.exp.name, "splits": {k: score_split(a.exp, v) for k, v in SPLITS.items()},
           "hands": hands_block(a.exp)}
    (a.exp / "report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    md = to_markdown(rep["experiment"], rep)
    if a.md:
        a.md.parent.mkdir(parents=True, exist_ok=True)
        a.md.write_text(md, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
