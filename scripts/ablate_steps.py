"""Ablation of the step stages from a finished run, without re-running the pipeline.

    python scripts/ablate_steps.py --runs outputs_gemini --labels labels/dev --json artifacts/measure/ablate_dev.json

For every clip in --runs that has labels, rebuild these timelines from saved files and score them:
  offline               offline exemplar matcher (actions_offline.json)
  gemini_pass1_only     Gemini 30 s windows, pass 1 only (rebuilt from the response cache)
  gemini_raw            Gemini windows after schema validation only (actions_vlm.json)
  gemini+verify         + Stage 7 verification (status/hand correction), no boundary snapping
  gemini+verify+snap    + Stage 8 boundary snapping (current default pipeline)
Decide on the dev labels; report on the held-out labels (labels/README.md).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

from egovision.config import Config
from egovision.evaluate import evaluate_dir
from egovision.export import write_actions_csv
from egovision.schema import Action, ContactSegment
from egovision.steps.gemini import stitch
from egovision.steps.postprocess import clean, from_raw
from egovision.verify import verify_and_refine

KEYS = ["n_pred", "frame_acc", "F1@10", "F1@25", "F1@50", "edit", "verb_acc_matched", "object_acc_matched",
        "hand_acc_matched", "boundary_mae_s", "gt_segments_matched_pct"]


def load_actions(p: Path) -> list[Action]:
    return [Action.model_validate(a) for a in json.loads(p.read_text(encoding="utf-8"))] if p.is_file() else []


def pass1_only(clip: str) -> list[Action]:
    """Rebuild the pass-1 timeline from the cached window responses (cache/api/<clip>/gemini_*.json)."""
    cfg = Config().steps
    acts: list[Action] = []
    for f in sorted((cfg.cache_dir / clip).glob("gemini_*.json")):
        c = json.loads(f.read_text(encoding="utf-8"))
        if c.get("pass") != 1:
            continue
        s, e = c["window"]
        for a in from_raw(c["parsed"], clip, cfg.verbs, source=f"gemini:{c['model']}:p1"):
            a_s, a_e = min(e, s + a.start_s), min(e, s + a.end_s)
            if a_e - a_s >= 0.2:
                acts.append(a.model_copy(update={"start_s": round(a_s, 2), "end_s": round(a_e, 2)}))
    return stitch(acts, cfg.stitch_gap_s)


def variants(run: Path) -> dict[str, list[Action]]:
    cfg = Config().verify
    with open(run / "signals.csv", encoding="utf-8") as f:
        sig = list(csv.DictReader(f))
    with open(run / "contacts.csv", encoding="utf-8") as f:
        contacts = [ContactSegment(clip=r["clip"], t_start=float(r["t_start"]), t_end=float(r["t_end"]), hand=r["hand"],
                                   state=r["state"], object_name=r["object_name"]) for r in csv.DictReader(f)]
    dur = json.loads((run / "run.json").read_text(encoding="utf-8"))["processed_until_s"]
    out = {"offline": load_actions(run / "actions_offline.json")}
    p1 = pass1_only(run.name)
    if p1:
        out["gemini_pass1_only"] = p1
    raw = load_actions(run / "actions_vlm.json")
    if raw:
        out["gemini_raw"] = raw
        out["gemini+verify"] = clean(verify_and_refine(raw, sig, contacts, cfg.model_copy(update={"snap": False})), dur)
        out["gemini+verify+snap"] = clean(verify_and_refine(raw, sig, contacts, cfg), dur)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("outputs_gemini"))
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    report: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for gt in sorted(a.labels.glob("*.actions.csv")):
            clip = gt.name.removesuffix(".actions.csv")
            run = a.runs / clip
            if not (run / "run.json").is_file():
                print(f"{clip}: no run in {a.runs}, skipped")
                continue
            report[clip] = {}
            for name, acts in variants(run).items():
                d = Path(tmp) / name / clip
                d.mkdir(parents=True, exist_ok=True)
                write_actions_csv(acts, d / "actions.csv")
                shutil.copy(run / "run.json", d / "run.json")
                report[clip][name] = evaluate_dir(Path(tmp) / name, a.labels)["steps"][clip]
    for clip, rows in report.items():
        print(f"\n{clip}\n| variant | " + " | ".join(KEYS) + " |")
        print("|---" * (len(KEYS) + 1) + "|")
        for name, r in rows.items():
            print(f"| {name} | " + " | ".join(str(r.get(k, "-")) for k in KEYS) + " |")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
