"""Stage 8 step order and completeness (FORGE): auto-discover a per-source step template (SOP) from
all clips of that recording source, then align each clip to it and flag missing / out-of-order /
repeated steps. Flags are reported, never used to reorder or invent steps.

Caveat for this dataset: clips are 3-minute slices of one long recording per source, so a slice can
legitimately start or end mid-cycle; "missing" at the start or end of a clip is expected there.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from egovision.config import SOPConfig
from egovision.schema import Action


def step_key(a: Action) -> str:
    obj = re.sub(r"\s+", " ", a.object.lower()).strip() or "unknown"
    return f"{a.verb}:{obj}"


def sequence(actions: list[Action], cfg: SOPConfig) -> list[str]:
    """Clip as a step sequence: rejected actions dropped, consecutive repeats collapsed."""
    seq: list[str] = []
    for a in sorted(actions, key=lambda x: x.start_s):
        if a.status == "rejected" or a.verb in cfg.ignore_verbs:
            continue
        k = step_key(a)
        if not seq or seq[-1] != k:
            seq.append(k)
    return seq


def discover(clips: dict[str, tuple[list[Action], float]], cfg: SOPConfig) -> list[dict[str, Any]]:
    """Template = steps present in >= min_support of clips, ordered by median relative first position."""
    support: dict[str, int] = defaultdict(int)
    pos: dict[str, list[float]] = defaultdict(list)
    for actions, duration in clips.values():
        seen: set[str] = set()
        for a in sorted(actions, key=lambda x: x.start_s):
            if a.status == "rejected" or a.verb in cfg.ignore_verbs:
                continue
            k = step_key(a)
            if k not in seen:
                seen.add(k)
                support[k] += 1
                pos[k].append(a.start_s / max(duration, 1e-6))
    n = max(len(clips), 1)
    keep = [k for k, c in support.items() if c / n >= cfg.min_support and c >= cfg.min_clips]
    keep.sort(key=lambda k: median(pos[k]))
    return [{"step_id": f"S{i + 1}", "key": k, "support": round(support[k] / n, 2),
             "median_rel_pos": round(median(pos[k]), 3)} for i, k in enumerate(keep)]


def lcs(a: list[str], b: list[str]) -> list[tuple[int, int]]:
    """Index pairs of a longest common subsequence (monotone alignment: steps cannot swap order)."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    out, i, j = [], 0, 0
    while i < m and j < n:
        if a[i] == b[j]:
            out.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def check(actions: list[Action], template: list[dict[str, Any]], cfg: SOPConfig) -> dict[str, Any]:
    tkeys = [t["key"] for t in template]
    seq = sequence(actions, cfg)
    pairs = lcs(seq, tkeys)
    matched_seq = {i for i, _ in pairs}
    matched_t = {j for _, j in pairs}
    counts: dict[str, int] = defaultdict(int)
    for k in seq:
        counts[k] += 1
    tset = set(tkeys)
    return {
        "sequence": seq,
        "matched": [template[j]["step_id"] for j in sorted(matched_t)],
        "missing": [template[j]["step_id"] for j in range(len(tkeys)) if j not in matched_t],
        "out_of_order": sorted({seq[i] for i in range(len(seq)) if i not in matched_seq and seq[i] in tset}),
        "repeated": sorted(k for k, c in counts.items() if c > 1 and k in tset),
        "extra": sorted({k for k in seq if k not in tset}),
        "completeness": round(len(matched_t) / len(tkeys), 2) if tkeys else None,
    }


def source_of(clip: str) -> str:
    return clip.rsplit("-", 1)[0]


def run_sop(out_root: Path, cfg: SOPConfig) -> dict[str, Any]:
    by_src: dict[str, dict[str, tuple[list[Action], float]]] = defaultdict(dict)
    for res in sorted(out_root.glob("*/result.json")):
        r = json.loads(res.read_text(encoding="utf-8"))
        acts = [Action.model_validate(a) for a in r["actions"]]
        dur = float(r["video"]["duration_s"])
        by_src[source_of(r["clip"])][r["clip"]] = (acts, dur)
    report: dict[str, Any] = {}
    (out_root / "sop").mkdir(parents=True, exist_ok=True)
    for src, clips in sorted(by_src.items()):
        template = discover(clips, cfg)
        per_clip = {c: check(a, template, cfg) for c, (a, _) in clips.items()}
        report[src] = {"clips": len(clips), "template": template, "per_clip": per_clip,
                       "note": "Template discovered from our own timelines; clips are 3-min slices of one recording."}
        (out_root / "sop" / f"{src}.json").write_text(json.dumps(report[src], indent=1), encoding="utf-8")
        for c, chk in per_clip.items():
            (out_root / c / "steps_check.json").write_text(json.dumps({"source": src, **chk}, indent=1),
                                                          encoding="utf-8")
    return report
