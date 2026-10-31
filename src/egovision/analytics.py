"""Stage 11 analytics (FORGE): per-clip step durations, idle time, two-handed share, work zones,
placement positions and repetitive cycles. Positions are camera-relative (`frame: camera`)."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median
from typing import Any

import numpy as np

from egovision.schema import Action
from egovision.steps.postprocess import canonical_object

SIDES = ("left", "right")


def _lanes(a: Action) -> tuple[str, ...]:
    return {"left": ("left",), "right": ("right",)}.get(a.hand, SIDES)


def _covered(actions: list[Action], side: str, t: np.ndarray) -> np.ndarray:
    m = np.zeros(t.shape, bool)
    for a in actions:
        if side in _lanes(a) and a.status != "rejected":
            m |= (t >= a.start_s) & (t < a.end_s)
    return m


def clip_analytics(actions: list[Action], signal_rows: list[dict[str, Any]], duration_s: float) -> dict[str, Any]:
    acts = [a for a in actions if a.status != "rejected"]
    per_verb: dict[str, list[float]] = defaultdict(list)
    per_step: dict[str, list[float]] = defaultdict(list)
    for a in acts:
        per_verb[a.verb].append(a.end_s - a.start_s)
        per_step[f"{a.verb}:{canonical_object(a.object)}"].append(a.end_s - a.start_s)
    out: dict[str, Any] = {
        "frame": "camera", "duration_s": duration_s, "actions": len(acts),
        "time_per_verb_s": {v: {"total": round(sum(d), 1), "count": len(d), "mean": round(sum(d) / len(d), 2)}
                            for v, d in sorted(per_verb.items(), key=lambda kv: -sum(kv[1]))},
    }
    t = np.array([float(r["t"]) for r in signal_rows]) if signal_rows else np.arange(0, duration_s, 0.1)
    busy = {s: _covered(acts, s, t) for s in SIDES}
    out["busy_share"] = {s: round(float(busy[s].mean()), 3) for s in SIDES}
    out["two_handed_share"] = round(float((busy["left"] & busy["right"]).mean()), 3)
    if signal_rows:
        vis = {s: np.array([float(r[f"{s}_visible"] or 0) for r in signal_rows]) > 0 for s in SIDES}
        out["idle"] = {s: {"visible_no_action_share": round(float((vis[s] & ~busy[s]).mean()), 3),
                           "not_visible_share": round(float((~vis[s]).mean()), 3)} for s in SIDES}
        zones = {s: Counter(r[f"{s}_zone"] for r in signal_rows if r[f"{s}_zone"]) for s in SIDES}
        out["zone_share"] = {s: {z: round(c / max(sum(zones[s].values()), 1), 3) for z, c in zones[s].most_common()}
                             for s in SIDES}
        # placement coordinate: hand position at the end of each place action
        places = []
        for a in acts:
            if a.verb != "place":
                continue
            j = int(np.argmin(np.abs(t - a.end_s)))
            for s in _lanes(a):
                x, y = signal_rows[j].get(f"{s}_x"), signal_rows[j].get(f"{s}_y")
                if x not in ("", None):
                    places.append({"t": a.end_s, "hand": s, "object": a.object, "xy": [float(x), float(y)],
                                   "zone": signal_rows[j][f"{s}_zone"]})
        out["placements"] = places
    # repetitive cycles: most frequent step and its start-to-start interval
    if per_step:
        top, n = max(((k, len(v)) for k, v in per_step.items()), key=lambda kv: kv[1])
        starts = sorted(a.start_s for a in acts if f"{a.verb}:{canonical_object(a.object)}" == top)
        gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
        out["cycles"] = {"step": top, "occurrences": n,
                         "median_cycle_s": round(median(gaps), 1) if gaps else None,
                         "slow_cycles": [round(s, 1) for s, g in zip(starts, gaps, strict=False)
                                         if gaps and g > 1.5 * median(gaps)]}
    return out
