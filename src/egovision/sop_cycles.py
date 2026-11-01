"""Cycle consistency, labour cost and automation candidate indicator per task (FORGE Stage 8 extension).

Describes TASKS and STEPS, never individual people. The automation candidate indicator flags steps whose
recorded pattern looks repetitive and stable; it is not a recommendation to replace anyone.

Pipeline (one source = one recording site; config `sop_cycles`, egovision/config.py SOPCyclesConfig):
1. Actions: only `statuses` (default: verified, Stage 7 in verify.py) count; the share excluded is reported.
   Clips from another labeller than the source's majority (run.json steps_backend) are excluded and listed.
2. Task groups: clips whose verified step sets overlap (Jaccard >= group_min_jaccard) form one task; one source
   can hold several tasks (MTL-SKL00089 does). Override with `groups`.
3. Anchor: a step that is not a background state, repeats in >= min_anchor_clips clips and maximises
   intervals x (1 - min(CV of its start-to-start period, 1)). Same start-to-start idea as analytics.py:65-73, but
   one anchor per task (not per clip) and never a `hold`. Override with `anchors`.
4. Complete cycle = one anchor start to the next. Segments before the first and after the last anchor of a clip
   are partial and never counted. A cycle must contain >= min_cycle_step_share of the task template.
5. Fewer than min_cycles complete cycles -> "insufficient data", no score.

Metrics (per task group):
- Template: sop.discover (sop.py:38) run over the cycles themselves (each cycle shifted to start at 0), so steps
  are ordered by their median relative position WITHIN the cycle.
- Order distance per cycle: Levenshtein(cycle step sequence, template) / max(len), 0 = identical, 1 = different.
- Timing: per step, CV = population std / mean of its time per cycle (union of its intervals); CV of cycle time.
- Consistency score = 100 * (w_order * (1 - mean order distance) + w_timing * (1 - min(CV_cycle / cv_cap, 1)))
  / (w_order + w_timing).
- Labour: labour hours per cycle (wage-independent); cost = hours x team-declared wage, or null if undeclared.
  Idle = cycle time covered by no non-rejected action, plus `wait`; time covered only by actions outside
  `statuses` is reported separately as unverified time, not as idle.
- Step order distance (CSV): share of the cycles containing the step in which it is NOT on the longest common
  subsequence with the template (sop.lcs, sop.py:59), i.e. done out of order.
- Indicator: see `step_indicator` / `task_indicator`.
- Flags: label_noise (the step's occurrences often overlap a differently labelled action of the same hand),
  name_variants (same verb, other object name, in cycles where the step is missing), few_cycles.
"""

from __future__ import annotations

import csv
import io
import json
import statistics as st
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from egovision.config import SOPConfig, SOPCyclesConfig
from egovision.schema import Action
from egovision.sop import discover, lcs, sequence, source_of, step_key

CAVEATS = [
    "Labels come from automatic timelines checked only against unverified draft labels.",
    "Variability can come from inconsistent model labels as well as from the task itself (see flags).",
    "Clips are 3-minute slices; only complete cycles between two anchor occurrences are counted.",
    "The wage is a team-declared assumption, not a measurement.",
    "The automation candidate indicator describes steps of a task; it is not a decision to replace anyone.",
]
STEP_COLUMNS = ["group", "step", "n_cycles", "presence_pct", "median_s", "cv_pct", "order_distance", "labour_h_per_cycle",
                "cost_per_cycle", "currency", "indicator", "flags"]


@dataclass
class ClipData:
    actions: list[Action]
    duration_s: float
    labeller: str = ""


# ---- small helpers ----------------------------------------------------------------------------------------------
def levenshtein(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def order_distance(seq: list[str], template: list[str]) -> float:
    return levenshtein(seq, template) / max(len(seq), len(template), 1)


def cv(xs: list[float]) -> float | None:
    """Coefficient of variation (population std / mean); None for fewer than 2 values or a zero mean."""
    if len(xs) < 2 or st.mean(xs) <= 0:
        return None
    return st.pstdev(xs) / st.mean(xs)


def union_s(intervals: list[tuple[float, float]], lo: float, hi: float) -> float:
    """Length of the union of intervals clipped to [lo, hi)."""
    segs = sorted((max(s, lo), min(e, hi)) for s, e in intervals if min(e, hi) > max(s, lo))
    total, cur_s, cur_e = 0.0, None, None
    for s, e in segs:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s  # type: ignore[operator]
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s  # type: ignore[operator]
    return total


def _r(x: float | None, nd: int = 3) -> float | None:
    return None if x is None else round(x, nd)


def _lanes(a: Action) -> set[str]:
    return {"left"} if a.hand == "left" else {"right"} if a.hand == "right" else {"left", "right"}


def _iou(a: Action, b: Action) -> float:
    inter = max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))
    union = max(a.end_s, b.end_s) - min(a.start_s, b.start_s)
    return inter / union if union > 0 else 0.0


# ---- steps 1-4: filtering, task groups, anchor, cycles ---------------------------------------------------------
def kept_actions(clip: ClipData, cfg: SOPCyclesConfig) -> list[Action]:
    return [a for a in clip.actions if a.status in cfg.statuses and a.verb not in cfg.ignore_verbs]


def task_groups(source: str, kept: dict[str, list[Action]], cfg: SOPCyclesConfig) -> list[list[str]]:
    if source in cfg.groups:
        return [[c for c in g if c in kept] for g in cfg.groups[source] if any(c in kept for c in g)]
    sets = {c: {step_key(a) for a in acts} for c, acts in kept.items()}
    groups, seen = [], set()
    for c in sorted(kept):
        if c in seen:
            continue
        g, stack = [], [c]
        while stack:  # single-link clustering on step-set overlap
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            g.append(x)
            stack += [y for y in sorted(kept) if y not in seen and sets[x] and sets[y]
                      and len(sets[x] & sets[y]) / len(sets[x] | sets[y]) >= cfg.group_min_jaccard]
        groups.append(sorted(g))
    return sorted(groups, key=lambda g: (-len(g), g[0]))


def occurrences(acts: list[Action], key: str, merge_gap_s: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a in sorted(acts, key=lambda x: x.start_s):
        if step_key(a) != key:
            continue
        if out and a.start_s - out[-1][1] < merge_gap_s:
            out[-1] = (out[-1][0], max(out[-1][1], a.end_s))
        else:
            out.append((a.start_s, a.end_s))
    return out


def choose_anchor(group: list[str], kept: dict[str, list[Action]], cfg: SOPCyclesConfig,
                  override: str | None) -> dict[str, Any] | None:
    keys = sorted({step_key(a) for c in group for a in kept[c] if a.verb not in cfg.anchor_exclude_verbs})
    best = None
    for k in ([override] if override else keys):
        occ = {c: occurrences(kept[c], k, cfg.merge_gap_s) for c in group}
        periods = [b[0] - a[0] for c in group for a, b in zip(occ[c], occ[c][1:], strict=False)]
        n_clips = sum(len(o) >= 2 for o in occ.values())
        if not override and (not periods or n_clips < min(cfg.min_anchor_clips, len(group))):
            continue
        c_v = cv(periods)
        score = len(periods) * (1 - min(c_v if c_v is not None else 1.0, 1.0))
        cand = {"step": k, "intervals": len(periods), "period_median_s": _r(st.median(periods), 2) if periods else None,
                "period_cv": _r(c_v), "score": round(score, 3), "chosen_by": "config" if override else "auto"}
        if best is None or (score, len(periods)) > (best["score"], best["intervals"]):
            best = cand
    return best


@dataclass
class Cycle:
    clip: str
    start: float
    end: float
    actions: list[Action]  # kept actions starting inside the cycle (absolute times)

    @property
    def length(self) -> float:
        return self.end - self.start


def split_cycles(group: list[str], kept: dict[str, list[Action]], anchor: str, cfg: SOPCyclesConfig) -> list[Cycle]:
    """Anchor start -> next anchor start. Partial segments at the clip edges are never returned."""
    out = []
    for c in group:
        occ = occurrences(kept[c], anchor, cfg.merge_gap_s)
        for (s, _), (e, _) in zip(occ, occ[1:], strict=False):
            out.append(Cycle(c, s, e, [a for a in kept[c] if s <= a.start_s < e]))
    return out


def cycle_template(cycles: list[Cycle], cfg: SOPCyclesConfig) -> list[dict[str, Any]]:
    """sop.discover over cycles as pseudo-clips: step order = median relative position within the cycle."""
    pseudo = {f"{cy.clip}@{cy.start:.2f}": ([a.model_copy(update={"start_s": a.start_s - cy.start,
                                                                   "end_s": a.end_s - cy.start}) for a in cy.actions],
                                            cy.length) for cy in cycles}
    return discover(pseudo, SOPConfig(min_support=cfg.template_min_support, min_clips=1, ignore_verbs=cfg.ignore_verbs))


# ---- step 5 + metrics ----------------------------------------------------------------------------------------------
def step_indicator(verb: str, presence: float, step_cv: float | None, task_distance: float, n: int,
                   cfg: SOPCyclesConfig) -> str:
    """candidate: stable, always present, automatable verb, task order consistent. keep manual: fine manipulation /
    judgement verb, or very variable timing. Everything else: needs review."""
    if n < cfg.min_cycles:
        return "insufficient data"
    if verb in cfg.manual_verbs or (step_cv is not None and step_cv > cfg.manual_min_cv):
        return "keep manual"
    if (verb in cfg.automatable_verbs and presence >= cfg.candidate_min_presence and step_cv is not None
            and step_cv <= cfg.candidate_max_cv and task_distance <= cfg.candidate_max_order_distance):
        return "candidate"
    return "needs review"


def task_indicator(score: float, steps: list[dict[str, Any]], cycle_median_s: float, cfg: SOPCyclesConfig) -> str:
    share = {k: sum(s["median_s"] or 0 for s in steps if s["indicator"] == k) / max(cycle_median_s, 1e-9)
             for k in ("candidate", "keep manual")}
    if score >= cfg.task_candidate_min_score and share["candidate"] >= cfg.task_min_time_share:
        return "candidate"
    if share["keep manual"] >= cfg.task_min_time_share:
        return "keep manual"
    return "needs review"


def wage_of(cfg: SOPCyclesConfig) -> dict[str, Any]:
    w = cfg.wage
    declared = bool(w.currency and w.rate_per_hour)
    return {"declared": declared, "currency": w.currency if declared else None,
            "rate_per_hour": w.rate_per_hour if declared else None,
            "note": w.note if declared else "wage not declared: costs are null (set sop_cycles.wage)"}


def cost(hours: float | None, wage: dict[str, Any]) -> float | None:
    return None if hours is None or not wage["declared"] else round(hours * wage["rate_per_hour"], 4)


def analyse_group(gid: str, group: list[str], clips: dict[str, ClipData], kept: dict[str, list[Action]],
                  cfg: SOPCyclesConfig, wage: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = gid.split("/")[0]
    out: dict[str, Any] = {"group": gid, "clips": group}
    anchor = choose_anchor(group, kept, cfg, cfg.anchors.get(gid) or cfg.anchors.get(source))
    out["anchor"] = anchor
    if anchor is None:
        out.update(complete_cycles=0, status="insufficient data",
                   reason="no step repeats regularly enough to mark cycle starts")
        return out, []
    candidates = split_cycles(group, kept, anchor["step"], cfg)
    template = cycle_template(candidates, cfg) if candidates else []
    tkeys = [t["key"] for t in template]
    cycles = [cy for cy in candidates
              if tkeys and len({step_key(a) for a in cy.actions} & set(tkeys)) / len(tkeys) >= cfg.min_cycle_step_share]
    if len(cycles) >= cfg.min_cycles and len(cycles) < len(candidates):
        template = cycle_template(cycles, cfg)  # template from the cycles that are actually used
        tkeys = [t["key"] for t in template]
    out.update(candidate_cycles=len(candidates), complete_cycles=len(cycles),
               dropped_sub_cycles=len(candidates) - len(cycles),
               cycles_per_clip=dict(Counter(cy.clip for cy in cycles)), template=template)
    if len(cycles) < cfg.min_cycles:
        out.update(status="insufficient data", reason=f"{len(cycles)} complete cycles (< {cfg.min_cycles})")
        return out, []

    seq_cfg = SOPConfig(ignore_verbs=cfg.ignore_verbs)
    rows, step_time = [], defaultdict(list)
    for cy in cycles:
        seq = sequence(cy.actions, seq_cfg)
        matched = {seq[i] for i, _ in lcs(seq, tkeys)}
        every = [a for a in clips[cy.clip].actions if a.status != "rejected"]
        busy = union_s([(a.start_s, a.end_s) for a in every if a.verb != "wait"], cy.start, cy.end)
        wait = union_s([(a.start_s, a.end_s) for a in every if a.verb == "wait"], cy.start, cy.end)
        kept_busy = union_s([(a.start_s, a.end_s) for a in kept[cy.clip]], cy.start, cy.end)
        times = {}
        for k in set(seq):
            times[k] = union_s([(a.start_s, a.end_s) for a in cy.actions if step_key(a) == k], cy.start, cy.end)
            step_time[k].append((times[k], k in matched))
        hours = cy.length / 3600
        rows.append({"clip": cy.clip, "start_s": round(cy.start, 2), "end_s": round(cy.end, 2),
                     "cycle_s": round(cy.length, 2), "sequence": seq,
                     "order_distance": round(order_distance(seq, tkeys), 3),
                     "idle_s": round(cy.length - busy + wait, 2), "unverified_s": round(max(busy - kept_busy, 0), 2),
                     "labour_h": round(hours, 5), "cost": cost(hours, wage)})
    dists = [r["order_distance"] for r in rows]
    lengths = [r["cycle_s"] for r in rows]
    cv_cycle = cv(lengths)
    mean_d = st.mean(dists)
    timing = 1 - min((cv_cycle or 0.0) / cfg.cv_cap, 1.0)
    score = 100 * (cfg.w_order * (1 - mean_d) + cfg.w_timing * timing) / max(cfg.w_order + cfg.w_timing, 1e-9)
    idle_share = sum(r["idle_s"] for r in rows) / sum(lengths)

    steps = []
    for k in tkeys + sorted(set(step_time) - set(tkeys)):
        vals = step_time.get(k, [])
        durs = [d for d, _ in vals]
        presence = len(vals) / len(cycles)
        verb = k.split(":", 1)[0]
        s_cv = cv(durs)
        med = st.median(durs) if durs else None
        flags = noise_flags(k, cycles, clips, step_time, tkeys, cfg)
        if len(vals) < cfg.min_cycles:
            flags.append("few_cycles")
        if k not in tkeys:
            flags.append("not_in_template")
        steps.append({"step": k, "in_template": k in tkeys, "n_cycles": len(vals), "presence": round(presence, 3),
                      "median_s": _r(med, 2), "cv": _r(s_cv),
                      "order_distance": _r(sum(not m for _, m in vals) / len(vals)) if vals and k in tkeys else None,
                      "labour_h_per_cycle": _r(med / 3600 if med is not None else None, 5),
                      "cost_per_cycle": cost(med / 3600 if med is not None else None, wage),
                      "indicator": step_indicator(verb, presence, s_cv, mean_d, len(vals), cfg), "flags": flags})
    med_cycle = st.median(lengths)
    out.update(
        status="ok", consistency_score=round(score, 1),
        order_distance={"mean": round(mean_d, 3), "std": round(st.pstdev(dists), 3)},
        cycle_time={"median_s": round(med_cycle, 2), "mean_s": round(st.mean(lengths), 2), "cv": _r(cv_cycle)},
        labour={"labour_h_per_cycle_median": round(med_cycle / 3600, 5),
                "cost_per_cycle_median": cost(med_cycle / 3600, wage),
                "idle_share": round(idle_share, 3), "idle_cost_per_cycle_median": cost(med_cycle * idle_share / 3600, wage),
                "unverified_share": round(sum(r["unverified_s"] for r in rows) / sum(lengths), 3)},
        indicator=task_indicator(score, steps, med_cycle, cfg), steps=steps, cycles=rows)
    return out, steps


def noise_flags(key: str, cycles: list[Cycle], clips: dict[str, ClipData], step_time: dict[str, list[Any]],
                tkeys: list[str], cfg: SOPCyclesConfig) -> list[str]:
    flags = []
    inst = [(cy, a) for cy in cycles for a in cy.actions if step_key(a) == key]
    if inst:
        conflicts = 0
        for cy, a in inst:
            others = [b for b in clips[cy.clip].actions if b.status != "rejected" and b is not a
                      and step_key(b) != key and _lanes(a) & _lanes(b)]
            conflicts += any(_iou(a, b) >= cfg.noise_iou for b in others)
        if conflicts / len(inst) >= cfg.noise_min_share:
            flags.append(f"label_noise ({conflicts}/{len(inst)} occurrences overlap a differently labelled action)")
    verb = key.split(":", 1)[0]
    variants = sorted({step_key(a) for cy in cycles if not any(step_key(a) == key for a in cy.actions)
                       for a in cy.actions if a.verb == verb and step_key(a) != key and step_key(a) not in tkeys})
    if variants:
        flags.append("name_variants (" + ", ".join(variants[:4]) + ")")
    return flags


# ---- one source / all sources --------------------------------------------------------------------------------
def analyse_source(source: str, clips: dict[str, ClipData], cfg: SOPCyclesConfig) -> dict[str, Any]:
    wage = wage_of(cfg)
    labellers = Counter(c.labeller for c in clips.values())
    majority = labellers.most_common(1)[0][0] if labellers else ""
    used = {n: c for n, c in clips.items() if not cfg.same_labeller_only or c.labeller == majority}
    excluded_clips = {n: f"labeller {c.labeller or 'unknown'} (majority: {majority})"
                      for n, c in clips.items() if n not in used}
    kept = {n: kept_actions(c, cfg) for n, c in used.items()}
    n_all = sum(len(c.actions) for c in used.values())
    n_kept = sum(1 for c in used.values() for a in c.actions if a.status in cfg.statuses)
    report: dict[str, Any] = {
        "source": source, "subject": "tasks and steps (not individual people)",
        "indicator_meaning": "automation candidate indicator: a pattern hint for engineers, not a replacement decision",
        "statuses_used": cfg.statuses,
        "actions": {"total": n_all, "used": n_kept, "excluded": n_all - n_kept,
                    "excluded_share": round((n_all - n_kept) / n_all, 3) if n_all else None},
        "clips_used": sorted(used), "clips_excluded": excluded_clips, "wage": wage, "caveats": CAVEATS,
        "formula": "score = 100 * (w_order * (1 - mean order distance) + w_timing * (1 - min(CV cycle time / cv_cap, 1)))"
                   f" / (w_order + w_timing); w_order={cfg.w_order}, w_timing={cfg.w_timing}, cv_cap={cfg.cv_cap}",
        "groups": [],
    }
    step_rows = []
    for i, g in enumerate(task_groups(source, kept, cfg), 1):
        res, steps = analyse_group(f"{source}/g{i}", g, used, kept, cfg, wage)
        report["groups"].append(res)
        step_rows += [{"group": res["group"], **s} for s in steps]
    report["complete_cycles"] = sum(g.get("complete_cycles", 0) for g in report["groups"])
    report["_step_rows"] = step_rows
    return report


def steps_csv(rows: list[dict[str, Any]], wage: dict[str, Any]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(STEP_COLUMNS)
    for s in rows:
        w.writerow([s["group"], s["step"], s["n_cycles"], round(100 * s["presence"], 1),
                    "" if s["median_s"] is None else s["median_s"], "" if s["cv"] is None else round(100 * s["cv"], 1),
                    "" if s["order_distance"] is None else s["order_distance"],
                    "" if s["labour_h_per_cycle"] is None else s["labour_h_per_cycle"],
                    "" if s["cost_per_cycle"] is None else s["cost_per_cycle"],
                    wage["currency"] or "wage not declared", s["indicator"], "; ".join(s["flags"])])
    return buf.getvalue()


def load_clips(out_root: Path) -> dict[str, dict[str, ClipData]]:
    """Same loading as sop.run_sop (sop.py:104-110), plus the labeller from run.json."""
    by_src: dict[str, dict[str, ClipData]] = defaultdict(dict)
    for res in sorted(out_root.glob("*/result.json")):
        r = json.loads(res.read_text(encoding="utf-8"))
        run = res.parent / "run.json"
        labeller = json.loads(run.read_text(encoding="utf-8")).get("stats", {}).get("steps_backend", "") \
            if run.is_file() else ""
        by_src[source_of(r["clip"])][r["clip"]] = ClipData([Action.model_validate(a) for a in r["actions"]],
                                                           float(r["video"]["duration_s"]), labeller or "")
    return by_src


def run_sop_cycles(out_root: Path, cfg: SOPCyclesConfig) -> dict[str, Any]:
    """Writes outputs/sop/<source>_consistency.json and <source>_steps.csv (new files only)."""
    out_dir = out_root / "sop"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for src, clips in sorted(load_clips(out_root).items()):
        rep = analyse_source(src, clips, cfg)
        rows = rep.pop("_step_rows")
        (out_dir / f"{src}_consistency.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        (out_dir / f"{src}_steps.csv").write_text(steps_csv(rows, rep["wage"]), encoding="utf-8")
        reports[src] = rep
    return reports
