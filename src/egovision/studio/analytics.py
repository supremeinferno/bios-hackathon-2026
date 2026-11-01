"""Worker analytics (the source of truth for every number the dashboard shows; the browser computes none).

- per worker: action count, mean / median / total / max task duration, waiting share, task breakdown
- per task (action + canonical object): count and duration statistics, per-worker split
- unusually long tasks: duration >= long_factor x median of comparable events AND >= long_min_excess_s above it.
  Comparable = same task in this video (leave-one-out), else same task in other analysed videos, else same
  action in this video, else an absolute threshold. The baseline used is reported with every flag.
- incomplete tasks: interrupted by the worker, or still running when the video (or the analysed range) ended
- potential bottlenecks: time share, repeated long instances, waiting, variability, worker-to-worker gaps
- SOP deviations (only when an SOP is given): per worker and per work cycle, skipped / out-of-order /
  repeated steps
"""

from __future__ import annotations

import time
from collections import defaultdict
from statistics import mean, median
from typing import Any

from egovision.steps.postprocess import canonical_object
from egovision.studio.models import EventRecord
from egovision.studio.settings import StudioSettings

NON_TASK = ("wait", "walk")  # not counted as productive task time


def task_key(e: EventRecord) -> str:
    return e.action if e.action in NON_TASK else f"{e.action} {canonical_object(e.object)}"


def _r(x: float | None, nd: int = 2) -> float | None:
    return None if x is None else round(float(x), nd)


def _stats(durs: list[float]) -> dict[str, Any]:
    if not durs:
        return {"count": 0, "total_s": 0.0, "avg_s": None, "median_s": None, "min_s": None, "max_s": None}
    return {"count": len(durs), "total_s": _r(sum(durs)), "avg_s": _r(mean(durs)), "median_s": _r(median(durs)),
            "min_s": _r(min(durs)), "max_s": _r(max(durs))}


def long_tasks(events: list[EventRecord], s: StudioSettings,
               other_videos: list[EventRecord] | None = None) -> list[dict[str, Any]]:
    by_task: dict[str, list[EventRecord]] = defaultdict(list)
    by_action: dict[str, list[EventRecord]] = defaultdict(list)
    for e in events:
        by_task[task_key(e)].append(e)
        by_action[e.action].append(e)
    other_task: dict[str, list[float]] = defaultdict(list)
    for e in other_videos or []:
        other_task[task_key(e)].append(e.duration_s)
    out = []
    for e in events:
        key = task_key(e)
        scope, base = None, []
        same = [o.duration_s for o in by_task[key] if o.event_id != e.event_id]
        if len(same) >= s.long_min_samples:
            scope, base = "same task in this video", same
        elif len(other_task[key]) >= s.long_min_samples:
            scope, base = "same task in other analysed videos", other_task[key]
        else:
            same_action = [o.duration_s for o in by_action[e.action] if o.event_id != e.event_id]
            if len(same_action) >= s.long_min_samples:
                scope, base = f"all '{e.action}' events in this video", same_action
        if scope is not None:
            med = median(base)
            if e.duration_s >= s.long_factor * med and e.duration_s - med >= s.long_min_excess_s:
                ratio = e.duration_s / med if med > 0 else None
                reason = (f"{e.duration_s:.1f} s is {ratio:.1f}x the median {med:.1f} s of {scope} (n={len(base)})"
                          if ratio else f"{e.duration_s:.1f} s vs a median of {med:.1f} s ({scope})")
                out.append(_long(e, key, med, len(base), scope, ratio, reason))
        elif e.duration_s >= s.long_fallback_s:
            reason = f"{e.duration_s:.1f} s, longer than the {s.long_fallback_s:.0f} s limit (no comparable events yet)"
            out.append(_long(e, key, None, 0, "absolute threshold", None, reason))
    return sorted(out, key=lambda x: -x["excess_s"])


def _long(e: EventRecord, key: str, med: float | None, n: int, scope: str, ratio: float | None,
          reason: str) -> dict[str, Any]:
    if e.completion_status == "in_progress":
        reason += " (still running)"
    return {"event_id": e.event_id, "worker_id": e.worker_id, "task": key, "action": e.action, "object": e.object,
            "start_ts": e.start_ts, "end_ts": e.end_ts, "duration_s": e.duration_s, "baseline_median_s": _r(med),
            "baseline_n": n, "baseline_scope": scope, "ratio": _r(ratio), "excess_s": _r(e.duration_s - (med or 0)),
            "completion_status": e.completion_status, "reason": reason}


def incomplete_tasks(events: list[EventRecord], analysed_until_s: float, video_end_s: float,
                     sample_every_s: float) -> list[dict[str, Any]]:
    out = []
    for e in events:
        if e.completion_status != "incomplete":
            continue
        if e.end_ts >= analysed_until_s - sample_every_s - 1e-6:  # ran into the last analysed frame
            why = ("still in progress when the video ended" if analysed_until_s >= video_end_s - sample_every_s
                   else f"still in progress when the analysed range ended ({analysed_until_s:.1f} s)")
        else:
            why = "interrupted before it was finished"
        out.append({"event_id": e.event_id, "worker_id": e.worker_id, "task": task_key(e), "action": e.action,
                    "object": e.object, "start_ts": e.start_ts, "end_ts": e.end_ts, "duration_s": e.duration_s,
                    "reason": why})
    return out


def worker_stats(worker: dict[str, Any], evs: list[EventRecord], long_ids: set[str]) -> dict[str, Any]:
    durs = [e.duration_s for e in evs]
    wait_s = sum(e.duration_s for e in evs if e.action == "wait")
    walk_s = sum(e.duration_s for e in evs if e.action == "walk")
    total = sum(durs)
    tasks: dict[str, list[EventRecord]] = defaultdict(list)
    for e in evs:
        tasks[task_key(e)].append(e)
    return {
        "worker_id": worker["worker_id"], "description": worker.get("description", ""),
        "is_camera_wearer": bool(worker.get("is_camera_wearer")),
        "n_actions": len(evs), **{k: v for k, v in _stats(durs).items() if k != "count"},
        "productive_s": _r(total - wait_s - walk_s), "wait_s": _r(wait_s), "walk_s": _r(walk_s),
        "wait_share": _r(wait_s / total, 3) if total else None,
        "first_seen_ts": min((e.start_ts for e in evs), default=None),
        "last_seen_ts": max((e.end_ts for e in evs), default=None),
        "n_long_tasks": sum(e.event_id in long_ids for e in evs),
        "n_incomplete": sum(e.completion_status == "incomplete" for e in evs),
        "n_in_progress": sum(e.completion_status == "in_progress" for e in evs),
        "n_low_confidence": sum(e.verification_status == "low_confidence" for e in evs),
        "tasks": sorted(({"task": k, "action": v[0].action, "object": canonical_object(v[0].object),
                          **_stats([e.duration_s for e in v])} for k, v in tasks.items()),
                        key=lambda t: -(t["total_s"] or 0)),
    }


def task_stats(events: list[EventRecord], long_ids: set[str]) -> list[dict[str, Any]]:
    by: dict[str, list[EventRecord]] = defaultdict(list)
    for e in events:
        by[task_key(e)].append(e)
    total = sum(e.duration_s for e in events if e.action not in NON_TASK) or 0.0
    out = []
    for k, evs in by.items():
        st = _stats([e.duration_s for e in evs])
        per_w: dict[str, list[float]] = defaultdict(list)
        for e in evs:
            per_w[e.worker_id].append(e.duration_s)
        out.append({"task": k, "action": evs[0].action, "object": canonical_object(evs[0].object), **st,
                    "share_of_task_time": _r(st["total_s"] / total, 3) if total and evs[0].action not in NON_TASK
                    else None,
                    "n_long": sum(e.event_id in long_ids for e in evs),
                    "workers": {w: _stats(d) for w, d in sorted(per_w.items())}})
    return sorted(out, key=lambda t: -(t["total_s"] or 0))


def bottlenecks(tasks: list[dict[str, Any]], workers: list[dict[str, Any]], longs: list[dict[str, Any]],
                s: StudioSettings) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in tasks:
        if t["action"] in NON_TASK:
            continue
        if (t["share_of_task_time"] or 0) >= s.bottleneck_share and t["count"] >= 1:
            out.append({"type": "time_share", "task": t["task"], "worker_id": None, "impact_s": t["total_s"],
                        "reason": f"'{t['task']}' takes {100 * t['share_of_task_time']:.0f}% of all observed task time "
                                  f"({t['count']} events, median {t['median_s']:.1f} s)"})
        task_longs = [x for x in longs if x["task"] == t["task"]]
        if len(task_longs) >= 2:
            out.append({"type": "repeated_long", "task": t["task"], "worker_id": None,
                        "impact_s": _r(sum(x["excess_s"] for x in task_longs)),
                        "reason": f"'{t['task']}' ran unusually long {len(task_longs)} times "
                                  f"({sum(x['excess_s'] for x in task_longs):.0f} s above its usual duration)"})
        if t["count"] >= 3 and t["median_s"] and t["max_s"] >= 3 * t["median_s"]:
            out.append({"type": "variability", "task": t["task"], "worker_id": None,
                        "impact_s": _r(t["max_s"] - t["median_s"]),
                        "reason": f"'{t['task']}' varies a lot: {t['min_s']:.1f}-{t['max_s']:.1f} s "
                                  f"(median {t['median_s']:.1f} s, n={t['count']})"})
        ws = {w: st for w, st in t["workers"].items() if st["count"] >= 1 and st["median_s"]}
        if len(ws) >= 2:
            fast = min(ws, key=lambda w: ws[w]["median_s"])
            slow = max(ws, key=lambda w: ws[w]["median_s"])
            f_med, s_med = ws[fast]["median_s"], ws[slow]["median_s"]
            if s_med >= 1.5 * f_med and s_med - f_med >= s.long_min_excess_s:
                out.append({"type": "worker_gap", "task": t["task"], "worker_id": slow,
                            "impact_s": _r((s_med - f_med) * ws[slow]["count"]),
                            "reason": f"{slow} takes {s_med / f_med:.1f}x longer than {fast} on '{t['task']}' "
                                      f"(median {s_med:.1f} s vs {f_med:.1f} s)"})
    for w in workers:
        if (w["wait_share"] or 0) >= s.wait_share and w["wait_s"] >= s.long_min_excess_s:
            out.append({"type": "waiting", "task": "wait", "worker_id": w["worker_id"], "impact_s": w["wait_s"],
                        "reason": f"{w['worker_id']} is waiting / idle {100 * w['wait_share']:.0f}% of their observed "
                                  f"time ({w['wait_s']:.0f} s)"})
    return sorted(out, key=lambda b: -(b["impact_s"] or 0))


def _lis(seq: list[int]) -> list[int]:
    """Longest strictly increasing subsequence (values), O(n^2) is plenty for SOP lengths."""
    if not seq:
        return []
    best = [[v] for v in seq]
    for i in range(len(seq)):
        for j in range(i):
            if seq[j] < seq[i] and len(best[j]) + 1 > len(best[i]):
                best[i] = best[j] + [seq[i]]
    return max(best, key=len)


def sop_deviations(events: list[EventRecord], sop: list[str] | None, complete: bool) -> dict[str, Any] | None:
    """Per worker, SOP-mapped events are split into cycles (repetitive work repeats the procedure): a new cycle
    starts at every return to step 1. Within each cycle: skipped steps, steps done out of order (not on the
    longest in-order run) and steps done twice (rework). The first cycle may start mid-procedure and the last
    may be cut off by the end of the video / analysed range, so steps before the first and after the last
    observed step of those two are not called skipped."""
    if not sop:
        return None
    n = len(sop)
    per_worker: dict[str, list[EventRecord]] = defaultdict(list)
    for e in sorted(events, key=lambda e: (e.start_ts, e.seq)):
        per_worker[e.worker_id].append(e)
    workers_out, deviations = [], []
    for w, evs in per_worker.items():
        runs: list[tuple[int, list[EventRecord]]] = []  # consecutive events of one step form one run
        for e in evs:
            if e.sop_step is None or not 1 <= e.sop_step <= n:
                continue
            if runs and runs[-1][0] == e.sop_step:
                runs[-1][1].append(e)
            else:
                runs.append((e.sop_step, [e]))
        if not runs:
            continue
        cycles: list[list[tuple[int, list[EventRecord]]]] = [[]]
        for r in runs:
            if r[0] == 1 and cycles[-1]:
                cycles.append([])
            cycles[-1].append(r)
        cyc_out, n_expected, n_in_order = [], 0, 0
        for ci, cyc in enumerate(cycles):
            steps = [st for st, _ in cyc]
            first = list(dict.fromkeys(steps))
            in_order = set(_lis(first))
            partial_start, partial_end = ci == 0 and steps[0] != 1, ci == len(cycles) - 1
            expected = list(range(steps[0] if partial_start else 1, (max(steps) if partial_end else n) + 1))
            missing = [st for st in expected if st not in first]
            out_of_order = [st for st in first if st not in in_order]
            repeated = sorted({st for st in steps if steps.count(st) > 1})
            n_expected += len(expected)
            n_in_order += len(in_order & set(expected))
            cevs = [e for _, es in cyc for e in es]
            t0, t1 = min(e.start_ts for e in cevs), max(e.end_ts for e in cevs)
            full = first == list(range(1, n + 1)) and not repeated
            cyc_out.append({"index": ci + 1, "start_ts": t0, "end_ts": t1, "duration_s": _r(t1 - t0), "steps": steps,
                            "complete": full, "partial_start": partial_start, "partial_end": partial_end and not full,
                            "missing": missing, "out_of_order": out_of_order, "repeated": repeated})

            def ids(step: int, cyc: list[tuple[int, list[EventRecord]]] = cyc) -> list[str]:
                return [e.event_id for st, es in cyc if st == step for e in es]

            where = f"cycle {ci + 1} ({t0:.0f}-{t1:.0f} s)"
            for st in missing:
                deviations.append({"worker_id": w, "type": "skipped", "cycle": ci + 1, "step": st,
                                   "step_text": sop[st - 1], "event_ids": [], "detail": f"{where}: step {st} was skipped"})
            for st in out_of_order:
                deviations.append({"worker_id": w, "type": "out_of_order", "cycle": ci + 1, "step": st,
                                   "step_text": sop[st - 1], "event_ids": ids(st),
                                   "detail": f"{where}: step {st} was done out of the SOP order ({steps})"})
            for st in repeated:
                deviations.append({"worker_id": w, "type": "repeated", "cycle": ci + 1, "step": st,
                                   "step_text": sop[st - 1], "event_ids": ids(st),
                                   "detail": f"{where}: step {st} was done {steps.count(st)} times (rework?)"})
        full_times = [c["duration_s"] for c in cyc_out if c["complete"]]
        workers_out.append({"worker_id": w, "observed_order": [st for st, _ in runs], "n_cycles": len(cyc_out),
                            "complete_cycles": sum(c["complete"] for c in cyc_out),
                            "median_cycle_s": _r(median(full_times)) if full_times else None,
                            "compliance_pct": _r(100 * n_in_order / n_expected, 1) if n_expected else None,
                            "cycles": cyc_out,
                            "unexpected_event_ids": [e.event_id for e in evs
                                                     if e.sop_step is None and e.action not in NON_TASK]})
    return {"steps": sop, "complete": complete, "workers": workers_out, "deviations": deviations}


def compute(video: dict[str, Any], workers: list[dict[str, Any]], events: list[EventRecord], s: StudioSettings,
            other_videos: list[EventRecord] | None = None) -> dict[str, Any]:
    complete = video["status"] == "completed"
    analysed_until = float(video.get("processed_until_s") or 0.0)
    video_end = float(video.get("analysis_limit_s") or video.get("duration_s") or 0.0)
    every = float(video.get("sample_every_s") or s.sample_every_s)
    longs = long_tasks(events, s, other_videos)
    long_ids = {x["event_id"] for x in longs}
    by_w: dict[str, list[EventRecord]] = defaultdict(list)
    for e in events:
        by_w[e.worker_id].append(e)
    known = {w["worker_id"]: w for w in workers}
    for w in by_w:
        known.setdefault(w, {"worker_id": w, "description": "not described"})
    wstats = [worker_stats(known[w], by_w.get(w, []), long_ids)
              for w in sorted(known, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)]
    tasks = task_stats(events, long_ids)
    durs = [e.duration_s for e in events]
    ver: dict[str, int] = defaultdict(int)
    for e in events:
        ver[e.verification_status] += 1
    shared = [{"task": t["task"], "workers": t["workers"],
               "fastest": min(t["workers"], key=lambda w: t["workers"][w]["median_s"] or 1e9)}
              for t in tasks if len(t["workers"]) >= 2 and t["action"] not in NON_TASK]
    incomplete = incomplete_tasks(events, analysed_until, video_end, every)
    return {
        "video_id": video["video_id"], "generated_at": time.time(), "analysis_status": video["status"],
        "is_partial": not complete, "processed_until_s": analysed_until,
        "summary": {"n_workers": len([w for w in wstats if w["n_actions"] > 0]), "n_events": len(events),
                    **{f"{k}": v for k, v in _stats(durs).items() if k != "count"},
                    "n_long_tasks": len(longs), "n_incomplete": len(incomplete),
                    "n_in_progress": sum(e.completion_status == "in_progress" for e in events),
                    "verification": dict(ver)},
        "workers": wstats,
        "tasks": tasks,
        "event_task": {e.event_id: task_key(e) for e in events},  # lets the UI place events in task rows
        "comparison": {"metrics": [{k: w[k] for k in ("worker_id", "n_actions", "avg_s", "median_s", "total_s",
                                                      "productive_s", "wait_share", "n_long_tasks", "n_incomplete")}
                                   for w in wstats if w["n_actions"] > 0],
                       "shared_tasks": shared},
        "long_tasks": longs,
        "incomplete_tasks": incomplete,
        "bottlenecks": bottlenecks(tasks, wstats, longs, s),
        "sop": sop_deviations(events, video.get("sop"), complete),
    }


def compare(entries: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    """Workers from several videos side by side (e.g. two people doing the same task in two clips)."""
    rows, by_task = [], defaultdict(dict)
    for video, an in entries:
        label = video.get("label") or video["filename"]
        for w in an["workers"]:
            if not w["n_actions"]:
                continue
            key = f"{video['video_id']}:{w['worker_id']}"
            rows.append({"key": key, "video_id": video["video_id"], "video_label": label, "worker_id": w["worker_id"],
                         "description": w["description"], **{k: w[k] for k in (
                             "n_actions", "avg_s", "median_s", "total_s", "productive_s", "wait_share", "n_long_tasks",
                             "n_incomplete")}})
            for t in w["tasks"]:
                by_task[t["task"]][key] = {k: t[k] for k in ("count", "median_s", "avg_s", "total_s")}
    shared = [{"task": k, "workers": v, "fastest": min(v, key=lambda x: v[x]["median_s"] or 1e9)}
              for k, v in by_task.items() if len(v) >= 2 and k not in NON_TASK]
    return {"workers": rows, "shared_tasks": sorted(shared, key=lambda t: -len(t["workers"]))}
