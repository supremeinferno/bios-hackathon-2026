"""CSV and JSON reports, generated only from validated database records and stored in
`outputs/videos/<video_id>/reports/`. Each stored file is registered in the `reports` table (path, status,
event count, sha256). A report is `final` only once the analysis has completed; otherwise it is `partial`,
which is written into every CSV row, the file name and the JSON header.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any

from egovision.studio.analytics import compute
from egovision.studio.db import Database
from egovision.studio.settings import StudioSettings

EVENT_COLUMNS = ["report_status", "video_id", "video_file", "event_id", "worker_id", "worker_description",
                 "is_camera_wearer", "action", "object", "tool", "start_ts", "end_ts", "duration_s", "confidence",
                 "verification_status", "completion_status", "unusually_long", "long_task_reason", "incomplete_reason",
                 "sop_step", "sop_step_text", "frames_seen", "source", "evidence", "corrections"]
WORKER_COLUMNS = ["report_status", "video_id", "video_file", "worker_id", "description", "is_camera_wearer", "n_actions",
                  "avg_s", "median_s", "total_s", "max_s", "productive_s", "wait_s", "walk_s", "wait_share",
                  "first_seen_ts", "last_seen_ts", "n_long_tasks", "n_incomplete", "n_in_progress", "n_low_confidence",
                  "tasks"]


def report_status(video: dict[str, Any]) -> str:
    return "final" if video["status"] == "completed" else "partial"


def status_note(video: dict[str, Any]) -> str:
    if video["status"] == "completed":
        return "analysis complete"
    end = float(video.get("analysis_limit_s") or video.get("duration_s") or 0)
    until = f"{float(video.get('processed_until_s') or 0):.1f} s of {end:.1f} s"
    return {"processing": f"INCOMPLETE: analysis still running ({until} analysed)",
            "queued": "INCOMPLETE: analysis queued",
            "cancelled": f"INCOMPLETE: analysis cancelled ({until} analysed)",
            "failed": f"INCOMPLETE: analysis failed ({until} analysed): {video.get('error') or ''}",
            "interrupted": f"INCOMPLETE: analysis interrupted ({until} analysed)"}.get(video["status"],
                                                                                        "INCOMPLETE: not analysed yet")


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def labeller(source: str) -> str:
    """'gemini' or 'offline': events from different labellers have systematically different durations."""
    return "gemini" if "gemini" in source else "offline"


def baseline_events(db: Database, video: dict[str, Any], events: list[Any]) -> list[Any]:
    """Events of other completed videos to compare durations against: never the same clip (an import and a new
    analysis of one clip are not independent), and only from the labeller that produced most of this video."""
    if not events:
        return []
    kinds = [labeller(e.source) for e in events]
    kind = max(set(kinds), key=kinds.count)
    stem = Path(video["filename"]).stem
    return [e for v in db.list_videos() if v["video_id"] != video["video_id"] and v["status"] == "completed"
            and Path(v["filename"]).stem != stem for e in db.list_events(v["video_id"]) if labeller(e.source) == kind]


def build(db: Database, s: StudioSettings, video_id: str) -> dict[str, Any]:
    """Everything a report needs, read from the database in one go."""
    video = db.get_video(video_id)
    if video is None:
        raise KeyError(video_id)
    workers = db.list_workers(video_id)
    events = db.list_events(video_id)
    others = baseline_events(db, video, events)
    return {"video": video, "workers": workers, "events": events,
            "analytics": compute(video, workers, events, s, others)}


def events_csv(data: dict[str, Any]) -> bytes:
    video, an = data["video"], data["analytics"]
    status = report_status(video)
    workers = {w["worker_id"]: w for w in data["workers"]}
    longs = {x["event_id"]: x["reason"] for x in an["long_tasks"]}
    incomplete = {x["event_id"]: x["reason"] for x in an["incomplete_tasks"]}
    sop = video.get("sop") or []
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(EVENT_COLUMNS)
    for e in data["events"]:
        wk = workers.get(e.worker_id, {})
        w.writerow([status, video["video_id"], video["filename"], e.event_id, e.worker_id, wk.get("description", ""),
                    str(bool(wk.get("is_camera_wearer"))).lower(), e.action, e.object, e.tool or "",
                    f"{e.start_ts:.2f}", f"{e.end_ts:.2f}", f"{e.duration_s:.2f}", f"{e.confidence:.2f}",
                    e.verification_status, e.completion_status, str(e.event_id in longs).lower(),
                    longs.get(e.event_id, ""), incomplete.get(e.event_id, ""), e.sop_step or "",
                    sop[e.sop_step - 1] if e.sop_step and e.sop_step <= len(sop) else "", e.frames_seen, e.source, e.evidence,
                    "; ".join(e.corrections)])
    return buf.getvalue().encode("utf-8")


def workers_csv(data: dict[str, Any]) -> bytes:
    video, an = data["video"], data["analytics"]
    status = report_status(video)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(WORKER_COLUMNS)
    for ws in an["workers"]:
        tasks = "; ".join(f"{t['task']}: {t['count']}x, median {t['median_s']:.1f} s" for t in ws["tasks"])
        w.writerow([status, video["video_id"], video["filename"]] +
                   ["" if ws[k] is None else (str(ws[k]).lower() if isinstance(ws[k], bool) else ws[k])
                    for k in WORKER_COLUMNS[3:-1]] + [tasks])
    return buf.getvalue().encode("utf-8")


def json_report(db: Database, data: dict[str, Any]) -> bytes:
    video = data["video"]
    batches = db.list_batches(video["video_id"])
    meta_keys = ("video_id", "filename", "label", "size_bytes", "duration_s", "fps", "width", "height", "n_frames",
                 "codec", "status", "processed_until_s", "analysis_limit_s", "sample_every_s", "batch_frames",
                 "frames_source", "origin", "analyzer", "model", "analysis_note", "sop", "error", "input_tokens",
                 "output_tokens", "created_at", "started_at",
                 "finished_at")
    doc = {
        "report": {"status": report_status(video), "note": status_note(video), "analysis": video.get("analysis_note"),
                   "generated_at": time.time(),
                   "n_events": len(data["events"]), "processed_until_s": video.get("processed_until_s")},
        "video": {k: video.get(k) for k in meta_keys},
        "workers": data["analytics"]["workers"],
        "events": [e.model_dump() for e in data["events"]],
        "analytics": {k: v for k, v in data["analytics"].items() if k != "workers"},
        "batches": [{k: b[k] for k in ("batch_index", "start_ts", "end_ts", "n_frames", "status", "n_new", "n_updated",
                                       "n_rejected", "rejected", "input_tokens", "output_tokens", "latency_s", "error",
                                       "notes")} for b in batches],
    }
    return json.dumps(doc, indent=1).encode("utf-8")


def write_report(db: Database, s: StudioSettings, video_id: str, kind: str, table: str = "events",
                 data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate one report from the database, store it under outputs/, register it, return the DB row."""
    data = data or build(db, s, video_id)
    video = data["video"]
    status = report_status(video)
    if kind == "csv":
        body, name = (events_csv(data), f"{video_id}_events_{status}.csv") if table == "events" else \
            (workers_csv(data), f"{video_id}_workers_{status}.csv")
    else:
        table, body, name = "report", json_report(db, data), f"{video_id}_report_{status}.json"
    path = s.outputs_dir / video_id / "reports" / name
    _write_atomic(path, body)
    return db.save_report(video_id, kind, table, status, path, len(data["events"]), float(video.get("processed_until_s") or 0),
                          len(body), hashlib.sha256(body).hexdigest())


def write_all(db: Database, s: StudioSettings, video_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Events CSV, workers CSV and JSON report from one database snapshot; returns (report rows, snapshot)."""
    data = build(db, s, video_id)
    rows = [write_report(db, s, video_id, "csv", "events", data), write_report(db, s, video_id, "csv", "workers", data),
            write_report(db, s, video_id, "json", "report", data)]
    if report_status(data["video"]) == "final":
        for p in db.delete_partial_reports(video_id):
            Path(p).unlink(missing_ok=True)
    return rows, data
