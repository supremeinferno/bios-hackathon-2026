"""HTTP API + static JS dashboard. Start with `egovision ui` (http://127.0.0.1:8000).

Privacy (FORGE Stage 1): the UI only ever streams the SANITIZED copy of a clip; originals are
never served. The server binds to localhost by default because the footage is confidential.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from egovision.config import Config

log = logging.getLogger(__name__)
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")
DOWNLOADS = {"actions.csv": "text/csv", "actions_reviewed.csv": "text/csv", "contacts.csv": "text/csv",
             "signals.csv": "text/csv", "hands.jsonl": "application/jsonl", "actions.json": "application/json",
             "result.json": "application/json", "analytics.json": "application/json", "qc.json": "application/json",
             "steps_check.json": "application/json", "trim_map.json": "application/json",
             "review.json": "application/json", "privacy_log.json": "application/json",
             "run.json": "application/json", "actions_trimmed.csv": "text/csv"}
WEB_DIR = Path(__file__).resolve().parents[2] / "web"


class ReviewRequest(BaseModel):
    action_id: str | None = None
    op: str
    fields: dict[str, Any] = Field(default_factory=dict)
    at_s: float | None = None
    by: str = Field("reviewer", min_length=1, max_length=40)


class TrimRequest(BaseModel):
    mode: Literal["cut", "keep_lite"] | None = None  # None: config default (trim.mode)


class RunRequest(BaseModel):
    clip: str
    max_seconds: float | None = Field(None, gt=0)
    stride: int = Field(2, ge=1, le=6)
    objects: bool = True
    privacy: bool = True
    render: bool = False


@dataclass
class Job:
    id: str
    clip: str
    request: dict[str, Any]
    kind: str = "pipeline"  # pipeline | trim
    status: str = "queued"  # queued | running | done | failed
    stage: str = ""
    progress: float = 0.0
    error: str | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None


def create_app(raw: Path = Path("data/raw"), uploads: Path = Path("data/uploads"), outputs: Path = Path("outputs"),
               labels: Path = Path("labels"), start_worker: bool = True) -> FastAPI:
    app = FastAPI(title="egovision", version="0.2")
    jobs: dict[str, Job] = {}
    q: queue.Queue[Job] = queue.Queue()

    def video_path(clip: str) -> Path:
        if not NAME_RE.match(clip):
            raise HTTPException(400, "bad clip name")
        for d in (raw, uploads):
            p = d / f"{clip}.mp4"
            if p.is_file():
                return p
        raise HTTPException(404, f"clip {clip} not found")

    def out_dir(clip: str) -> Path:
        if not NAME_RE.match(clip):
            raise HTTPException(400, "bad clip name")
        d = outputs / clip
        if not (d / "result.json").is_file():
            raise HTTPException(404, f"no results for {clip}; run the pipeline first")
        return d

    def sanitized_of(clip: str) -> Path | None:
        d = outputs / clip
        if not (d / "result.json").is_file():
            return None
        p = json.loads((d / "result.json").read_text(encoding="utf-8")).get("files", {}).get("sanitized_video")
        return Path(p) if p and Path(p).is_file() else None

    # ---- worker: one job at a time (4 GB GPU) ------------------------------------------------
    def worker() -> None:
        from egovision.pipeline import run_video

        while True:
            job = q.get()
            job.status, job.stage = "running", "starting"
            r = job.request
            if job.kind == "trim":
                def trim_progress(frac: float, job: Job = job) -> None:
                    job.progress = round(float(frac), 3)

                try:
                    job.stage = f"trim ({r.get('mode') or Config().trim.mode})"
                    run_trim_job(job.clip, r.get("mode"), trim_progress)
                    job.status, job.progress = "done", 1.0
                except Exception as e:  # noqa: BLE001 - report any failure to the UI
                    log.exception("trim job %s failed", job.id)
                    job.status, job.error = "failed", f"{type(e).__name__}: {e}"
                job.finished = time.time()
                continue
            ov = [f"ingest.stride={r['stride']}", f"objects.enabled={str(r['objects']).lower()}",
                  f"privacy.enabled={str(r['privacy']).lower()}"]
            if r.get("max_seconds"):
                ov.append(f"ingest.max_seconds={r['max_seconds']}")

            def progress(stage: str, frac: float, job: Job = job) -> None:
                job.stage, job.progress = stage, round(float(frac), 3)

            try:
                run_video(video_path(job.clip), outputs, Config.load(None, ov), render=r["render"], progress=progress)
                job.status, job.progress = "done", 1.0
            except Exception as e:  # noqa: BLE001 - report any pipeline failure to the UI
                log.exception("job %s failed", job.id)
                job.status, job.error = "failed", f"{type(e).__name__}: {e}"
            job.finished = time.time()

    def reviewed_actions(clip: str) -> tuple[list[Any], list[Any]]:
        from egovision.review import apply_review, load_log
        from egovision.schema import Action

        d = out_dir(clip)
        r = json.loads((d / "result.json").read_text(encoding="utf-8"))
        acts = [Action.model_validate(a) for a in r["actions"]]
        log_ = load_log(d)
        return apply_review(acts, log_), log_

    def trim_state(clip: str) -> tuple[list[Any], list[Any], list[dict[str, Any]], float]:
        """Reviewed actions, review log, current trim plan and clip duration."""
        from egovision.trim import plan

        acts, log_ = reviewed_actions(clip)
        r = json.loads((out_dir(clip) / "result.json").read_text(encoding="utf-8"))
        dur = float(r["video"]["duration_s"])
        return acts, log_, plan(acts, dur, Config().trim), dur

    def trim_refusal(clip: str) -> str | None:
        """Why the trim cannot run now, or None."""
        from egovision.review import trim_approved
        from egovision.trim import plan_sha

        _, log_, segs, _ = trim_state(clip)
        if Config().trim.require_approval and not trim_approved(log_, plan_sha(segs)):
            return "the current trim plan has not been approved in review (approve it again after any edit)"
        if sanitized_of(clip) is None:
            return "no sanitized copy; trimming only ever works on the sanitized video"
        return None

    def run_trim_job(clip: str, mode: str | None = None, progress: Any = None) -> None:
        from egovision.export import load_result
        from egovision.trim import run_trim

        why = trim_refusal(clip)  # checked again: the plan may have changed while the job waited
        if why:
            raise RuntimeError(why)
        d = out_dir(clip)
        acts, _ = reviewed_actions(clip)
        res = load_result(d)
        run_trim(d, Path(res["sanitized"]), Path(res["meta"].path), acts, res["meta"].duration_s, res["meta"].fps,
                 Config().trim, mode, progress)  # type: ignore[arg-type]

    def trimmed_file(run: dict[str, Any]) -> Path | None:
        p = Path(str(run.get("trimmed_path", "")))
        ok = p.is_file() and p.resolve().is_relative_to(Config().trim.out_dir.resolve())  # only trim-stage files
        return p if ok else None

    def trimmed_runs(clip: str) -> dict[str, dict[str, Any]]:
        """Trim runs whose output file still exists, per mode."""
        from egovision.trim import load_map

        return {m: r for m, r in load_map(outputs / clip)["runs"].items() if trimmed_file(r) is not None}

    if start_worker:
        threading.Thread(target=worker, daemon=True, name="egovision-worker").start()

    # ---- clips ------------------------------------------------------------------------------
    @app.get("/api/clips")
    def clips() -> list[dict[str, Any]]:
        out = []
        for src, d in (("raw", raw), ("upload", uploads)):
            for p in sorted(d.glob("*.mp4")) if d.is_dir() else []:
                res = outputs / p.stem / "result.json"
                run = outputs / p.stem / "run.json"
                info: dict[str, Any] = {"name": p.stem, "source": src, "size_mb": round(p.stat().st_size / 1e6, 1),
                                        "has_results": res.is_file(), "has_sanitized": sanitized_of(p.stem) is not None,
                                        "has_labels": (labels / f"{p.stem}.actions.csv").is_file()}
                if run.is_file():
                    rj = json.loads(run.read_text(encoding="utf-8"))
                    info["processed_until_s"] = rj.get("processed_until_s")
                out.append(info)
        return out

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)) -> dict[str, str]:  # noqa: B008 - FastAPI idiom
        name = Path(file.filename or "").stem
        if not file.filename or not file.filename.lower().endswith(".mp4") or not NAME_RE.match(name):
            raise HTTPException(400, "upload an .mp4 with a simple file name (letters, digits, . _ -)")
        uploads.mkdir(parents=True, exist_ok=True)
        dst = uploads / f"{name}.mp4"
        with open(dst, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
        from egovision.ingest import VideoError, probe

        try:
            probe(dst, 1.0)
        except VideoError as e:
            dst.unlink(missing_ok=True)
            raise HTTPException(400, str(e)) from e
        return {"name": name}

    # ---- jobs -------------------------------------------------------------------------------
    @app.post("/api/jobs")
    def submit(req: RunRequest) -> dict[str, str]:
        video_path(req.clip)
        job = Job(id=uuid.uuid4().hex[:8], clip=req.clip, request=req.model_dump())
        jobs[job.id] = job
        q.put(job)
        return {"id": job.id}

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return [asdict(j) for j in sorted(jobs.values(), key=lambda j: -j.created)]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        if job_id not in jobs:
            raise HTTPException(404, "no such job")
        return asdict(jobs[job_id])

    # ---- results ----------------------------------------------------------------------------
    @app.get("/api/results/{clip}")
    def results(clip: str) -> dict[str, Any]:
        d = out_dir(clip)
        r = json.loads((d / "result.json").read_text(encoding="utf-8"))
        run = json.loads((d / "run.json").read_text(encoding="utf-8")) if (d / "run.json").is_file() else {}
        privacy = None
        if (d / "privacy_log.json").is_file():
            pl = json.loads((d / "privacy_log.json").read_text(encoding="utf-8"))
            privacy = {k: v for k, v in pl.items() if k not in ("keyframe_regions", "config")}
        evaluation = None
        gt = labels / f"{clip}.actions.csv"
        if gt.is_file():
            from egovision.evaluate import evaluate_dir

            sub = evaluate_dir(outputs, labels)["steps"].get(clip)
            evaluation = {"steps": sub, "labels_note": "Labels drafted by Claude Code from 1 fps frames; "
                                                       "not human-verified (labels/README.md)."}
        cfg = run.get("config", {})
        extra: dict[str, Any] = {}
        for name in ("analytics.json", "steps_check.json"):
            if (d / name).is_file():
                extra[name.removesuffix(".json")] = json.loads((d / name).read_text(encoding="utf-8"))
        from egovision.trim import load_map

        if (d / "trim_map.json").is_file():
            extra["trim_map"] = load_map(d)
        if (d / "qc.json").is_file():
            extra["qc"] = json.loads((d / "qc.json").read_text(encoding="utf-8"))["summary"]
        return {
            **extra,
            "clip": clip, "video": r["video"], "actions": r["actions"], "contacts": r["contacts"],
            "stats": run.get("stats", {}), "timings": run.get("timings", {}), "privacy": privacy,
            "verbs": cfg.get("steps", {}).get("verbs") or Config().steps.verbs, "evaluation": evaluation,
            "media": {"sanitized": f"/media/{clip}/sanitized.mp4" if sanitized_of(clip) else None,
                      "trimmed": {m: f"/media/{clip}/trimmed/{m}.mp4" for m in trimmed_runs(clip)}},
            "downloads": [n for n in DOWNLOADS if (d / n).is_file()],
        }

    # ---- Stage 9 review --------------------------------------------------------------------
    @app.get("/api/results/{clip}/review")
    def review_state(clip: str) -> dict[str, Any]:
        from egovision.review import approved_plan, progress, queue
        from egovision.trim import MODES, plan_sha, summary

        acts, log_, segs, dur = trim_state(clip)
        sha = plan_sha(segs)
        approved = approved_plan(log_)
        runs = {m: {k: r.get(k) for k in ("output_duration_s", "size_mb", "plan_sha", "cuts", "kept_pct", "created")}
                | {"url": f"/media/{clip}/trimmed/{m}.mp4", "current": r.get("plan_sha") == sha}
                for m, r in trimmed_runs(clip).items()}
        return {"queue": [a.model_dump() for a in queue(acts, log_)], "progress": progress(acts, log_, sha),
                "trim_plan": {"segments": segs, "duration_s": dur, **summary(segs, dur),
                              "approved": approved == sha, "approval_outdated": approved not in (None, sha),
                              "modes": list(MODES), "default_mode": Config().trim.mode,
                              "has_sanitized": sanitized_of(clip) is not None, "runs": runs}}

    @app.post("/api/results/{clip}/review")
    def review_decide(clip: str, req: ReviewRequest) -> dict[str, Any]:
        from egovision.review import Decision, append

        d = out_dir(clip)
        try:
            dec = Decision(**req.model_dump())
        except Exception as e:  # noqa: BLE001 - pydantic validation -> 400
            raise HTTPException(400, f"bad decision: {e}") from e
        if dec.op != "approve_trim":
            ids = {a.id for a in reviewed_actions(clip)[0]}
            if dec.action_id not in ids:
                raise HTTPException(404, f"no action {dec.action_id}")
        if dec.op == "approve_trim":  # the server records which plan was approved, never the client
            from egovision.trim import plan_sha

            dec = dec.model_copy(update={"fields": {"plan_sha": plan_sha(trim_state(clip)[2])}})
        append(d, dec)
        acts, _ = reviewed_actions(clip)
        from egovision.export import write_actions_csv
        from egovision.review import reviewed_export

        write_actions_csv(reviewed_export(acts), d / "actions_reviewed.csv")
        return {"ok": True}

    @app.post("/api/results/{clip}/trim")
    def trim(clip: str, req: TrimRequest | None = None) -> dict[str, str]:
        out_dir(clip)
        why = trim_refusal(clip)
        if why:
            raise HTTPException(409, why)
        mode = (req.mode if req else None) or Config().trim.mode
        job = Job(id=uuid.uuid4().hex[:8], clip=clip, request={"mode": mode}, kind="trim")
        jobs[job.id] = job
        q.put(job)
        return {"id": job.id}

    @app.get("/api/results/{clip}/hands")
    def hands(clip: str) -> JSONResponse:
        """Compact per-frame hands: [id, side(0=L,1=R), score%, interp, x1, y1, x2, y2, kp(42 ints)]."""
        d = out_dir(clip)
        frames = []
        with open(d / "hands.jsonl", encoding="utf-8") as f:
            for line in f:
                fr = json.loads(line)
                frames.append([[h.get("id") if h.get("id") is not None else -1, 0 if h["side"] == "left" else 1,
                                int(round(h["score"] * 100)), int(bool(h.get("interp"))),
                                *[int(round(v)) for v in h["bbox"]],
                                *[int(round(v)) for p in h["kp2d"] for v in p]] for h in fr["hands"]])
        return JSONResponse(frames)

    @app.get("/api/results/{clip}/keyframes")
    def keyframes(clip: str) -> list[dict[str, Any]]:
        r = json.loads((out_dir(clip) / "result.json").read_text(encoding="utf-8"))
        return [{"frame": k["frame"], "t": k["t"],
                 "objects": [[o["label"], round(o["score"], 2), [int(v) for v in o["bbox"]]] for o in k["objects"]],
                 "contacts": [[c["hand_id"], c["side"], c["state"], c.get("object"),
                               [int(v) for v in c["object_bbox"]] if c.get("object_bbox") else None]
                              for c in k["contacts"]]} for k in r["keyframes"]]

    @app.get("/api/results/{clip}/download/{name}")
    def download(clip: str, name: str) -> FileResponse:
        d = out_dir(clip)
        if name not in DOWNLOADS or not (d / name).is_file():
            raise HTTPException(404, "no such file")
        return FileResponse(d / name, media_type=DOWNLOADS[name], filename=f"{clip}_{name}")

    @app.get("/media/{clip}/sanitized.mp4")
    def sanitized(clip: str) -> FileResponse:
        p = sanitized_of(clip) if NAME_RE.match(clip) else None
        if p is None:
            raise HTTPException(404, "no sanitized copy; run the pipeline with privacy enabled")
        return FileResponse(p, media_type="video/mp4")  # supports HTTP Range, so the player can seek

    @app.get("/media/{clip}/trimmed/{mode}.mp4")
    def trimmed_media(clip: str, mode: str) -> FileResponse:
        run = trimmed_runs(clip).get(mode) if NAME_RE.match(clip) else None
        p = trimmed_file(run) if run else None
        if p is None:
            raise HTTPException(404, "no trimmed video for this clip and mode; run the trim in review")
        return FileResponse(p, media_type="video/mp4")  # supports HTTP Range, so the player can seek

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "queue": q.qsize(), "jobs": len(jobs)}

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()
