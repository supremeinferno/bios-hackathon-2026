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
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from egovision.config import Config

log = logging.getLogger(__name__)
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")
DOWNLOADS = {"actions.csv": "text/csv", "contacts.csv": "text/csv", "hands.jsonl": "application/jsonl",
             "actions.json": "application/json", "result.json": "application/json",
             "privacy_log.json": "application/json", "run.json": "application/json"}
WEB_DIR = Path(__file__).resolve().parents[2] / "web"


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
        return {
            "clip": clip, "video": r["video"], "actions": r["actions"], "contacts": r["contacts"],
            "stats": run.get("stats", {}), "timings": run.get("timings", {}), "privacy": privacy,
            "verbs": cfg.get("steps", {}).get("verbs") or Config().steps.verbs, "evaluation": evaluation,
            "media": {"sanitized": f"/media/{clip}/sanitized.mp4" if sanitized_of(clip) else None},
            "downloads": [n for n in DOWNLOADS if (d / n).is_file()],
        }

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

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "queue": q.qsize(), "jobs": len(jobs)}

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()
