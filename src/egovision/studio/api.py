"""Studio HTTP + WebSocket API and the static frontend (web/studio). Start with `egovision studio`.

Binds to localhost by default: uploaded factory footage is confidential.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from egovision.studio import export
from egovision.studio.analyzer import Analyzer
from egovision.studio.frames import VideoError, probe_video
from egovision.studio.gemini import gemini_key
from egovision.studio.models import AnalyzeRequest
from egovision.studio.service import OfflineRunner, Studio, status_of
from egovision.studio.settings import ACTIONS, StudioSettings

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parents[3] / "web" / "studio"
VIDEO_ID_RE = re.compile(r"^[0-9a-f]{12}$")


class NoCacheStatic(StaticFiles):
    """The page's JS/CSS: browsers must revalidate on every load, or an update keeps running the old modules."""

    async def get_response(self, path: str, scope: Any) -> Any:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


class ImportRequest(BaseModel):
    clip: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")
VIDEO_EXT = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".avi": "video/x-msvideo",
             ".mkv": "video/x-matroska", ".webm": "video/webm"}


def create_app(settings: StudioSettings | None = None, analyzer_factory: Callable[[StudioSettings], Analyzer] | None = None,
               start_workers: bool = True, offline_runner: OfflineRunner | None = None) -> FastAPI:
    s = settings or StudioSettings.from_env()
    studio = Studio(s, analyzer_factory, start_workers, offline_runner)
    db = studio.db
    app = FastAPI(title="egovision studio", version="1.0")
    app.state.studio = studio

    def video_or_404(video_id: str) -> dict[str, Any]:
        v = db.get_video(video_id) if VIDEO_ID_RE.match(video_id) else None
        if v is None:
            raise HTTPException(404, f"no video {video_id}")
        return v

    def public(v: dict[str, Any]) -> dict[str, Any]:
        out = dict(v)
        out["media_url"] = f"/api/videos/{v['video_id']}/media"
        out["stored_file"] = Path(v["stored_path"]).as_posix()
        out.pop("stored_path", None)
        out.pop("preview_path", None)
        return out

    # ---- health + settings ----------------------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "gemini_configured": gemini_key() is not None, "model": s.gemini_model,
                "fallback_models": s.gemini_fallback_models, "default_analyzer": s.analyzer,
                "analysing": studio.current, "queued": studio.jobs.qsize()}

    @app.get("/api/settings")
    def settings_view() -> dict[str, Any]:
        keys = ("analyzer", "gemini_model", "sample_every_s", "batch_frames", "image_width", "face_blur", "min_confidence",
                "long_factor", "long_min_excess_s", "long_min_samples", "long_fallback_s", "bottleneck_share",
                "wait_share", "max_upload_mb")
        return {**{k: getattr(s, k) for k in keys}, "actions": ACTIONS}

    # ---- upload ---------------------------------------------------------------------------------
    @app.post("/api/videos/upload")
    async def upload(file: UploadFile = File(...), label: str = Form(""), sop: str = Form("")) -> dict[str, Any]:  # noqa: B008
        name = Path(file.filename or "").name
        ext = Path(name).suffix.lower()
        if ext not in VIDEO_EXT:
            raise HTTPException(400, f"unsupported file type {ext or '(none)'}; upload one of {', '.join(VIDEO_EXT)}")
        video_id = secrets.token_hex(6)
        folder = s.videos_dir / video_id
        folder.mkdir(parents=True, exist_ok=True)
        dst = folder / f"{video_id}{ext}"  # unique stem: FORGE names its outputs and caches after it
        size, limit = 0, s.max_upload_mb * 1024 * 1024
        try:
            with open(dst, "wb") as f:
                while chunk := await file.read(1 << 20):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"file is larger than {s.max_upload_mb} MB")
                    f.write(chunk)
            info = await run_in_threadpool(probe_video, dst)
        except (HTTPException, VideoError) as e:
            await run_in_threadpool(_rmtree, folder)
            if isinstance(e, VideoError):
                raise HTTPException(400, f"not a readable video: {e}") from e
            raise
        sop_steps = AnalyzeRequest(sop=sop).sop or None
        v = db.create_video(video_id=video_id, filename=name, label=" ".join(label.split())[:120] or Path(name).stem,
                            stored_path=str(dst), size_bytes=size, duration_s=info.duration_s, fps=info.fps,
                            width=info.width, height=info.height, n_frames=info.n_frames, codec=info.codec,
                            preview_status="not_needed" if info.browser_playable else "pending", sop=sop_steps,
                            sample_every_s=s.sample_every_s, batch_frames=s.batch_frames, model=s.gemini_model,
                            analyzer=s.analyzer)
        if v["preview_status"] == "pending":
            studio.previews.put(video_id)
        log.info("upload %s: %s (%.1f MB, %s, %.1f s)", video_id, name, size / 1e6, info.codec, info.duration_s)
        return public(v)

    # ---- analysis -------------------------------------------------------------------------------
    @app.post("/api/videos/{video_id}/analyze")
    def analyze(video_id: str, req: AnalyzeRequest | None = None) -> dict[str, Any]:
        video_or_404(video_id)
        try:
            return studio.submit(video_id, req or AnalyzeRequest())
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e

    @app.post("/api/videos/{video_id}/cancel")
    def cancel(video_id: str) -> dict[str, Any]:
        video_or_404(video_id)
        try:
            return studio.cancel(video_id)
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e

    @app.delete("/api/videos/{video_id}")
    def delete(video_id: str) -> dict[str, Any]:
        video_or_404(video_id)
        try:
            studio.delete(video_id)
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e
        return {"deleted": video_id}

    # ---- 3D hands ---------------------------------------------------------------------------------
    @app.post("/api/videos/{video_id}/hands3d")
    def hands3d_start(video_id: str) -> dict[str, Any]:
        video_or_404(video_id)
        try:
            return studio.submit_hands3d(video_id)
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e

    @app.get("/api/videos/{video_id}/hands3d")
    def hands3d_data(video_id: str) -> JSONResponse:
        """Compact per-frame 3D hands for the viewer: [t, [hand, ...]] with hand = [side (0 left, 1 right), score,
        dist_m, aperture_m, speed_mps (-1 = unknown), fit_err_deg, wrist xyz, wrist axes (9), keypoints (63)]."""
        video_or_404(video_id)
        d = studio.hands3d_dir(video_id)
        if not (d / "hands3d.jsonl").is_file():
            raise HTTPException(404, "3D hands not computed yet; start them first")
        frames = []
        with open(d / "hands3d.jsonl", encoding="utf-8") as f:
            for line in f:
                fr = json.loads(line)
                frames.append([fr["t"], [[0 if h["side"] == "left" else 1, h["score"], h["dist_m"], h["aperture_m"],
                                          -1 if h["speed_mps"] is None else h["speed_mps"], h["fit_err_deg"],
                                          *[round(v, 3) for v in h["wrist"]],
                                          *[round(v, 3) for row in h["axes"] for v in row],
                                          *[round(v, 3) for p in h["kp"] for v in p]] for h in fr["hands"]]])
        summary = json.loads((d / "hands3d_summary.json").read_text(encoding="utf-8"))
        return JSONResponse({"summary": summary, "frames": frames}, headers={"Cache-Control": "no-store"})

    @app.get("/api/videos/{video_id}/export/hands3d")
    def hands3d_download(video_id: str) -> FileResponse:
        v = video_or_404(video_id)
        p = studio.hands3d_dir(video_id) / "hands3d.jsonl"
        if not p.is_file():
            raise HTTPException(404, "3D hands not computed yet")
        label = re.sub(r"[^A-Za-z0-9._-]+", "_", v.get("label") or Path(v["filename"]).stem).strip("_") or "video"
        return FileResponse(p, media_type="application/jsonl", filename=f"{label}_{video_id}_hands3d.jsonl")

    # ---- import existing FORGE results ----------------------------------------------------------
    @app.get("/api/forge/clips")
    def forge_clips() -> list[dict[str, Any]]:
        return studio.forge_clips()

    @app.post("/api/forge/import")
    def forge_import(req: ImportRequest) -> dict[str, Any]:
        try:
            return public(studio.import_forge(req.clip))
        except KeyError as e:
            raise HTTPException(404, f"no FORGE clip {req.clip}") from e
        except (RuntimeError, VideoError) as e:
            raise HTTPException(409, str(e)) from e

    # ---- reads ----------------------------------------------------------------------------------
    @app.get("/api/videos")
    def videos() -> list[dict[str, Any]]:
        return [public(v) | {"n_events": len(db.list_events(v["video_id"]))} for v in db.list_videos()]

    @app.get("/api/videos/{video_id}")
    def video(video_id: str) -> dict[str, Any]:
        v = video_or_404(video_id)
        return public(v) | {"workers": db.list_workers(video_id), "reports": db.list_reports(video_id),
                            "n_events": len(db.list_events(video_id))}

    @app.get("/api/videos/{video_id}/status")
    def status(video_id: str) -> dict[str, Any]:
        return status_of(video_or_404(video_id))

    @app.get("/api/videos/{video_id}/events")
    def events(video_id: str, since: float | None = Query(None, description="only events updated after this time")  # noqa: B008
               ) -> list[dict[str, Any]]:
        video_or_404(video_id)
        evs = db.list_events(video_id)
        return [e.model_dump() for e in evs if since is None or e.updated_at > since]

    @app.get("/api/videos/{video_id}/workers")
    def workers(video_id: str) -> dict[str, Any]:
        video_or_404(video_id)
        an = studio.analytics(video_id)
        return {"workers": an["workers"], "comparison": an["comparison"], "is_partial": an["is_partial"]}

    @app.get("/api/videos/{video_id}/analytics")
    def analytics(video_id: str) -> dict[str, Any]:
        video_or_404(video_id)
        return studio.analytics(video_id)

    @app.get("/api/videos/{video_id}/anomalies")
    def anomalies(video_id: str) -> dict[str, Any]:
        video_or_404(video_id)
        an = studio.analytics(video_id)
        return {k: an[k] for k in ("is_partial", "processed_until_s", "long_tasks", "incomplete_tasks", "bottlenecks")} | \
            {"sop_deviations": (an["sop"] or {}).get("deviations", []), "sop": an["sop"]}

    @app.get("/api/videos/{video_id}/batches")
    def batches(video_id: str) -> list[dict[str, Any]]:
        video_or_404(video_id)
        return [{k: v for k, v in b.items() if k != "raw_path"} for b in db.list_batches(video_id)]

    @app.get("/api/videos/{video_id}/reports")
    def reports(video_id: str) -> list[dict[str, Any]]:
        video_or_404(video_id)
        return db.list_reports(video_id)

    @app.get("/api/compare")
    def compare(video_ids: str = Query(..., description="comma-separated video ids")) -> dict[str, Any]:
        from egovision.studio.analytics import compare as compare_workers

        ids = [x for x in dict.fromkeys(video_ids.split(",")) if x]
        if not 1 <= len(ids) <= 8:
            raise HTTPException(400, "give 1 to 8 video ids")
        return compare_workers([(video_or_404(i), studio.analytics(i)) for i in ids])

    # ---- exports (always generated by the backend from the database) ----------------------------
    def export_file(video_id: str, kind: Literal["csv", "json"], table: str) -> FileResponse:
        v = video_or_404(video_id)
        if v["status"] == "uploaded" and not db.list_events(video_id):
            raise HTTPException(409, "this video has not been analysed yet; start the analysis first")
        status_ = export.report_status(v)
        # Regenerated from the database on every download and stored again: a partial report gets the latest batch,
        # and a final one stays consistent with the dashboard (long-task baselines include other videos).
        row = export.write_report(db, s, video_id, kind, table)
        label = re.sub(r"[^A-Za-z0-9._-]+", "_", v.get("label") or Path(v["filename"]).stem).strip("_") or "video"
        suffix = "" if status_ == "final" else "_PARTIAL"
        name = f"{label}_{video_id}_{table}{suffix}.{kind}"
        assert row is not None
        return FileResponse(row["path"], media_type="text/csv" if kind == "csv" else "application/json", filename=name,
                            headers={"X-Report-Status": status_, "X-Report-Events": str(row["n_events"]),
                                     "X-Report-Note": export.status_note(v)[:200].encode("ascii", "replace").decode(),
                                     "Cache-Control": "no-store"})

    @app.get("/api/videos/{video_id}/export/csv")
    def export_csv(video_id: str, table: Literal["events", "workers"] = "events") -> FileResponse:
        return export_file(video_id, "csv", table)

    @app.get("/api/videos/{video_id}/export/json")
    def export_json(video_id: str) -> FileResponse:
        return export_file(video_id, "json", "report")

    # ---- media ----------------------------------------------------------------------------------
    @app.get("/api/videos/{video_id}/media")
    def media(video_id: str) -> FileResponse:
        v = video_or_404(video_id)
        p = Path(v["preview_path"]) if v["preview_status"] == "ready" and v.get("preview_path") else Path(v["stored_path"])
        if not p.is_file():
            raise HTTPException(404, "video file is missing from backend storage")
        return FileResponse(p, media_type=VIDEO_EXT.get(p.suffix.lower(), "video/mp4"))  # supports Range requests

    # ---- live updates -----------------------------------------------------------------------------
    def snapshot(video_id: str) -> dict[str, Any]:
        data = export.build(db, s, video_id)
        return {"status": status_of(data["video"]), "events": [e.model_dump() for e in data["events"]],
                "workers": data["workers"], "analytics": data["analytics"],
                "reports": db.list_reports(video_id),
                "batches": [{k: v for k, v in b.items() if k != "raw_path"} for b in db.list_batches(video_id)]}

    @app.websocket("/api/videos/{video_id}/live")
    async def live(ws: WebSocket, video_id: str) -> None:
        await ws.accept()
        if not VIDEO_ID_RE.match(video_id) or db.get_video(video_id) is None:
            await ws.send_json({"type": "error", "error": f"no video {video_id}"})
            await ws.close(code=4404)
            return
        q = studio.hub.subscribe(video_id)  # subscribe before the snapshot so nothing falls in between
        try:
            await ws.send_json({"type": "snapshot", **await run_in_threadpool(snapshot, video_id)})
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=20)
                except TimeoutError:
                    msg = {"type": "ping", "t": time.time()}
                if msg.get("type") == "resync":
                    msg = {"type": "snapshot", **await run_in_threadpool(snapshot, video_id)}
                await ws.send_json(msg)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            studio.hub.unsubscribe(video_id, q)

    if WEB_DIR.is_dir():
        app.mount("/", NoCacheStatic(directory=WEB_DIR, html=True), name="studio-web")
    return app


def _rmtree(p: Path) -> None:
    import shutil

    shutil.rmtree(p, ignore_errors=True)
