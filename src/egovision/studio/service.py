"""Background work: one analysis at a time (API limits, one 4 GB GPU) and one preview transcode at a time.

Analysis paths:
- gemini: batch by batch; each batch's events are committed to SQLite before the next request, so nothing is
  lost if the page is refreshed or the server stops, and a cancelled run resumes from the next batch. If Gemini
  is not configured, rejects the key, has no quota left on any model, or `max_failed_batches` batches fail in a
  row, the rest of the video (from the first batch Gemini did not cover) is labelled offline by FORGE.
- offline: FORGE's offline labeller on the whole range (existing FORGE results are reused when they match).
Existing FORGE results can also be imported as a finished analysis without any processing.
"""

from __future__ import annotations

import json
import logging
import queue
import secrets
import shutil
import threading
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from egovision.studio import export, offline
from egovision.studio.analyzer import Analyzer, AnalyzerFatal, AnalyzerNotConfigured, BatchContext, BatchFailed
from egovision.studio.db import Database
from egovision.studio.frames import FrameReader, VideoError, frame_source, probe_video, sample_plan, split_batches
from egovision.studio.gemini import GeminiAnalyzer
from egovision.studio.live import LiveHub
from egovision.studio.merge import apply_reply, finish_open
from egovision.studio.models import AnalyzeRequest, EventRecord
from egovision.studio.settings import StudioSettings

log = logging.getLogger(__name__)
STATUS_KEYS = ("video_id", "status", "stage", "progress", "frames_total", "frames_done", "batches_total", "batches_done",
               "processed_until_s", "analysis_limit_s", "duration_s", "error", "input_tokens", "output_tokens",
               "started_at", "finished_at", "preview_status", "frames_source", "model", "analyzer", "analysis_note",
               "origin", "hands3d_status", "hands3d_progress", "hands3d_error")
# FORGE progress stages -> (start, end) share of its run time (hands dominate; measured realtime factor ~0.27)
OFFLINE_STAGES = {"hands": (0.0, 0.6), "objects": (0.6, 0.9), "privacy": (0.9, 0.92), "steps": (0.92, 0.97),
                  "render": (0.97, 0.98), "done": (0.98, 0.99)}
OfflineRunner = Callable[[StudioSettings, dict[str, Any], "float | None", "Callable[[str, float], None] | None"],
                         "tuple[list[Any], float]"]


class Cancelled(Exception):
    pass


def status_of(video: dict[str, Any]) -> dict[str, Any]:
    return {k: video.get(k) for k in STATUS_KEYS}


class Studio:
    def __init__(self, settings: StudioSettings, analyzer_factory: Callable[[StudioSettings], Analyzer] | None = None,
                 start_workers: bool = True, offline_runner: OfflineRunner | None = None) -> None:
        self.s = settings
        self.db = Database(settings.db_path)
        self.hub = LiveHub()
        self.analyzer_factory = analyzer_factory or GeminiAnalyzer
        self.offline_runner = offline_runner or offline.run_forge
        self.jobs: queue.Queue[str] = queue.Queue()
        self.previews: queue.Queue[str] = queue.Queue()
        self.hands3d_jobs: queue.Queue[str] = queue.Queue()
        self.cancel_requested: set[str] = set()
        self.current: str | None = None
        for vid in self.db.mark_interrupted():
            log.warning("analysis of %s was interrupted by a server restart; it can be resumed", vid)
        for v in self.db.list_videos():
            if v["preview_status"] == "pending":
                self.previews.put(v["video_id"])
        if start_workers:
            threading.Thread(target=self._analysis_worker, daemon=True, name="studio-analysis").start()
            threading.Thread(target=self._preview_worker, daemon=True, name="studio-preview").start()
            threading.Thread(target=self._hands3d_worker, daemon=True, name="studio-hands3d").start()

    # ---- helpers --------------------------------------------------------------------------------
    def video_dir(self, video_id: str) -> Path:
        return self.s.outputs_dir / video_id

    def publish_status(self, video_id: str) -> dict[str, Any] | None:
        v = self.db.get_video(video_id)
        if v is not None:
            self.hub.publish(video_id, {"type": "status", "status": status_of(v)})
        return v

    def analytics(self, video_id: str) -> dict[str, Any]:
        return export.build(self.db, self.s, video_id)["analytics"]

    # ---- submit / cancel / delete ---------------------------------------------------------------
    def submit(self, video_id: str, req: AnalyzeRequest) -> dict[str, Any]:
        v = self.db.get_video(video_id)
        if v is None:
            raise KeyError(video_id)
        if v["status"] in ("queued", "processing"):
            raise RuntimeError(f"analysis is already {v['status']}")
        fresh = req.restart or v["status"] in ("uploaded", "completed") or v["batches_done"] == 0
        if fresh:
            self.db.clear_analysis(video_id)
            for sub in ("reports", "raw", "forge"):
                shutil.rmtree(self.video_dir(video_id) / sub, ignore_errors=True)
            limit = min(req.max_seconds, v["duration_s"]) if req.max_seconds else None
            analyzer = req.analyzer or self.s.analyzer
            self.db.update_video(video_id, status="queued", stage="queued", progress=0.0, frames_done=0, batches_done=0,
                                 processed_until_s=0.0, error=None, started_at=None, finished_at=None,
                                 input_tokens=0, output_tokens=0, analysis_limit_s=limit, analyzer=analyzer,
                                 sample_every_s=req.sample_every_s or self.s.sample_every_s,
                                 batch_frames=req.batch_frames or self.s.batch_frames, analysis_note=None,
                                 model=self.s.gemini_model if analyzer == "gemini" else "offline FORGE labeller",
                                 sop=(req.sop or None) if req.sop is not None else v.get("sop"))
        else:  # resume with the settings the analysis started with
            self.db.update_video(video_id, status="queued", stage=f"queued (resuming after batch {v['batches_done']})",
                                 error=None)
        self.cancel_requested.discard(video_id)
        self.jobs.put(video_id)
        return self.publish_status(video_id)  # type: ignore[return-value]

    def cancel(self, video_id: str) -> dict[str, Any]:
        v = self.db.get_video(video_id)
        if v is None:
            raise KeyError(video_id)
        if v["status"] == "queued":
            self.cancel_requested.add(video_id)
            self.db.update_video(video_id, status="cancelled", stage="cancelled before it started")
        elif v["status"] == "processing":
            self.cancel_requested.add(video_id)
            self.db.update_video(video_id, stage="cancelling after the current step")
        else:
            raise RuntimeError(f"nothing to cancel (status {v['status']})")
        return self.publish_status(video_id)  # type: ignore[return-value]

    def delete(self, video_id: str) -> None:
        """Remove the video's database rows, reports, raw replies and Studio copies. FORGE's own files
        (data/raw, data/sanitized, outputs/<clip>) are never touched, also for imported videos."""
        v = self.db.get_video(video_id)
        if v is None:
            raise KeyError(video_id)
        if v["status"] in ("queued", "processing"):
            raise RuntimeError("cancel the analysis before deleting the video")
        self.db.delete_video(video_id)
        shutil.rmtree(self.video_dir(video_id), ignore_errors=True)
        shutil.rmtree(self.s.videos_dir / video_id, ignore_errors=True)

    # ---- 3D hands (egovision/hands3d.py) ----------------------------------------------------------
    def hands3d_dir(self, video_id: str) -> Path:
        return self.video_dir(video_id) / "hands3d"

    def submit_hands3d(self, video_id: str) -> dict[str, Any]:
        v = self.db.get_video(video_id)
        if v is None:
            raise KeyError(video_id)
        if v["hands3d_status"] in ("queued", "running"):
            raise RuntimeError(f"3D hands are already {v['hands3d_status']}")
        self.db.update_video(video_id, hands3d_status="queued", hands3d_progress=0.0, hands3d_error=None)
        self.hands3d_jobs.put(video_id)
        return self.publish_status(video_id)  # type: ignore[return-value]

    def run_hands3d(self, video_id: str) -> None:
        from egovision import hands3d
        from egovision.config import Config

        v = self.db.get_video(video_id)
        assert v is not None
        self.db.update_video(video_id, hands3d_status="running", hands3d_progress=0.0)
        self.publish_status(video_id)

        def progress(frac: float) -> None:
            self.db.update_video(video_id, hands3d_progress=round(frac, 3))
            self.publish_status(video_id)

        try:
            cfg = Config.load(None, [f"hands3d.hfov_deg={self.s.hands3d_hfov_deg}"])
            frames, summary = hands3d.compute(v["stored_path"], cfg, progress)
            hands3d.write(frames, summary, self.hands3d_dir(video_id))
            self.db.update_video(video_id, hands3d_status="ready", hands3d_progress=1.0)
        except Exception as e:  # noqa: BLE001 - report any failure to the page
            log.exception("3D hands for %s failed", video_id)
            self.db.update_video(video_id, hands3d_status="failed", hands3d_error=f"{type(e).__name__}: {e}"[:500])
        self.publish_status(video_id)
        self.hub.publish(video_id, {"type": "hands3d", "status": (self.db.get_video(video_id) or {}).get("hands3d_status")})

    def _hands3d_worker(self) -> None:
        while True:
            vid = self.hands3d_jobs.get()
            v = self.db.get_video(vid)
            if v is not None and v["hands3d_status"] == "queued":
                self.run_hands3d(vid)

    # ---- import existing FORGE results -----------------------------------------------------------
    def forge_clips(self) -> list[dict[str, Any]]:
        imported = {v["filename"]: v["video_id"] for v in self.db.list_videos() if v.get("origin") == "forge_import"}
        out = []
        for res in sorted(self.s.forge_outputs_dir.glob("*/result.json")) if self.s.forge_outputs_dir.is_dir() else []:
            clip = res.parent.name
            video = next((d / f"{clip}.mp4" for d in self.s.forge_video_dirs if (d / f"{clip}.mp4").is_file()), None)
            if video is None:
                continue
            run = res.parent / "run.json"
            rj = json.loads(run.read_text(encoding="utf-8")) if run.is_file() else {}
            out.append({"clip": clip, "processed_until_s": rj.get("processed_until_s"),
                        "duration_s": rj.get("video", {}).get("duration_s"),
                        "steps_backend": rj.get("stats", {}).get("steps_backend"),
                        "imported_video_id": imported.get(f"{clip}.mp4")})
        return out

    def import_forge(self, clip: str) -> dict[str, Any]:
        existing = next((v for v in self.db.list_videos() if v.get("origin") == "forge_import"
                         and v["filename"] == f"{clip}.mp4"), None)
        if existing is not None:
            return existing
        path = next((d / f"{clip}.mp4" for d in self.s.forge_video_dirs if (d / f"{clip}.mp4").is_file()), None)
        if path is None:
            raise KeyError(clip)
        info = probe_video(path)
        res = offline.forge_result(self.s, path.name, info.duration_s)
        if res is None:
            raise RuntimeError(f"no FORGE results for {clip} that cover this video")
        end = min(res["processed_until_s"], info.duration_s)
        preview = {"preview_status": "ready", "preview_path": res["sanitized"]} if res["sanitized"] else \
            {"preview_status": "not_needed" if info.browser_playable else "pending"}
        now, vid = time.time(), secrets.token_hex(6)
        note = (f"imported from FORGE {res['dir'].as_posix()} (labeller: {res['steps_backend']}); "
                f"covers 0-{end:.1f} s; no new processing")
        v = self.db.create_video(
            video_id=vid, filename=path.name, label=clip, stored_path=str(path), origin="forge_import",
            size_bytes=path.stat().st_size, duration_s=info.duration_s, fps=info.fps, width=info.width,
            height=info.height, n_frames=info.n_frames, codec=info.codec, status="completed", stage="imported",
            progress=1.0, processed_until_s=end, analysis_limit_s=end if end < info.duration_s - 0.5 else None,
            sample_every_s=self.s.sample_every_s, batch_frames=self.s.batch_frames, analyzer="offline",
            model=f"FORGE results ({res['steps_backend']})", frames_source="FORGE outputs", analysis_note=note,
            started_at=now, finished_at=now, **preview)
        self.db.upsert_worker(vid, "W1", offline.WEARER, True)
        for ev in offline.events_from_actions(res["actions"], v, self.s, 1, 0.0, end, f"import:forge ({res['steps_backend']})"):
            self.db.save_event(ev)
            self.db.upsert_worker(vid, "W1", None, None, ev.start_ts, ev.end_ts)
        if preview["preview_status"] == "pending":
            self.previews.put(vid)
        export.write_all(self.db, self.s, vid)
        log.info("imported FORGE results of %s as %s", clip, vid)
        return self.db.get_video(vid)  # type: ignore[return-value]

    # ---- workers --------------------------------------------------------------------------------
    def _analysis_worker(self) -> None:
        while True:
            vid = self.jobs.get()
            v = self.db.get_video(vid)
            if v is None or v["status"] != "queued" or vid in self.cancel_requested:
                self.cancel_requested.discard(vid)
                continue
            self.current = vid
            try:
                self.run(vid)
            except Exception:  # noqa: BLE001 - the worker thread must survive any job failure
                log.exception("analysis of %s crashed", vid)
            finally:
                self.current = None

    def _preview_worker(self) -> None:
        from egovision.media import transcode

        while True:
            vid = self.previews.get()
            v = self.db.get_video(vid)
            if v is None or v["preview_status"] != "pending":
                continue
            out = self.s.videos_dir / vid / "preview.mp4"  # never next to an imported FORGE clip
            try:
                t0 = time.perf_counter()
                transcode(v["stored_path"], out, self.s.preview_width, float(v["fps"]))
                self.db.update_video(vid, preview_status="ready", preview_path=str(out))
                log.info("preview for %s ready in %.0f s", vid, time.perf_counter() - t0)
            except Exception as e:  # noqa: BLE001 - the original stays playable in some browsers / analysable
                log.exception("preview transcode of %s failed", vid)
                self.db.update_video(vid, preview_status="failed", error=f"preview failed: {e}"[:300])
            v = self.db.get_video(vid)
            if v is not None:
                self.hub.publish(vid, {"type": "preview", "preview_status": v["preview_status"]})

    # ---- one analysis ---------------------------------------------------------------------------
    def run(self, video_id: str) -> None:
        db, s = self.db, self.s
        v = db.get_video(video_id)
        assert v is not None
        db.update_video(video_id, status="processing", stage="preparing frames", started_at=v["started_at"] or time.time())
        self.publish_status(video_id)
        open_events: dict[str, EventRecord] = {e.event_id: e for e in db.list_events(video_id)
                                               if e.completion_status == "in_progress"}
        try:
            src, kind = frame_source(Path(v["stored_path"]), v["filename"], probe_video(v["stored_path"]), s.sanitized_dir)
            info = probe_video(src)
            plan = sample_plan(info, float(v["sample_every_s"]), v["analysis_limit_s"])
            batches = split_batches(plan, int(v["batch_frames"]))
            end_s = float(v["analysis_limit_s"] or info.duration_s)
            first = db.last_done_batch(video_id) + 1
            db.update_video(video_id, frames_total=len(plan), batches_total=len(batches),
                            frames_source=kind + (" (faces pixelated per frame)" if s.face_blur and kind == "upload" else ""))
            offline_from, reason = None, ""
            if v["analyzer"] == "offline":
                offline_from, reason = first, "offline analyser selected"
            else:
                try:
                    analyzer = self.analyzer_factory(s)
                except AnalyzerNotConfigured as e:
                    offline_from, reason = first, str(e)
                else:
                    db.update_video(video_id, model=analyzer.model)
                    offline_from, reason = self._gemini_batches(video_id, src, info, batches, first, analyzer,
                                                                open_events, end_s, kind)
            if offline_from is not None and offline_from < len(batches):
                self._offline(video_id, batches, offline_from, end_s, open_events, reason)
            if open_events:  # e.g. the last batch failed while tasks were running
                for ev in finish_open(db, open_events):
                    self.hub.publish(video_id, {"type": "event", "op": "updated", "event": ev.model_dump()})
            db.update_video(video_id, status="completed", stage="completed", progress=1.0, processed_until_s=end_s,
                            finished_at=time.time())
            self._publish_after_batch(video_id, *export.write_all(db, s, video_id))
            self.hub.publish(video_id, {"type": "done", "status": status_of(db.get_video(video_id) or {})})
        except Cancelled:
            self.cancel_requested.discard(video_id)
            db.update_video(video_id, status="cancelled", stage="cancelled; press Start to resume")
            self._publish_after_batch(video_id, *export.write_all(db, s, video_id))
        except (VideoError, RuntimeError) as e:
            self._fail(video_id, f"{type(e).__name__}: {e}")
        except Exception as e:  # noqa: BLE001 - unexpected: keep what was stored, report the error
            log.exception("analysis of %s failed", video_id)
            self._fail(video_id, f"{type(e).__name__}: {e}")

    def _gemini_batches(self, video_id: str, src: Path, info: Any, batches: list[list[tuple[int, float]]], first: int,
                        analyzer: Analyzer, open_events: dict[str, EventRecord], end_s: float,
                        kind: str) -> tuple[int | None, str]:
        """Run the remaining batches. Returns (first batch to label offline, reason) or (None, "")."""
        fails, fail_start, used = 0, None, Counter()
        with FrameReader(src, info, self.s.image_width, self.s.jpeg_quality, self.s.face_blur and kind == "upload") as reader:
            for bi in range(first, len(batches)):
                if video_id in self.cancel_requested:
                    raise Cancelled()
                try:
                    model = self._run_batch(video_id, bi, batches, reader, analyzer, open_events, end_s)
                except AnalyzerFatal as e:
                    return bi, str(e)
                if model is not None:
                    used[model] += 1
                    self._note_models(video_id, analyzer, used)
                    fails, fail_start = 0, None
                    continue
                fails += 1
                fail_start = bi if fail_start is None else fail_start
                if fails >= self.s.max_failed_batches:
                    return fail_start, f"{fails} Gemini batches failed in a row"
        return None, ""

    def _note_models(self, video_id: str, analyzer: Analyzer, used: Counter[str]) -> None:
        """Which Gemini models actually answered (fallbacks happen silently otherwise)."""
        if not used:
            return
        parts = [f"{m} x{n}" for m, n in used.most_common()]
        gone = sorted(getattr(analyzer, "exhausted", set()))
        note = "Gemini batches answered by " + ", ".join(parts) + (f" ({', '.join(gone)}: quota used up)" if gone else "")
        self.db.update_video(video_id, model=used.most_common(1)[0][0], analysis_note=note)

    def _offline(self, video_id: str, batches: list[list[tuple[int, float]]], from_bi: int, end_s: float,
                 open_events: dict[str, EventRecord], reason: str) -> None:
        """Label [start of batch from_bi, end_s) with FORGE's offline labeller."""
        db, s = self.db, self.s
        v = db.get_video(video_id)
        assert v is not None
        from_s = batches[from_bi][0][1] if from_bi > 0 else 0.0
        for eid in list(open_events):  # Gemini tasks still running when it stopped: end where last seen
            ev = open_events.pop(eid).model_copy(update={"completion_status": "completed", "updated_at": time.time()})
            ev.corrections = ev.corrections + ["end uncertain: analysis switched to the offline labeller"]
            db.save_event(ev)
            self.hub.publish(video_id, {"type": "event", "op": "updated", "event": ev.model_dump()})
        note = f"offline FORGE labeller for {from_s:.1f}-{end_s:.1f} s ({reason})"
        if from_bi > 0:
            note = f"{v.get('analysis_note') or 'Gemini'} for 0-{from_s:.1f} s; " + note
        db.update_video(video_id, analysis_note=note, stage="offline labeller: preparing")
        self.publish_status(video_id)
        existing = offline.forge_result(s, v["filename"], float(v["duration_s"]))
        if existing is not None and existing["processed_until_s"] >= end_s - 0.5:
            actions, source = existing["actions"], f"offline:forge ({existing['steps_backend']}, existing results)"
            db.update_video(video_id, analysis_note=note + f"; reused FORGE results in {existing['dir'].as_posix()}")
        else:
            last = {"t": 0.0}
            base = from_bi / max(len(batches), 1)

            def progress(stage: str, frac: float) -> None:
                if video_id in self.cancel_requested:
                    raise Cancelled()  # aborts the FORGE run at its next progress report
                if time.time() - last["t"] < 1.0 and frac < 1.0:
                    return
                last["t"] = time.time()
                lo, hi = OFFLINE_STAGES.get(stage, (0.95, 0.95))  # share of the FORGE run each stage covers
                done = base + (1 - base) * (lo + (hi - lo) * frac)
                db.update_video(video_id, stage=f"offline labeller (local GPU): {stage} {100 * frac:.0f}%",
                                progress=round(min(done, 0.99), 4))
                self.publish_status(video_id)

            actions, _ = self.offline_runner(s, v, v["analysis_limit_s"], progress)
            source = "offline:forge"
        evs = offline.events_from_actions(actions, v, s, db.next_seq(video_id), from_s, end_s, source)
        if not any(w["worker_id"] == "W1" for w in db.list_workers(video_id)):
            db.upsert_worker(video_id, "W1", offline.WEARER, True)
        for ev in evs:
            db.save_event(ev)
            db.upsert_worker(video_id, "W1", None, None, ev.start_ts, ev.end_ts)
            self.hub.publish(video_id, {"type": "event", "op": "created", "event": ev.model_dump()})
        batch = {"start_ts": from_s, "end_ts": end_s, "n_frames": 0, "status": "offline", "n_new": len(evs), "notes": note}
        db.save_batch(video_id, len(batches), **batch)
        self.hub.publish(video_id, {"type": "batch", "batch": {"batch_index": len(batches), "n_updated": 0,
                                                               "n_rejected": 0, **batch}})
        db.update_video(video_id, batches_done=len(batches), frames_done=sum(len(b) for b in batches),
                        processed_until_s=end_s, progress=1.0)

    def _fail(self, video_id: str, msg: str) -> None:
        self.db.update_video(video_id, status="failed", stage="failed", error=msg[:1000], finished_at=time.time())
        self._publish_after_batch(video_id, *export.write_all(self.db, self.s, video_id))
        self.hub.publish(video_id, {"type": "error", "error": msg[:1000]})

    def _run_batch(self, video_id: str, bi: int, batches: list[list[tuple[int, float]]], reader: FrameReader,
                   analyzer: Analyzer, open_events: dict[str, EventRecord], end_s: float) -> str | None:
        """One Gemini batch. The model that answered, or None when the batch failed (logged, the run continues)."""
        db, s = self.db, self.s
        v = db.get_video(video_id)
        assert v is not None
        n = len(batches)
        wanted = batches[bi]
        db.update_video(video_id, stage=f"batch {bi + 1}/{n}: reading frames {wanted[0][1]:.1f}-{wanted[-1][1]:.1f} s")
        self.publish_status(video_id)
        ctx_wanted = [batches[bi - 1][-1]] if bi > 0 else []
        frames = list(reader.read(ctx_wanted + wanted))
        ctx_frame, frames = (frames[0], frames[1:]) if ctx_wanted else (None, frames)
        events = db.list_events(video_id)
        ctx = BatchContext(video_label=v.get("label") or v["filename"], duration_s=end_s,
                           sample_every_s=float(v["sample_every_s"]), batch_index=bi, n_batches=n, frames=frames,
                           context_frame=ctx_frame, workers=db.list_workers(video_id),
                           open_events=list(open_events.values()),
                           recent_events=sorted(events, key=lambda e: (e.end_ts, e.seq))[-8:], sop=v.get("sop"))
        db.update_video(video_id, stage=f"batch {bi + 1}/{n}: Gemini is analysing {len(frames)} frames "
                                        f"({frames[0].t:.1f}-{frames[-1].t:.1f} s)")
        self.publish_status(video_id)
        raw_dir = self.video_dir(video_id) / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"batch_{bi:04d}.json"
        request_info = {"batch_index": bi, "frames": [{"frame": f.index, "t": f.t, "faces_pixelated": f.faces}
                                                      for f in frames],
                        "context_frame_t": ctx_frame.t if ctx_frame else None, "model": analyzer.model}
        common = {"start_ts": frames[0].t, "end_ts": frames[-1].t, "n_frames": len(frames), "raw_path": str(raw_path)}
        result = None
        try:
            result = analyzer.analyze(ctx)
            raw_path.write_text(json.dumps({"request": request_info, "response": result.raw}, indent=1), encoding="utf-8")
            merged = apply_reply(db, s, v, bi, [f.t for f in frames], result.reply, open_events, end_s, bi == n - 1,
                                 source=f"gemini:{result.model}")
        except (BatchFailed, ValidationError) as e:
            raw = getattr(e, "raw", None) or (result.raw if result is not None else None)
            raw_path.write_text(json.dumps({"request": request_info, "response": raw, "error": str(e)}, indent=1),
                                encoding="utf-8")
            msg = f"invalid reply envelope: {e}" if isinstance(e, ValidationError) else str(e)
            db.save_batch(video_id, bi, status="failed", error=msg[:1000], **common)
            log.warning("batch %d of %s failed: %s", bi, video_id, msg)
            # Tasks that were running cannot be followed across a failed batch: close them where last seen.
            for eid in list(open_events):
                ev = open_events.pop(eid)
                ev = ev.model_copy(update={"completion_status": "completed", "verification_status": "unverified",
                                           "corrections": ev.corrections + [f"end uncertain: batch {bi + 1} failed"],
                                           "updated_at": time.time()})
                db.save_event(ev)
                self.hub.publish(video_id, {"type": "event", "op": "updated", "event": ev.model_dump()})
            self._progress(video_id, bi, n, batches, end_s)
            self.hub.publish(video_id, {"type": "batch", "batch": {"batch_index": bi, "status": "failed", "error": msg,
                                                                   **common}})
            self._publish_after_batch(video_id, *export.write_all(db, s, video_id))
            return None
        served = result.raw.get("model_version")
        db.save_batch(video_id, bi, status="done", n_new=len(merged.created), n_updated=len(merged.updated),
                      n_rejected=len(merged.rejected), rejected=merged.rejected, input_tokens=result.input_tokens,
                      output_tokens=result.output_tokens, latency_s=result.latency_s, **common,
                      notes=(merged.notes + (f" [model {result.model}" + (f", served as {served}]" if served else "]")
                                             if result.model != analyzer.model else "")).strip())
        db.add_tokens(video_id, result.input_tokens, result.output_tokens)
        self._progress(video_id, bi, n, batches, end_s)
        for op, evs in (("created", merged.created), ("updated", merged.updated)):
            for ev in evs:
                self.hub.publish(video_id, {"type": "event", "op": op, "event": ev.model_dump()})
        self.hub.publish(video_id, {"type": "batch", "batch": {
            "batch_index": bi, "status": "done", "n_new": len(merged.created), "n_updated": len(merged.updated),
            "n_rejected": len(merged.rejected), "rejected": merged.rejected, "latency_s": result.latency_s,
            "input_tokens": result.input_tokens, "output_tokens": result.output_tokens, "notes": merged.notes, **common}})
        if bi < n - 1:  # the final reports are written once the status is "completed"
            self._publish_after_batch(video_id, *export.write_all(db, s, video_id))
        return result.model

    def _progress(self, video_id: str, bi: int, n: int, batches: list[list[tuple[int, float]]], end_s: float) -> None:
        v = self.db.get_video(video_id)
        assert v is not None
        every = float(v["sample_every_s"])
        until = min(end_s, batches[bi][-1][1] + every)
        self.db.update_video(video_id, batches_done=bi + 1, frames_done=sum(len(b) for b in batches[:bi + 1]),
                             processed_until_s=round(until, 3), progress=round((bi + 1) / n, 4))

    def _publish_after_batch(self, video_id: str, reports: list[dict[str, Any]], data: dict[str, Any]) -> None:
        """Workers, analytics, report pointers and status, all recomputed in Python from the database."""
        self.hub.publish(video_id, {"type": "workers", "workers": data["workers"]})
        self.hub.publish(video_id, {"type": "analytics", "analytics": data["analytics"]})
        self.hub.publish(video_id, {"type": "report", "reports": [
            {k: r[k] for k in ("kind", "table_name", "status", "n_events", "processed_until_s", "created_at")}
            for r in reports]})
        self.publish_status(video_id)
