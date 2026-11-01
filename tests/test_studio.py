"""Studio end to end without network: upload -> analyse (scripted fake analyser, fake offline labeller) ->
SQLite -> live WebSocket -> analytics -> CSV / JSON exports generated and stored by the backend; FORGE import."""

from __future__ import annotations

import csv
import io
import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from egovision.schema import Action
from egovision.studio.analytics import compute
from egovision.studio.analyzer import AnalyzerFatal, BatchContext, BatchResult
from egovision.studio.api import create_app
from egovision.studio.models import EventRecord
from egovision.studio.settings import StudioSettings


def _ev(worker: str, action: str, obj: str, s: float, e: float, status: str = "completed", conf: float = 0.9,
        cont: str | None = None, tool: str | None = None, sop: int | None = None) -> dict[str, Any]:
    return {"continues_event_id": cont, "worker_id": worker, "action": action, "object": obj, "tool": tool,
            "start_ts": s, "end_ts": e, "status": status, "confidence": conf, "sop_step": sop, "evidence": "visible"}


class FakeAnalyzer:
    """Scripted replies for a 12 s video sampled at 1 fps in batches of 4 frames (t = 0-3, 4-7, 8-11)."""

    model = "fake-model"

    def __init__(self, gate: threading.Semaphore | None = None, fatal_at: int | None = None) -> None:
        self.gate = gate
        self.fatal_at = fatal_at
        self.contexts: list[BatchContext] = []

    def analyze(self, ctx: BatchContext) -> BatchResult:
        if self.gate is not None:
            assert self.gate.acquire(timeout=20)
        self.contexts.append(ctx)
        if ctx.batch_index == self.fatal_at:
            raise AnalyzerFatal("Gemini quota is exhausted on every model (test)")
        if ctx.batch_index == 0:
            reply = {"workers": [{"worker_id": "W1", "description": "camera wearer, black gloves", "is_camera_wearer": True}],
                     "events": [_ev("W1", "pick", "Screw", 0, 1, sop=1),
                                _ev("W1", "fasten", "screw", 2, 3, "in_progress", tool="screwdriver", sop=2)],
                     "notes": ""}
        elif ctx.batch_index == 1:
            open_id = ctx.open_events[0].event_id
            reply = {"workers": [{"worker_id": "W2", "description": "person in blue overalls", "is_camera_wearer": False}],
                     "events": [_ev("W1", "fasten", "screw", 4, 5, cont=open_id, tool="screwdriver", sop=2),
                                _ev("W2", "carry", "box", 4, 7),
                                _ev("W1", "wait", "bench", 6, 6.6, conf=0.4),
                                _ev("W1", "dance", "floor", 6, 7)],  # not in the vocabulary -> rejected
                     "notes": "one unknown action"}
        else:
            reply = {"workers": [], "events": [_ev("W1", "inspect", "panel", 8, 11, "in_progress", sop=3)], "notes": ""}
        return BatchResult(reply=reply, raw={"fake": True, "batch": ctx.batch_index}, model=self.model,
                           input_tokens=100, output_tokens=20, latency_s=0.01)


@pytest.fixture
def long_video(tmp_path: Path) -> Path:
    p = tmp_path / "line_clip.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    for i in range(120):
        img = np.full((120, 160, 3), (i * 2) % 255, np.uint8)
        cv2.putText(img, str(i), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        w.write(img)
    w.release()
    return p


def _settings(tmp_path: Path) -> StudioSettings:
    return StudioSettings(videos_dir=tmp_path / "videos", outputs_dir=tmp_path / "outputs" / "videos",
                          db_path=tmp_path / "studio.sqlite3", sanitized_dir=tmp_path / "sanitized",
                          sample_every_s=1.0, batch_frames=4, face_blur=False, long_min_samples=2)


def _upload(c: TestClient, video: Path, **form: str) -> dict[str, Any]:
    with open(video, "rb") as f:
        r = c.post("/api/videos/upload", files={"file": (video.name, f, "video/mp4")}, data=form)
    assert r.status_code == 200, r.text
    return r.json()


def _wait(c: TestClient, vid: str, statuses: tuple[str, ...], timeout: float = 30) -> dict[str, Any]:
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = c.get(f"/api/videos/{vid}/status").json()
        if st["status"] in statuses:
            return st
        time.sleep(0.05)
    raise AssertionError(f"status stayed {st['status']}")


def test_upload_analyse_store_and_export(tmp_path: Path, long_video: Path):
    s = _settings(tmp_path)
    fake = FakeAnalyzer()
    with TestClient(create_app(s, analyzer_factory=lambda _: fake)) as c:
        v = _upload(c, long_video, label="Line 3", sop="pick screw\nfasten screw\ninspect panel\nlabel box")
        vid = v["video_id"]
        assert v["status"] == "uploaded" and v["duration_s"] == pytest.approx(12.0) and v["sop"][3] == "label box"
        assert Path(s.videos_dir / vid / f"{vid}.mp4").is_file()
        assert c.get(f"/api/videos/{vid}/export/csv").status_code == 409  # nothing analysed yet

        assert c.post(f"/api/videos/{vid}/analyze", json={}).json()["status"] == "queued"
        st = _wait(c, vid, ("completed", "failed"))
        assert st["status"] == "completed", st
        assert (st["batches_done"], st["frames_done"], st["processed_until_s"]) == (3, 12, 12.0)
        assert st["analysis_note"] == "Gemini batches answered by fake-model x3" and st["model"] == "fake-model"

        evs = {e["event_id"]: e for e in c.get(f"/api/videos/{vid}/events").json()}
        assert list(evs) == ["E0001", "E0002", "E0003", "E0004", "E0005"]
        pick, fasten, carry, wait, inspect = evs.values()
        assert (pick["start_ts"], pick["end_ts"], pick["duration_s"], pick["object"]) == (0, 2.0, 2.0, "screw")
        assert pick["verification_status"] == "verified"
        # fasten continued across batches 1 -> 2: one event 2.0-6.0 s seen in 4 frames
        assert (fasten["start_ts"], fasten["end_ts"], fasten["frames_seen"], fasten["completion_status"]) == \
            (2.0, 6.0, 4, "completed")
        assert (fasten["first_batch"], fasten["last_batch"], fasten["tool"]) == (0, 1, "screwdriver")
        assert wait["verification_status"] == "low_confidence" and wait["end_ts"] == 8.0  # 6.6 snapped to frame 7
        assert any("moved to the sampled frame" in x for x in wait["corrections"])
        assert carry["worker_id"] == "W2"
        assert inspect["completion_status"] == "incomplete"  # still running when the video ended
        # the open-event context reached the analyser
        assert [e.event_id for e in fake.contexts[1].open_events] == ["E0002"]
        assert fake.contexts[1].context_frame is not None and fake.contexts[1].context_frame.t == pytest.approx(3.0)

        batches = c.get(f"/api/videos/{vid}/batches").json()
        assert [b["n_rejected"] for b in batches] == [0, 1, 0]
        assert "vocabulary" in batches[1]["rejected"][0]["errors"][0]
        assert json.loads((s.outputs_dir / vid / "raw" / "batch_0001.json").read_text())["response"]["fake"]

        an = c.get(f"/api/videos/{vid}/analytics").json()
        assert an["is_partial"] is False and an["summary"]["n_workers"] == 2 and an["summary"]["n_events"] == 5
        w1 = next(w for w in an["workers"] if w["worker_id"] == "W1")
        assert w1["n_actions"] == 4 and w1["wait_s"] == pytest.approx(2.0)
        assert [x["event_id"] for x in an["incomplete_tasks"]] == ["E0005"]
        sop = c.get(f"/api/videos/{vid}/anomalies").json()["sop"]
        # one cycle (steps 1-3) cut off by the end of the video: step 4 is not called skipped
        assert sop["workers"][0]["cycles"][0]["steps"] == [1, 2, 3] and sop["workers"][0]["cycles"][0]["partial_end"]
        assert sop["workers"][0]["compliance_pct"] == 100.0 and sop["deviations"] == []

        # CSV: generated from the database, stored under outputs/, registered in the DB, downloaded as-is
        r = c.get(f"/api/videos/{vid}/export/csv")
        assert r.status_code == 200 and r.headers["x-report-status"] == "final"
        assert "Line_3" in r.headers["content-disposition"] and "PARTIAL" not in r.headers["content-disposition"]
        rows = list(csv.DictReader(io.StringIO(r.text)))
        assert [x["event_id"] for x in rows] == list(evs)
        assert {x["report_status"] for x in rows} == {"final"}
        assert rows[1]["worker_description"] == "camera wearer, black gloves" and rows[1]["duration_s"] == "4.00"
        stored = [x for x in c.get(f"/api/videos/{vid}/reports").json() if x["kind"] == "csv" and x["table_name"] == "events"]
        assert len(stored) == 1 and stored[0]["status"] == "final" and stored[0]["n_events"] == 5
        assert Path(stored[0]["path"]).read_bytes().decode() == r.text
        assert Path(stored[0]["path"]).parent == s.outputs_dir / vid / "reports"
        wr = list(csv.DictReader(io.StringIO(c.get(f"/api/videos/{vid}/export/csv?table=workers").text)))
        assert [x["worker_id"] for x in wr] == ["W1", "W2"]

        rep = c.get(f"/api/videos/{vid}/export/json").json()
        assert rep["report"]["status"] == "final" and len(rep["events"]) == 5 and len(rep["batches"]) == 3
        assert rep["video"]["input_tokens"] == 300

        # refresh: a brand-new app on the same database serves the same stored results
    with TestClient(create_app(s, analyzer_factory=lambda _: fake, start_workers=False)) as c2:
        assert len(c2.get(f"/api/videos/{vid}/events").json()) == 5
        assert c2.get(f"/api/videos/{vid}").json()["status"] == "completed"


def test_live_websocket_and_partial_export(tmp_path: Path, long_video: Path):
    s = _settings(tmp_path)
    gate = threading.Semaphore(0)
    with TestClient(create_app(s, analyzer_factory=lambda _: FakeAnalyzer(gate))) as c:
        vid = _upload(c, long_video)["video_id"]
        c.post(f"/api/videos/{vid}/analyze", json={"sop": ["pick screw", "fasten screw"]})
        with c.websocket_connect(f"/api/videos/{vid}/live") as ws:
            snap = ws.receive_json()
            assert snap["type"] == "snapshot" and snap["events"] == []
            gate.release()  # let batch 0 through
            seen: list[dict[str, Any]] = []
            while not any(m["type"] == "report" for m in seen):
                seen.append(ws.receive_json())
            created = [m["event"]["event_id"] for m in seen if m["type"] == "event" and m["op"] == "created"]
            assert created == ["E0001", "E0002"]
            assert any(m["type"] == "analytics" and m["analytics"]["summary"]["n_events"] == 2 for m in seen)

            # partial export while batch 1 waits at the gate
            r = c.get(f"/api/videos/{vid}/export/csv")
            assert r.headers["x-report-status"] == "partial" and "PARTIAL" in r.headers["content-disposition"]
            rows = list(csv.DictReader(io.StringIO(r.text)))
            assert len(rows) == 2 and {x["report_status"] for x in rows} == {"partial"}
            assert c.get(f"/api/videos/{vid}/export/json").json()["report"]["note"].startswith("INCOMPLETE")

            gate.release()
            gate.release()
            while True:
                m = ws.receive_json()
                seen.append(m)
                if m["type"] in ("done", "error"):
                    break
            assert m["type"] == "done"
            updated = [x["event"] for x in seen if x["type"] == "event" and x["op"] == "updated"]
            assert any(e["event_id"] == "E0002" and e["end_ts"] == 6.0 for e in updated)
        reps = c.get(f"/api/videos/{vid}/reports").json()
        assert {x["status"] for x in reps} == {"final"}  # partial files replaced by the final ones


def test_cancel_and_resume(tmp_path: Path, long_video: Path):
    s = _settings(tmp_path)
    gate = threading.Semaphore(0)
    fake = FakeAnalyzer(gate)
    with TestClient(create_app(s, analyzer_factory=lambda _: fake)) as c:
        vid = _upload(c, long_video)["video_id"]
        c.post(f"/api/videos/{vid}/analyze", json={})
        gate.release()
        t0 = time.time()
        while c.get(f"/api/videos/{vid}/status").json()["batches_done"] < 1 and time.time() - t0 < 10:
            time.sleep(0.05)
        assert c.post(f"/api/videos/{vid}/cancel").status_code == 200
        gate.release()  # if batch 1 already started it may finish; the worker stops before the next one
        assert _wait(c, vid, ("cancelled",))["batches_done"] in (1, 2)
        assert c.get(f"/api/videos/{vid}/export/csv").headers["x-report-status"] == "partial"
        gate.release()
        gate.release()
        c.post(f"/api/videos/{vid}/analyze", json={})  # resume: only the remaining batches are sent
        assert _wait(c, vid, ("completed",))["batches_done"] == 3
        assert [x.batch_index for x in fake.contexts] == [0, 1, 2]
        assert len(c.get(f"/api/videos/{vid}/events").json()) == 5


def fake_forge(s: StudioSettings, video: dict[str, Any], limit: float | None, progress: Any) -> tuple[list[Action], float]:
    """Stands in for the FORGE pipeline: what its offline labeller would return for the 12 s test clip."""
    progress("hands", 0.5)
    clip = Path(video["stored_path"]).stem
    return [Action(clip=clip, start_s=0.0, end_s=3.0, verb="pick", object="screw", hand="right", status="verified"),
            Action(clip=clip, start_s=5.0, end_s=9.0, verb="fasten", object="screw", hand="both", status="corrected",
                   confidence=0.7),
            Action(clip=clip, start_s=6.0, end_s=7.0, verb="wipe", object="panel", hand="left", status="rejected"),
            Action(clip=clip, start_s=10.0, end_s=12.0, verb="inspect", object="panel", hand="both", status="unverified",
                   confidence=0.62)], 12.0


def test_no_gemini_key_falls_back_to_offline(tmp_path: Path, long_video: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("egovision.studio.gemini.gemini_key", lambda: None)
    s = _settings(tmp_path)
    with TestClient(create_app(s, offline_runner=fake_forge)) as c:
        vid = _upload(c, long_video, sop="pick up screw\nfasten screw\ninspect panel")["video_id"]
        c.post(f"/api/videos/{vid}/analyze", json={})
        st = _wait(c, vid, ("completed", "failed"))
        assert st["status"] == "completed", st
        assert "GEMINI_API_KEY" in st["analysis_note"] and st["processed_until_s"] == 12.0
        evs = c.get(f"/api/videos/{vid}/events").json()
        assert [(e["action"], e["worker_id"], e["source"]) for e in evs] == [
            ("pick", "W1", "offline:forge"), ("fasten", "W1", "offline:forge"), ("inspect", "W1", "offline:forge")]
        assert [e["sop_step"] for e in evs] == [1, 2, 3]  # matched by keywords
        assert evs[1]["verification_status"] == "corrected" and evs[2]["completion_status"] == "incomplete"
        assert c.get(f"/api/videos/{vid}/workers").json()["workers"][0]["is_camera_wearer"]
        rows = list(csv.DictReader(io.StringIO(c.get(f"/api/videos/{vid}/export/csv").text)))
        assert [r["source"] for r in rows] == ["offline:forge"] * 3 and {r["report_status"] for r in rows} == {"final"}


def test_gemini_failure_mid_run_switches_to_offline(tmp_path: Path, long_video: Path):
    s = _settings(tmp_path)
    fake = FakeAnalyzer(fatal_at=1)
    with TestClient(create_app(s, analyzer_factory=lambda _: fake, offline_runner=fake_forge)) as c:
        vid = _upload(c, long_video)["video_id"]
        c.post(f"/api/videos/{vid}/analyze", json={})
        st = _wait(c, vid, ("completed", "failed"))
        assert st["status"] == "completed" and "4.0-12.0 s" in st["analysis_note"], st
        evs = c.get(f"/api/videos/{vid}/events").json()
        # Gemini covered 0-4 s (batch 0); the offline labeller covers 4-12 s, so its pick at 0-3 s is not added
        assert [(e["event_id"], e["action"], e["source"].split(":")[0]) for e in evs] == [
            ("E0001", "pick", "gemini"), ("E0002", "fasten", "gemini"), ("E0003", "fasten", "offline"),
            ("E0004", "inspect", "offline")]
        assert any("switched to the offline labeller" in x for x in evs[1]["corrections"])
        batches = c.get(f"/api/videos/{vid}/batches").json()
        assert [b["status"] for b in batches] == ["done", "offline"]


def test_import_forge_results(tmp_path: Path, long_video: Path):
    from egovision.export import write_outputs
    from egovision.ingest import probe
    from egovision.schema import FrameObjects

    raw, forge = tmp_path / "raw", tmp_path / "forge_outputs"
    raw.mkdir()
    shutil.copy(long_video, raw / "clipA.mp4")
    meta = probe(raw / "clipA.mp4").model_copy(update={"name": "clipA"})
    acts, _ = fake_forge(StudioSettings(), {"stored_path": str(raw / "clipA.mp4")}, None, lambda *a: None)
    (forge / "clipA").mkdir(parents=True)
    write_outputs(forge / "clipA", meta, [], [FrameObjects(frame=0, t=0.0)], [], acts, [], None)
    (forge / "clipA" / "run.json").write_text(json.dumps({"processed_until_s": 12.0, "video": {"duration_s": 12.0},
                                                          "stats": {"steps_backend": "exemplar"}}))
    s = _settings(tmp_path).model_copy(update={"forge_outputs_dir": forge, "forge_video_dirs": [raw]})
    with TestClient(create_app(s, start_workers=False)) as c:
        clips = c.get("/api/forge/clips").json()
        assert [(x["clip"], x["imported_video_id"]) for x in clips] == [("clipA", None)]
        v = c.post("/api/forge/import", json={"clip": "clipA"}).json()
        assert v["status"] == "completed" and v["origin"] == "forge_import" and "exemplar" in v["model"]
        assert c.post("/api/forge/import", json={"clip": "clipA"}).json()["video_id"] == v["video_id"]  # no duplicate
        evs = c.get(f"/api/videos/{v['video_id']}/events").json()
        assert [e["action"] for e in evs] == ["pick", "fasten", "inspect"]  # the rejected wipe is left out
        assert evs[0]["source"].startswith("import:forge")
        # 2D hand overlay reads FORGE's hands.jsonl of the same clip
        kp = [[100 + j, 200 + j] for j in range(21)]
        (forge / "clipA" / "hands.jsonl").write_text("".join(json.dumps(x) + "\n" for x in [
            {"frame": 0, "t": 0.0, "hands": []},
            {"frame": 1, "t": 0.04, "hands": [{"id": 3, "side": "right", "score": 0.87, "bbox": [100, 200, 120.4, 220],
                                               "kp2d": kp, "interp": True}]}]))
        d = c.get(f"/api/videos/{v['video_id']}/hands2d").json()
        assert d["source"] == "forge" and d["t"] == [0.0, 0.04] and (d["width"], d["height"]) == (v["width"], v["height"])
        assert d["frames"][0] == [] and d["frames"][1][0][:8] == [3, 1, 87, 1, 100, 200, 120, 220]
        assert d["frames"][1][0][8:12] == [100, 200, 101, 201] and len(d["frames"][1][0]) == 8 + 42
        r = c.get(f"/api/videos/{v['video_id']}/export/csv")
        assert r.headers["x-report-status"] == "final" and len(list(csv.DictReader(io.StringIO(r.text)))) == 3
        assert c.delete(f"/api/videos/{v['video_id']}").status_code == 200
        assert (raw / "clipA.mp4").is_file() and (forge / "clipA" / "result.json").is_file()  # FORGE files untouched
        assert c.post("/api/forge/import", json={"clip": "nope"}).status_code == 404


class _ApiError(Exception):
    def __init__(self, code: int, msg: str) -> None:
        super().__init__(msg)
        self.code = code


def _gemini(script: dict[str, list[Any]]) -> Any:
    """GeminiAnalyzer with a fake client: script[model] = outcomes per call (an exception or a reply dict)."""
    from types import SimpleNamespace

    from egovision.studio.gemini import GeminiAnalyzer

    def generate_content(model: str, contents: list[Any], config: Any) -> Any:
        out = script[model].pop(0)
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(text=json.dumps(out), prompt_feedback=None, model_dump=lambda **_: {"model_version": model},
                               candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
                               usage_metadata=SimpleNamespace(prompt_token_count=900, candidates_token_count=50,
                                                              thoughts_token_count=10))

    g = GeminiAnalyzer.__new__(GeminiAnalyzer)
    g.s, g.model, g.models, g.exhausted, g.config = StudioSettings(max_retries=0), "m1", ["m1", "m2"], set(), None
    g.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return g


def test_gemini_fallbacks_and_fatal_errors():
    from egovision.studio.analyzer import BatchFailed
    from egovision.studio.frames import SampledFrame

    ctx = BatchContext(video_label="v", duration_s=4, sample_every_s=1, batch_index=0, n_batches=1,
                       frames=[SampledFrame(index=0, t=0.0, jpeg=b"\xff\xd8", faces=0)])
    ok = {"workers": [], "events": [], "notes": ""}
    r = _gemini({"m1": [_ApiError(503, "overloaded")], "m2": [ok]}).analyze(ctx)  # overload -> next model
    assert (r.model, r.input_tokens, r.output_tokens) == ("m2", 900, 60)
    with pytest.raises(AnalyzerFatal, match="quota"):
        _gemini({"m1": [_ApiError(429, "Quota exceeded")], "m2": [_ApiError(429, "quota exceeded")]}).analyze(ctx)
    with pytest.raises(AnalyzerFatal, match="API key"):
        _gemini({"m1": [_ApiError(400, "API key not valid")], "m2": []}).analyze(ctx)
    with pytest.raises(BatchFailed):
        _gemini({"m1": [_ApiError(503, "busy")], "m2": [_ApiError(500, "internal")]}).analyze(ctx)


def test_sop_keyword_matching():
    from egovision.studio.offline import match_sop

    sop = ["Take the bracket from the bin", "Fasten bracket with 4 screws", "Check the weld", "Place it on the conveyor"]
    assert [match_sop(a, o, sop) for a, o in [("pick", "bracket"), ("fasten", "screw"), ("inspect", "weld"),
                                              ("place", "part"), ("wipe", "panel")]] == [1, 2, 3, 4, None]


def _rec(eid: str, worker: str, action: str, obj: str, s: float, e: float, sop: int | None = None,
         completion: str = "completed") -> EventRecord:
    return EventRecord(event_id=eid, video_id="v", seq=int(eid[1:]), worker_id=worker, action=action, object=obj,
                       start_ts=s, end_ts=e, duration_s=e - s, confidence=0.9, verification_status="verified",
                       completion_status=completion, frames_seen=2, sop_step=sop, first_batch=0, last_batch=0,  # type: ignore[arg-type]
                       created_at=0, updated_at=0)


def test_long_tasks_bottlenecks_and_sop():
    s = StudioSettings(long_min_samples=3)
    evs = [_rec("E1", "W1", "fasten", "screw", 0, 4, 2), _rec("E2", "W1", "fasten", "screws", 10, 14, 2),
           _rec("E3", "W1", "fasten", "screw", 20, 24, 2), _rec("E4", "W1", "fasten", "screw", 30, 60, 2),
           _rec("E5", "W1", "pick", "screw", 60, 62, 1), _rec("E6", "W2", "fasten", "screw", 0, 8),
           _rec("E7", "W2", "wait", "bench", 8, 36)]
    video = {"video_id": "v", "status": "completed", "processed_until_s": 62.0, "duration_s": 62.0,
             "sample_every_s": 1.0, "sop": ["pick screw", "fasten screw", "inspect"]}
    workers = [{"worker_id": "W1", "description": "a"}, {"worker_id": "W2", "description": "b"}]
    an = compute(video, workers, evs, s)
    longs = {x["event_id"]: x for x in an["long_tasks"]}
    assert set(longs) == {"E4"} and longs["E4"]["baseline_scope"] == "same task in this video"
    assert longs["E4"]["baseline_median_s"] == 4.0 and longs["E4"]["ratio"] == 7.5
    types = {(b["type"], b["worker_id"]) for b in an["bottlenecks"]}
    assert ("waiting", "W2") in types and ("time_share", None) in types
    fasten = next(t for t in an["tasks"] if t["task"] == "fasten screw")  # "screws" grouped with "screw"
    assert fasten["count"] == 5 and fasten["workers"]["W2"]["count"] == 1
    w1_sop = an["sop"]["workers"][0]  # clip starts mid-cycle at step 2, then a new cycle starts at step 1
    assert w1_sop["observed_order"] == [2, 1] and w1_sop["n_cycles"] == 2 and w1_sop["cycles"][0]["missing"] == [3]
    assert [(d["type"], d["cycle"], d["step"]) for d in an["sop"]["deviations"]] == [("skipped", 1, 3)]
    assert an["comparison"]["shared_tasks"][0]["fastest"] == "W1"


def test_sop_cycles():
    steps = [1, 2, 3, 4, 1, 3, 2, 4, 1, 2, 4, 1, 2, 2]  # cycle 2 swaps 2/3, cycle 3 skips 3, last cycle is cut off
    evs, t = [], 0.0
    for i, st in enumerate(steps):
        evs.append(_rec(f"E{i + 1}", "W1", "fasten", "part", t, t + 5, st))
        t += 5 + (1 if i in (12,) else 0)
    evs[-1] = _rec("E14", "W1", "fasten", "part", 66, 70, 2)  # separate run of step 2 after a gap -> repeated
    evs.insert(13, _rec("E99", "W1", "inspect", "part", 65, 66, 3))
    video = {"video_id": "v", "status": "completed", "processed_until_s": 70.0, "duration_s": 70.0, "sample_every_s": 1.0,
             "sop": ["a", "b", "c", "d"]}
    sop = compute(video, [{"worker_id": "W1", "description": ""}], evs, StudioSettings())["sop"]
    w = sop["workers"][0]
    assert [c["steps"] for c in w["cycles"]] == [[1, 2, 3, 4], [1, 3, 2, 4], [1, 2, 4], [1, 2, 3, 2]]
    assert [c["complete"] for c in w["cycles"]] == [True, False, False, False]
    assert w["median_cycle_s"] == 20.0
    devs = {(d["type"], d["cycle"], d["step"]) for d in sop["deviations"]}
    assert devs == {("out_of_order", 2, 2), ("skipped", 3, 3), ("repeated", 4, 2)}


def test_duration_baseline_excludes_same_clip_and_other_labellers(tmp_path: Path):
    from egovision.studio.db import Database
    from egovision.studio.export import baseline_events

    db = Database(tmp_path / "db.sqlite3")
    for vid, name in (("a" * 12, "clipX.mp4"), ("b" * 12, "clipX.mp4"), ("c" * 12, "clipY.mp4"), ("d" * 12, "clipZ.mp4")):
        db.create_video(video_id=vid, filename=name, stored_path=name, status="completed")
    srcs = {"a" * 12: "gemini:m", "b" * 12: "gemini:m", "c" * 12: "gemini:m", "d" * 12: "offline:forge"}
    for i, (vid, src) in enumerate(srcs.items()):
        db.save_event(_rec(f"E{i + 1}", "W1", "fasten", "screw", 0, 5).model_copy(update={"video_id": vid, "source": src}))
    video = db.get_video("a" * 12)
    base = baseline_events(db, video, db.list_events("a" * 12))
    assert [e.video_id for e in base] == ["c" * 12]  # not the same clip (b), not another labeller (d)


def test_hands3d_job_data_and_download(tmp_path: Path, long_video: Path, monkeypatch: pytest.MonkeyPatch):
    from egovision import hands3d

    def fake_compute(video: Any, cfg: Any, progress: Any = None, max_seconds: Any = None) -> tuple[list[Any], dict]:
        assert cfg.hands3d.hfov_deg == 140.0  # the Studio setting reaches the 3D fit
        progress(0.5)
        kp = [[0.1 + 0.01 * j, 0.4, -0.3] for j in range(21)]
        frames = [{"frame": 2 * i, "t": i / 15, "hands": [hands3d.hand_record("left", 0.9, np.array(kp), 0.4)
                                                          | {"kp2d": [[10.0 + j, 20.0] for j in range(21)],
                                                             "bbox": [10.0, 20.0, 30.0, 20.0]}] if i else []}
                  for i in range(3)]
        hands3d.smooth_and_speed(frames, cfg.hands3d)
        return frames, {"frames": 3, "rate_hz": 15.0, "coords": hands3d.COORDS}

    monkeypatch.setattr(hands3d, "compute", fake_compute)
    s = _settings(tmp_path).model_copy(update={"hands3d_hfov_deg": 140.0})
    with TestClient(create_app(s, analyzer_factory=lambda _: FakeAnalyzer())) as c:
        vid = _upload(c, long_video)["video_id"]
        assert c.get(f"/api/videos/{vid}/hands3d").status_code == 404
        assert c.get(f"/api/videos/{vid}/hands2d").status_code == 404
        assert c.post(f"/api/videos/{vid}/hands3d").json()["hands3d_status"] == "queued"
        t0 = time.time()
        while c.get(f"/api/videos/{vid}/status").json()["hands3d_status"] != "ready" and time.time() - t0 < 10:
            time.sleep(0.05)
        d = c.get(f"/api/videos/{vid}/hands3d").json()
        assert [len(f[1]) for f in d["frames"]] == [0, 1, 1]
        h = d["frames"][1][1][0]
        assert len(h) == 6 + 3 + 9 + 63 and h[0] == 0 and h[6:9] == [0.1, 0.4, -0.3]
        r = c.get(f"/api/videos/{vid}/export/hands3d")
        assert r.status_code == 200 and "hands3d.jsonl" in r.headers["content-disposition"]
        assert json.loads(r.text.splitlines()[1])["hands"][0]["side"] == "left"
        d2 = c.get(f"/api/videos/{vid}/hands2d").json()  # the 3D job's 2D keypoints feed the overlay
        assert d2["source"] == "hands3d" and [len(f) for f in d2["frames"]] == [0, 1, 1]
        assert d2["frames"][1][0][:8] == [-1, 0, 90, 0, 10, 20, 30, 20]
