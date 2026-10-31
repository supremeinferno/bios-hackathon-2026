from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from egovision.api import create_app
from egovision.export import write_outputs
from egovision.ingest import probe
from egovision.schema import Action, FrameHands, FrameObjects

from .conftest import make_hand


def _setup(tmp_path: Path, synthetic_video: Path, sanitized: bool) -> TestClient:
    raw, out, labels = tmp_path / "raw", tmp_path / "outputs", tmp_path / "labels"
    raw.mkdir()
    labels.mkdir()
    shutil.copy(synthetic_video, raw / "clipA.mp4")
    meta = probe(raw / "clipA.mp4").model_copy(update={"name": "clipA"})
    hands = [FrameHands(frame=i, t=i / 10, hands=[make_hand("left", hid=0)] if i < 5 else []) for i in range(20)]
    acts = [Action(clip="clipA", start_s=0.0, end_s=1.0, verb="pick", object="cloth", hand="left")]
    san = None
    if sanitized:
        san = tmp_path / "san" / "clipA.mp4"
        san.parent.mkdir()
        shutil.copy(synthetic_video, san)
    d = out / "clipA"
    d.mkdir(parents=True)
    write_outputs(d, meta, hands, [FrameObjects(frame=0, t=0.0)], [], acts, [], san)
    (d / "run.json").write_text(json.dumps({"processed_until_s": 2.0, "stats": {"track_ids": 1}, "timings": {}}))
    return TestClient(create_app(raw=raw, uploads=tmp_path / "up", outputs=out, labels=labels, start_worker=False))


def test_clips_results_hands_and_downloads(tmp_path: Path, synthetic_video: Path):
    c = _setup(tmp_path, synthetic_video, sanitized=True)
    clips = c.get("/api/clips").json()
    assert [(x["name"], x["has_results"], x["has_sanitized"]) for x in clips] == [("clipA", True, True)]
    r = c.get("/api/results/clipA").json()
    assert r["actions"][0]["verb"] == "pick" and r["media"]["sanitized"] == "/media/clipA/sanitized.mp4"
    assert "actions.csv" in r["downloads"] and "pick" in r["verbs"]
    hands = c.get("/api/results/clipA/hands").json()
    assert len(hands) == 20 and len(hands[0][0]) == 8 + 42 and hands[0][0][:2] == [0, 0] and hands[10] == []
    csv_text = c.get("/api/results/clipA/download/actions.csv").text
    assert csv_text.splitlines()[0] == "clip,start_s,end_s,verb,object,hand"
    v = c.get("/media/clipA/sanitized.mp4", headers={"Range": "bytes=0-99"})
    assert v.status_code == 206 and len(v.content) == 100  # seekable


def test_originals_never_served_and_names_validated(tmp_path: Path, synthetic_video: Path):
    c = _setup(tmp_path, synthetic_video, sanitized=False)
    assert c.get("/api/results/clipA").json()["media"]["sanitized"] is None
    assert c.get("/media/clipA/sanitized.mp4").status_code == 404
    assert c.get("/api/results/..%2F..%2Fsecret").status_code in (400, 404)
    assert c.get("/api/results/clipA/download/..%2Frun.json").status_code == 404
    assert c.get("/api/results/nope").status_code == 404
    assert c.post("/api/jobs", json={"clip": "nope"}).status_code == 404
    ok = c.post("/api/jobs", json={"clip": "clipA", "max_seconds": 1})
    assert ok.status_code == 200 and c.get(f"/api/jobs/{ok.json()['id']}").json()["status"] == "queued"


def test_review_flow_and_trim_requires_approval(tmp_path: Path, synthetic_video: Path):
    c = _setup(tmp_path, synthetic_video, sanitized=True)
    # write_outputs in _setup gave the action no id; give it one like the pipeline does
    res = tmp_path / "outputs" / "clipA" / "result.json"
    r = json.loads(res.read_text())
    r["actions"][0]["id"] = "a000"
    res.write_text(json.dumps(r))
    st = c.get("/api/results/clipA/review").json()
    assert [a["id"] for a in st["queue"]] == ["a000"] and st["progress"] == {"total": 1, "reviewed": 0,
                                                                            "trim_approved": False}
    assert c.post("/api/results/clipA/review", json={"action_id": "zzz", "op": "approve"}).status_code == 404
    assert c.post("/api/results/clipA/review", json={"action_id": "a000", "op": "nonsense"}).status_code == 400
    ok = c.post("/api/results/clipA/review", json={"action_id": "a000", "op": "edit", "fields": {"verb": "place"},
                                                   "by": "khushi"})
    assert ok.status_code == 200
    st = c.get("/api/results/clipA/review").json()
    assert st["queue"][0]["verb"] == "place" and st["queue"][0]["review"]["by"] == "khushi"
    assert "place" in c.get("/api/results/clipA/download/actions_reviewed.csv").text
    c.post("/api/results/clipA/review", json={"op": "approve_trim"})
    assert c.get("/api/results/clipA/review").json()["trim_plan"]["approved"] is True
