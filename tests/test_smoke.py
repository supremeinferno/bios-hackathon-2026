"""End-to-end smoke test on the first 10 s of a real clip. Must pass before every commit.

Skips (does not fail) on machines without the data or weights, e.g. CI, because the data may not
leave the team laptops. Override the clip with EGOVISION_SMOKE_VIDEO.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from egovision.config import Config
from egovision.export import load_result, read_actions_csv, read_hands_jsonl

VIDEO = Path(os.environ.get("EGOVISION_SMOKE_VIDEO", "data/raw/MTL-SKL00089-005.mp4"))
WEIGHTS = Path("weights/hand_landmarker.task")


def run_keyframes(out: Path) -> int:
    return json.loads((out / "run.json").read_text())["stats"]["keyframes"]


@pytest.mark.smoke
@pytest.mark.skipif(not VIDEO.is_file() or not WEIGHTS.is_file(), reason="needs data/raw clip and weights")
def test_smoke_10s(tmp_path: Path):
    from egovision.pipeline import run_video

    cfg = Config.load(None, [
        "ingest.max_seconds=10", f"ingest.cache_dir={(tmp_path / 'cache').as_posix()}",
        "render.width=480",
    ])
    res = run_video(VIDEO, tmp_path / "out", cfg, render=True)
    out = res.out_dir
    for f in ["hands.jsonl", "actions.csv", "contacts.csv", "result.json", "run.json", "overlay.mp4"]:
        assert (out / f).is_file() and (out / f).stat().st_size > 0, f
    frames = read_hands_jsonl(out / "hands.jsonl")
    assert len(frames) == 300 and [f.frame for f in frames] == list(range(300))
    # PS3 deliverable: exact suggested columns, valid times, verbs from the closed list, valid hands
    acts = read_actions_csv(out / "actions.csv")
    with open(out / "actions.csv", encoding="utf-8") as f:
        assert next(csv.reader(f)) == ["clip", "start_s", "end_s", "verb", "object", "hand"]
    assert all(0 <= a.start_s < a.end_s <= 10.0 and a.verb in cfg.steps.verbs and a.clip == VIDEO.stem for a in acts)
    with open(out / "contacts.csv", encoding="utf-8") as f:
        assert next(csv.reader(f)) == ["clip", "t_start", "t_end", "hand", "state", "object_name"]
    result = load_result(out)
    assert len(result["objects"]) == run_keyframes(out) and len(result["hands"]) == 300
    run = json.loads((out / "run.json").read_text())
    assert run["stats"]["processed_frames"] == 150
    # This clip shows the worker's hands at the bench for most of the time; zero hands means broken.
    assert run["stats"]["processed_with_hand_pct"] > 0
