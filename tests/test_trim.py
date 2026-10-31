from __future__ import annotations

import json
from pathlib import Path

import av

from egovision.config import TrimConfig
from egovision.export import read_actions_csv
from egovision.schema import Action
from egovision.trim import load_map, plan, plan_sha, run_trim, summary, time_map, to_original, to_trimmed, write_trimmed


def A(s, e, status=None, review=None):  # noqa: N802
    return Action(clip="c", start_s=s, end_s=e, verb="wipe", object="x", hand="right", status=status, review=review)


def test_plan_pads_cuts_long_idle_and_keeps_model_rejected():
    cfg = TrimConfig(pad_s=1.0, min_idle_s=2.0)
    acts = [A(5, 6), A(8, 9), A(20, 21, status="rejected"), A(30, 31, review={"status": "rejected"})]
    segs = plan(acts, 40, cfg)
    # Stage 7 'rejected' (20-21 s) is kept: often real work with the hands out of view; reviewer-rejected is cut
    assert segs == [{"start_s": 0.0, "end_s": 4.0, "mode": "idle"}, {"start_s": 4.0, "end_s": 10.0, "mode": "keep"},
                    {"start_s": 10.0, "end_s": 19.0, "mode": "idle"}, {"start_s": 19.0, "end_s": 22.0, "mode": "keep"},
                    {"start_s": 22.0, "end_s": 40.0, "mode": "idle"}]
    dropped = plan(acts, 40, cfg.model_copy(update={"drop_model_rejected": True}))
    assert [s["mode"] for s in dropped] == ["idle", "keep", "idle"]


def test_short_idle_is_not_cut_and_summary():
    cfg = TrimConfig(pad_s=0.5, min_idle_s=2.0)
    segs = plan([A(1.0, 5.0), A(6.0, 9.0), A(14.0, 18.5)], 20, cfg)
    # 0-0.5 lead-in and the 5.5-5.5 gap are too short to cut; 9.5-13.5 is cut; the 19-20 tail is too short
    assert segs == [{"start_s": 0.0, "end_s": 9.5, "mode": "keep"}, {"start_s": 9.5, "end_s": 13.5, "mode": "idle"},
                    {"start_s": 13.5, "end_s": 20.0, "mode": "keep"}]
    s = summary(segs, 20)
    assert s["cuts"] == 1 and s["idle_s"] == 4.0 and s["cut_duration_s"] == 16.0 and s["kept_pct"] == 80.0
    assert plan_sha(segs) == s["sha"] and plan_sha(segs) != plan_sha(plan([A(1.0, 5.0)], 20, cfg))


def test_time_map_round_trip():
    segs = [{"start_s": 0.0, "end_s": 1.0, "mode": "keep"}, {"start_s": 1.0, "end_s": 4.0, "mode": "idle"},
            {"start_s": 4.0, "end_s": 6.0, "mode": "keep"}]
    tm = time_map(segs, 10.0)
    assert tm == [{"orig_start_s": 0.0, "orig_end_s": 1.0, "trim_start_s": 0.0, "trim_end_s": 1.0},
                  {"orig_start_s": 4.0, "orig_end_s": 6.0, "trim_start_s": 1.0, "trim_end_s": 3.0}]
    assert to_trimmed(4.5, tm) == 1.5 and to_original(1.5, tm) == 4.5
    assert to_trimmed(2.0, tm) is None  # inside a cut idle stretch


def _times(p: Path) -> list[float]:
    with av.open(str(p)) as c:
        return [round(float(f.pts * f.time_base), 2) for f in c.decode(video=0)]


SEGS = [{"start_s": 0.0, "end_s": 1.0, "mode": "keep"}, {"start_s": 1.0, "end_s": 2.0, "mode": "idle"}]


def test_keep_lite_preserves_timestamps(tmp_path: Path, synthetic_video: Path):
    cfg = TrimConfig(lite_fps=2.0, width=160, out_dir=tmp_path)
    n = write_trimmed(synthetic_video, tmp_path / "t.mp4", SEGS, 10.0, cfg, "keep_lite")
    assert n == {"keep_frames": 10, "lite_frames": 2, "lite_every_n": 5}
    times = _times(tmp_path / "t.mp4")
    assert times[:10] == [i / 10 for i in range(10)] and times[10:] == [1.0, 1.5]  # original timestamps kept


def test_keep_lite_container_duration_matches_the_clip(tmp_path: Path, synthetic_video: Path):
    # B-frames + timestamp gaps used to make the MP4 report a shorter duration than the clip (players stopped early)
    cfg = TrimConfig(lite_fps=1.0, width=160, out_dir=tmp_path)
    segs = [{"start_s": 0.0, "end_s": 0.5, "mode": "keep"}, {"start_s": 0.5, "end_s": 2.0, "mode": "idle"}]
    write_trimmed(synthetic_video, tmp_path / "d.mp4", segs, 10.0, cfg, "keep_lite", n_frames=20)
    assert _times(tmp_path / "d.mp4")[-2:] == [1.0, 1.9]  # the last source frame is kept
    with av.open(str(tmp_path / "d.mp4")) as c:
        s = c.streams.video[0]
        assert abs(float(s.duration * s.time_base) - 2.0) < 0.01


def test_cut_makes_the_video_shorter(tmp_path: Path, synthetic_video: Path):
    cfg = TrimConfig(width=160, out_dir=tmp_path)
    segs = [{"start_s": 0.0, "end_s": 0.5, "mode": "keep"}, {"start_s": 0.5, "end_s": 1.5, "mode": "idle"},
            {"start_s": 1.5, "end_s": 2.0, "mode": "keep"}]
    n = write_trimmed(synthetic_video, tmp_path / "c.mp4", segs, 10.0, cfg, "cut")
    assert n["keep_frames"] == 10 and n["lite_frames"] == 0
    assert _times(tmp_path / "c.mp4") == [i / 10 for i in range(10)]  # 1 s of video, continuous timestamps


def test_run_trim_writes_map_and_actions_in_trimmed_time(tmp_path: Path, synthetic_video: Path):
    out = tmp_path / "clipA"
    out.mkdir()
    cfg = TrimConfig(width=160, out_dir=tmp_path / "trimmed", pad_s=0.0, min_idle_s=0.5)
    acts = [A(0.0, 0.5), A(1.5, 2.0)]
    info = run_trim(out, synthetic_video, synthetic_video, acts, 2.0, 10.0, cfg, "cut")
    assert info["output_duration_s"] == 1.0 and info["cuts"] == 1 and Path(info["trimmed_path"]).is_file()
    moved = read_actions_csv(out / "actions_trimmed.csv")
    assert [(a.start_s, a.end_s) for a in moved] == [(0.0, 0.5), (0.5, 1.0)]
    run_trim(out, synthetic_video, synthetic_video, acts, 2.0, 10.0, cfg, "keep_lite")
    m = load_map(out)
    assert set(m["runs"]) == {"cut", "keep_lite"} and m["last"] == "keep_lite"
    assert m["runs"]["keep_lite"]["output_duration_s"] == 2.0


def test_load_map_reads_the_first_format(tmp_path: Path):
    (tmp_path / "trim_map.json").write_text(json.dumps({"trimmed_path": "x.mp4", "size_mb": {}}), encoding="utf-8")
    assert load_map(tmp_path)["runs"]["keep_lite"]["trimmed_path"] == "x.mp4"
