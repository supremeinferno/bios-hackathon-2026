from __future__ import annotations

from pathlib import Path

import pytest

from egovision.config import IngestConfig
from egovision.ingest import FrameSource, VideoError, iter_all_frames, probe


def test_probe(synthetic_video: Path):
    m = probe(synthetic_video)
    assert (m.width, m.height, m.n_frames) == (160, 120, 20)
    assert m.fps == pytest.approx(10.0)
    assert m.duration_s == pytest.approx(2.0)
    assert m.name == "synthetic"


def test_probe_errors(tmp_path: Path, synthetic_video: Path):
    with pytest.raises(VideoError):
        probe(tmp_path / "missing.mp4")
    bad = tmp_path / "corrupt.mp4"
    bad.write_bytes(b"not a video at all" * 100)
    with pytest.raises(VideoError):
        probe(bad)
    with pytest.raises(VideoError, match="too short"):
        probe(synthetic_video, min_duration_s=5.0)


def test_stride_resize_and_cache(tmp_path: Path, synthetic_video: Path):
    cfg = IngestConfig(stride=3, width=80, cache_dir=tmp_path / "cache", max_seconds=None)
    src = FrameSource(probe(synthetic_video), cfg)
    first = list(src)
    assert [f.index for f in first] == list(range(0, 20, 3))
    assert first[0].image.shape == (60, 80, 3)
    assert first[1].t == pytest.approx(0.3)
    assert (src.cache_path / "manifest.json").is_file()
    second = list(FrameSource(probe(synthetic_video), cfg))  # served from cache
    assert [f.index for f in second] == [f.index for f in first]
    assert second[0].image.shape == first[0].image.shape


def test_max_seconds(tmp_path: Path, synthetic_video: Path):
    cfg = IngestConfig(stride=1, width=0, cache_frames=False, max_seconds=1.0)
    assert [f.index for f in FrameSource(probe(synthetic_video), cfg)] == list(range(10))


def test_iter_all_frames(synthetic_video: Path):
    frames = list(iter_all_frames(str(synthetic_video), width=80))
    assert len(frames) == 20 and frames[0][1].shape == (60, 80, 3)
