"""Video ingest: probing, strided decoding and an on-disk frame cache.

Decoding uses PyAV with threaded decode and swscale resizing, measured at 357 fps versus 106 fps
for OpenCV on our 1080p HEVC clips (docs/DECISIONS.md D5). Every frame is decoded (HEVC cannot
skip), but only kept frames are converted. Kept, resized frames are cached as JPEG for re-runs.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np

from egovision.config import IngestConfig
from egovision.schema import VideoMeta

log = logging.getLogger(__name__)


class VideoError(RuntimeError):
    """Raised for missing, corrupt, unreadable or too-short videos."""


@dataclass(frozen=True)
class Frame:
    index: int  # frame index in the source video
    t: float  # seconds
    image: np.ndarray  # BGR at processing resolution
    scale: float  # processing width / original width


def _open(path: str | Path) -> av.container.InputContainer:
    try:
        c = av.open(str(path))
    except (av.error.FFmpegError, OSError) as e:
        raise VideoError(f"cannot open video (corrupt or unsupported): {path}: {e}") from e
    if not c.streams.video:
        c.close()
        raise VideoError(f"no video stream in {path}")
    return c


def probe(path: str | Path, min_duration_s: float = 0.0) -> VideoMeta:
    p = Path(path)
    if not p.is_file():
        raise VideoError(f"video not found: {p}")
    c = _open(p)
    try:
        s = c.streams.video[0]
        rate = s.average_rate or s.base_rate
        fps = float(rate) if rate else 0.0
        duration = float(s.duration * s.time_base) if s.duration and s.time_base else (c.duration or 0) / 1e6
        n = s.frames or int(round(duration * fps))
        w, h, codec = s.codec_context.width, s.codec_context.height, s.codec_context.name
        try:
            next(c.decode(s))
        except (StopIteration, av.error.FFmpegError) as e:
            raise VideoError(f"first frame does not decode: {p}") from e
    finally:
        c.close()
    if fps <= 0 or n <= 0 or w <= 0 or h <= 0:
        raise VideoError(f"invalid stream metadata (fps={fps}, frames={n}, size={w}x{h}): {p}")
    duration = n / fps
    if duration < min_duration_s:
        raise VideoError(f"video too short ({duration:.2f}s < {min_duration_s}s): {p}")
    return VideoMeta(path=str(p), name=p.stem, width=w, height=h, fps=fps, n_frames=n,
                     duration_s=round(duration, 3), codec=codec)


def proc_size(meta: VideoMeta, width: int) -> tuple[int, int]:
    if width <= 0 or width >= meta.width:
        return meta.width, meta.height
    return width, int(round(meta.height * width / meta.width / 2) * 2)


def decode_frames(path: str, size: tuple[int, int] | None = None, keep: set[int] | None = None,
                  last: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index, BGR image) for decoded frames; convert only indices in `keep` (all if None)."""
    c = _open(path)
    try:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        kw: dict[str, Any] = {"width": size[0], "height": size[1], "interpolation": "AREA"} if size else {}
        i = -1
        for i, f in enumerate(c.decode(s)):
            if last is not None and i > last:
                break
            if keep is None or i in keep:
                yield i, f.to_ndarray(format="bgr24", **kw)
    except av.error.FFmpegError as e:
        log.warning("ingest: decode error after frame %d of %s: %s (stopping)", i, path, e)
    finally:
        c.close()


class FrameSource:
    """Iterate every `stride`-th frame, resized to `cfg.width`, optionally through a JPEG cache."""

    def __init__(self, meta: VideoMeta, cfg: IngestConfig):
        self.meta = meta
        self.cfg = cfg
        self.size = proc_size(meta, cfg.width)
        self.scale = self.size[0] / meta.width
        last = meta.n_frames
        if cfg.max_seconds is not None:
            last = min(last, int(round(cfg.max_seconds * meta.fps)))
        self.indices = list(range(0, last, cfg.stride))

    @property
    def cache_path(self) -> Path:
        st = os.stat(self.meta.path)
        key = f"{self.meta.name}_{st.st_size}_{int(st.st_mtime)}"
        return self.cfg.cache_dir / key / f"w{self.size[0]}_s{self.cfg.stride}_n{len(self.indices)}"

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[Frame]:
        if self.cfg.cache_frames and (self.cache_path / "manifest.json").is_file():
            yield from self._from_cache()
        else:
            yield from self._decode()

    def load(self, indices: list[int]) -> Iterator[Frame]:
        """Specific frames (in order): from the cache when complete, else by decoding."""
        if not indices:
            return
        d = self.cache_path
        if self.cfg.cache_frames and (d / "manifest.json").is_file():
            for i in indices:
                img = cv2.imread(str(d / f"{i:06d}.jpg"), cv2.IMREAD_COLOR)
                if img is not None:
                    yield Frame(i, i / self.meta.fps, img, self.scale)
            return
        size = None if self.size == (self.meta.width, self.meta.height) else self.size
        for i, img in decode_frames(self.meta.path, size, set(indices), max(indices)):
            yield Frame(i, i / self.meta.fps, img, self.scale)

    def _from_cache(self) -> Iterator[Frame]:
        d = self.cache_path
        manifest = json.loads((d / "manifest.json").read_text())
        log.info("ingest: reading %d cached frames from %s", len(manifest["indices"]), d)
        for i in manifest["indices"]:
            img = cv2.imread(str(d / f"{i:06d}.jpg"), cv2.IMREAD_COLOR)
            if img is None:
                raise VideoError(f"cache is damaged (missing frame {i}); delete {d}")
            yield Frame(i, i / self.meta.fps, img, self.scale)

    def _decode(self) -> Iterator[Frame]:
        write = self.cfg.cache_frames
        d = self.cache_path
        if write:
            d.mkdir(parents=True, exist_ok=True)
        size = None if self.size == (self.meta.width, self.meta.height) else self.size
        last = self.indices[-1] if self.indices else -1
        got: list[int] = []
        params = [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality]
        for i, img in decode_frames(self.meta.path, size, set(self.indices), last):
            if write:
                cv2.imwrite(str(d / f"{i:06d}.jpg"), img, params)
            got.append(i)
            yield Frame(i, i / self.meta.fps, img, self.scale)
        if len(got) < len(self.indices):
            log.warning("ingest: got %d of %d expected frames from %s", len(got), len(self.indices), self.meta.name)
        if write and got:
            # Written last, so an interrupted run never leaves a cache that looks complete.
            (d / "manifest.json").write_text(json.dumps({"indices": got, "size": self.size}))


def iter_all_frames(path: str, width: int | None = None, last: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Every frame in order (for rendering), optionally resized to `width`."""
    size = None
    if width:
        c = _open(path)
        cc = c.streams.video[0].codec_context
        w, h = cc.width, cc.height
        c.close()
        if w != width:
            size = (width, int(round(h * width / w / 2) * 2))
    yield from decode_frames(path, size, None, last)
