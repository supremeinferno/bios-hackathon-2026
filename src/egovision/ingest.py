"""Video ingest: probing, strided decoding and an on-disk frame cache.

Measured on our laptop (docs/DECISIONS.md D3): decode-only 163 fps, decode+convert+resize 57 fps,
random seek ~280 ms. So we `grab()` every frame (cheap), `retrieve()` only the frames we keep,
never seek in the hot path, and cache the kept, resized frames as JPEG for re-runs.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

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


def probe(path: str | Path, min_duration_s: float = 0.0) -> VideoMeta:
    p = Path(path)
    if not p.is_file():
        raise VideoError(f"video not found: {p}")
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise VideoError(f"cannot open video (corrupt or unsupported codec): {p}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cc = int(cap.get(cv2.CAP_PROP_FOURCC))
        ok, _ = cap.read()
    finally:
        cap.release()
    if not ok:
        raise VideoError(f"first frame does not decode: {p}")
    if fps <= 0 or n <= 0 or w <= 0 or h <= 0:
        raise VideoError(f"invalid stream metadata (fps={fps}, frames={n}, size={w}x{h}): {p}")
    duration = n / fps
    if duration < min_duration_s:
        raise VideoError(f"video too short ({duration:.2f}s < {min_duration_s}s): {p}")
    codec = "".join(chr((cc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")
    return VideoMeta(path=str(p), name=p.stem, width=w, height=h, fps=fps, n_frames=n,
                     duration_s=round(duration, 3), codec=codec)


def _proc_size(meta: VideoMeta, width: int) -> tuple[int, int]:
    if width <= 0 or width >= meta.width:
        return meta.width, meta.height
    return width, int(round(meta.height * width / meta.width / 2) * 2)


class FrameSource:
    """Iterate every `stride`-th frame, resized to `cfg.width`, optionally through a JPEG cache."""

    def __init__(self, meta: VideoMeta, cfg: IngestConfig):
        self.meta = meta
        self.cfg = cfg
        self.size = _proc_size(meta, cfg.width)
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
        cap = cv2.VideoCapture(self.meta.path)
        if not cap.isOpened():
            raise VideoError(f"cannot open video: {self.meta.path}")
        write = self.cfg.cache_frames
        d = self.cache_path
        if write:
            d.mkdir(parents=True, exist_ok=True)
        wanted = set(self.indices)
        last = self.indices[-1] if self.indices else -1
        got: list[int] = []
        params = [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality]
        try:
            i = 0
            while i <= last:
                if not cap.grab():
                    log.warning("ingest: stream ended early at frame %d of %d", i, self.meta.n_frames)
                    break
                if i in wanted:
                    ok, img = cap.retrieve()
                    if not ok:
                        log.warning("ingest: frame %d failed to decode; skipped", i)
                    else:
                        if self.size != (self.meta.width, self.meta.height):
                            img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
                        if write:
                            cv2.imwrite(str(d / f"{i:06d}.jpg"), img, params)
                        got.append(i)
                        yield Frame(i, i / self.meta.fps, img, self.scale)
                i += 1
        finally:
            cap.release()
        if write and got:
            # Written last, so an interrupted run never leaves a cache that looks complete.
            (d / "manifest.json").write_text(json.dumps({"indices": got, "size": self.size}))


def iter_all_frames(path: str, width: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Every frame in order (for rendering). Optionally resized to `width`."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoError(f"cannot open video: {path}")
    try:
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if width and img.shape[1] != width:
                h = int(round(img.shape[0] * width / img.shape[1] / 2) * 2)
                img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
            yield i, img
            i += 1
    finally:
        cap.release()
