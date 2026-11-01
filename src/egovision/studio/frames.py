"""OpenCV video probing and timestamped frame sampling for the analyser batches.

A sampled frame at time t stands for the interval [t, t + sample_every_s): an event first seen at frame
t0 and last seen at frame t1 therefore runs from t0 to t1 + sample_every_s (capped at the video end).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)
BROWSER_CODECS = {"avc1", "h264", "x264", "vp09", "vp9", "av01", "vp80", "vp8"}


class VideoError(ValueError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    n_frames: int
    duration_s: float
    width: int
    height: int
    codec: str

    @property
    def browser_playable(self) -> bool:
        return self.codec.lower() in BROWSER_CODECS


@dataclass(frozen=True)
class SampledFrame:
    index: int  # frame number in the source video
    t: float  # seconds from video start
    jpeg: bytes
    faces: int  # face regions pixelated before upload


def probe_video(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError("OpenCV cannot open this file as a video")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ").lower() or "unknown"
        ok, _ = cap.read()
        if not ok:
            raise VideoError("the video has no decodable frames")
        if fps <= 0 or n <= 0:
            raise VideoError(f"cannot read frame rate / frame count (fps={fps}, frames={n})")
        return VideoInfo(fps=fps, n_frames=n, duration_s=round(n / fps, 3), width=w, height=h, codec=codec)
    finally:
        cap.release()


def sample_plan(info: VideoInfo, every_s: float, limit_s: float | None = None) -> list[tuple[int, float]]:
    """(frame index, timestamp) of every frame sent to the analyser, one per `every_s` seconds."""
    end = info.duration_s if limit_s is None else min(info.duration_s, limit_s)
    out: list[tuple[int, float]] = []
    k = 0
    while k * every_s < end - 1e-6:
        idx = min(info.n_frames - 1, int(round(k * every_s * info.fps)))
        if not out or idx > out[-1][0]:
            out.append((idx, round(idx / info.fps, 3)))
        k += 1
    return out


def split_batches(plan: list[tuple[int, float]], size: int) -> list[list[tuple[int, float]]]:
    return [plan[i:i + size] for i in range(0, len(plan), size)]


class FaceBlur:
    """Best-effort privacy layer: pixelate frontal faces (OpenCV's bundled Haar cascade) before upload."""

    def __init__(self) -> None:
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
        self.cascade = cv2.CascadeClassifier(str(path))
        if self.cascade.empty():
            raise RuntimeError(f"cannot load face cascade {path}")

    def __call__(self, img: np.ndarray) -> int:
        h, w = img.shape[:2]
        grey = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        min_side = max(24, int(0.05 * h))
        faces = self.cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=6, minSize=(min_side, min_side))
        for x, y, fw, fh in faces:
            px, py = int(0.2 * fw), int(0.25 * fh)
            x0, y0, x1, y1 = max(0, x - px), max(0, y - py), min(w, x + fw + px), min(h, y + fh + py)
            roi = img[y0:y1, x0:x1]
            small = cv2.resize(roi, (8, 8), interpolation=cv2.INTER_AREA)
            img[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        return len(faces)


class FrameReader:
    """Reads the planned frames in order; skipped frames are only grabbed (not converted)."""

    SEEK_GAP_S = 10.0  # jump with a seek instead of grabbing when the next frame is this far ahead

    def __init__(self, path: str | Path, info: VideoInfo, width: int, jpeg_quality: int, face_blur: bool) -> None:
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise VideoError(f"cannot open {path}")
        self.info, self.width, self.quality = info, width, jpeg_quality
        self.blur = FaceBlur() if face_blur else None
        self.pos = 0  # index of the next frame grab() returns

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> FrameReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def read(self, wanted: list[tuple[int, float]]) -> Iterator[SampledFrame]:
        for idx, t in wanted:
            if idx < self.pos or idx - self.pos > self.SEEK_GAP_S * self.info.fps:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                self.pos = idx
            while self.pos < idx:
                if not self.cap.grab():
                    raise VideoError(f"video ended at frame {self.pos}, expected frame {idx}")
                self.pos += 1
            ok, img = self.cap.read()
            self.pos += 1
            if not ok:
                raise VideoError(f"cannot decode frame {idx} (t={t:.2f}s)")
            yield self._encode(idx, t, img)

    def _encode(self, idx: int, t: float, img: np.ndarray) -> SampledFrame:
        h, w = img.shape[:2]
        if w > self.width:
            img = cv2.resize(img, (self.width, int(round(h * self.width / w))), interpolation=cv2.INTER_AREA)
        faces = self.blur(img) if self.blur is not None else 0
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            raise VideoError(f"cannot JPEG-encode frame {idx}")
        return SampledFrame(index=idx, t=t, jpeg=buf.tobytes(), faces=faces)


def frame_source(original: Path, filename: str, info: VideoInfo, sanitized_dir: Path) -> tuple[Path, str]:
    """Prefer the FORGE sanitized copy of the same clip (faces + company names pixelated) when it exists and
    covers the whole video; otherwise the uploaded original (faces are then pixelated per frame)."""
    cand = sanitized_dir / f"{Path(filename).stem}.mp4"
    if cand.is_file():
        try:
            s = probe_video(cand)
            if abs(s.duration_s - info.duration_s) <= 0.5:
                return cand, "forge_sanitized"
        except VideoError:
            log.warning("sanitized copy %s is unreadable; using the upload", cand)
    return original, "upload"
