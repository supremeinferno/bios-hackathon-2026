"""Video writing helpers (PyAV). Picks the first H.264 encoder this machine has, else MPEG-4."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from egovision.ingest import iter_all_frames

log = logging.getLogger(__name__)

ENCODER_PREFERENCE = ["libx264", "h264_mf", "libopenh264", "h264_nvenc", "mpeg4"]


def pick_encoder() -> str:
    for name in ENCODER_PREFERENCE:
        try:
            av.codec.Codec(name, "w")
            return name
        except Exception:  # noqa: BLE001 - PyAV raises several types for "unknown codec"
            continue
    raise RuntimeError("no usable video encoder found in PyAV")


def write_video(frames: Iterable[np.ndarray], out: str | Path, fps: float, bitrate: int = 2_500_000) -> str:
    """Write BGR frames to an .mp4 atomically (temp file, renamed when complete). Returns the encoder used.

    An interrupted run must never leave a half-written video that the UI could serve.
    """
    enc = pick_encoder()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}.partial{out.suffix}")
    try:
        _encode(frames, tmp, fps, bitrate, enc)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    return enc


def _encode(frames: Iterable[np.ndarray], out: Path, fps: float, bitrate: int, enc: str) -> None:
    with av.open(str(out), mode="w", format="mp4") as container:
        stream = None
        for img in frames:
            if stream is None:
                stream = container.add_stream(enc, rate=Fraction(fps).limit_denominator(1001))
                stream.width, stream.height = img.shape[1], img.shape[0]
                stream.pix_fmt = "yuv420p"
                stream.bit_rate = bitrate
            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            for pkt in stream.encode(frame):
                container.mux(pkt)
        if stream is not None:
            for pkt in stream.encode():
                container.mux(pkt)


def transcode(src: str, out: str | Path, width: int, fps: float, max_frames: int | None = None) -> str:
    """Downscale a clip (video only) for API upload, optionally keeping only the first frames."""
    last = None if max_frames is None else max_frames - 1
    frames = (img for _, img in iter_all_frames(src, width, last))
    return write_video(frames, out, fps, bitrate=1_200_000)
