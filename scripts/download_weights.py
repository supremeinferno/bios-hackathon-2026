"""Download model weights into weights/ (git-ignored). Re-running skips files that verify.

    python scripts/download_weights.py                                # required weights only
    python scripts/download_weights.py rtdetr_hand_detector_v1.onnx   # + optional RT-DETR recovery (D16)
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

WEIGHTS = {
    # MediaPipe Hand Landmarker, float16 bundle (Apache-2.0). URL from the hackathon PDF, step 0.
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/"
        "hand_landmarker.task",
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    ),
    # NVIDIA RT-DETR Hand Detector v1 (OpenMDW-1.1), Hugging Face nvidia/RT-DETR-Hand-Detector-v1 at a pinned
    # revision; 80,371,970 bytes. Optional: only for hands.recover_detector="rtdetr" (DECISIONS D16).
    "rtdetr_hand_detector_v1.onnx": (
        "https://huggingface.co/nvidia/RT-DETR-Hand-Detector-v1/resolve/"
        "55572ddc93a5916c3e486811dc6e62bb2b7bdb9d/hand_detector.onnx",
        "85d36428cc1ecf8832ca2d48f7aa04df0659cf94dde358352fb3cf651a319810",
    ),
}
REQUIRED = ["hand_landmarker.task"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(names: list[str] | None = None, dest: Path = Path("weights")) -> int:
    dest.mkdir(exist_ok=True)
    names = REQUIRED + [n for n in (names or []) if n not in REQUIRED]
    unknown = [n for n in names if n not in WEIGHTS]
    if unknown:
        print(f"ERROR unknown weights {unknown}; known: {sorted(WEIGHTS)}", file=sys.stderr)
        return 2
    for name in names:
        url, digest = WEIGHTS[name]
        p = dest / name
        if p.is_file() and (digest is None or sha256(p) == digest):
            print(f"ok    {p} ({p.stat().st_size} bytes, sha256 {sha256(p)[:16]}...)")
            continue
        print(f"fetch {url}")
        tmp = p.with_suffix(p.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        if digest is not None and sha256(tmp) != digest:
            tmp.unlink()
            print(f"ERROR checksum mismatch for {name}", file=sys.stderr)
            return 1
        tmp.replace(p)
        print(f"saved {p} ({p.stat().st_size} bytes, sha256 {sha256(p)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
