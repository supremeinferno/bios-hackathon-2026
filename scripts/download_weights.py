"""Download model weights into weights/ (git-ignored). Re-running skips files that verify."""

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
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(dest: Path = Path("weights")) -> int:
    dest.mkdir(exist_ok=True)
    for name, (url, digest) in WEIGHTS.items():
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
    sys.exit(main())
