"""Measure Stage 3 options on real clips: baseline vs +recovery vs +recovery+coworker filter.

    python scripts/measure_hands.py --clips MTL-SKL00089-005 mytronlabs-000060-016 --seconds 60

Label-free numbers (share of processed frames with a hand / both hands, recovered / dropped counts,
hand-model fps) plus example crops of recovered and dropped hands in artifacts/measure/ (git-ignored)
for a visual check. Recovered or dropped hands can be wrong: look at the crops before trusting a gain.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from egovision.config import Config
from egovision.hands import make_backend
from egovision.ingest import FrameSource, probe

VARIANTS = {
    "baseline": {"recover": False, "filter_other_person": False},
    "+recovery": {"recover": True, "filter_other_person": False},
    "+recovery+filter": {"recover": True, "filter_other_person": True},
}


def run_variant(video: Path, seconds: float, opts: dict, crops_dir: Path | None) -> dict:
    cfg = Config.load(None, [f"ingest.max_seconds={seconds}"])
    cfg = cfg.model_copy(update={"hands": cfg.hands.model_copy(update=opts)})
    meta = probe(video)
    src = FrameSource(meta, cfg.ingest)
    n = with_hand = two = 0
    t_model = 0.0
    examples: list[np.ndarray] = []
    with make_backend(cfg.hands, meta, expected=2) as hb:
        for fr in src:
            t0 = time.perf_counter()
            hands = hb.process(fr)
            t_model += time.perf_counter() - t0
            n += 1
            with_hand += bool(hands)
            two += len(hands) >= 2
            if crops_dir is not None and len(examples) < 12:
                for h in hands:
                    if h.source == "recovered":
                        s = fr.scale
                        x1, y1, x2, y2 = (int(v * s) for v in h.bbox)
                        pad = 30
                        crop = fr.image[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
                        if crop.size:
                            examples.append(cv2.resize(crop, (160, 160)))
        counters = dict(getattr(hb, "counters", {}))
    if crops_dir is not None and examples:
        crops_dir.mkdir(parents=True, exist_ok=True)
        while len(examples) % 6:
            examples.append(np.zeros((160, 160, 3), np.uint8))
        rows = [np.hstack(examples[i:i + 6]) for i in range(0, len(examples), 6)]
        cv2.imwrite(str(crops_dir / f"{video.stem}_recovered.jpg"), np.vstack(rows))
    return {"frames": n, "with_hand_pct": round(100 * with_hand / max(n, 1), 1),
            "two_hands_pct": round(100 * two / max(n, 1), 1), "hand_model_fps": round(n / max(t_model, 1e-9), 1),
            **counters}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--out", type=Path, default=Path("artifacts/measure"))
    a = ap.parse_args()
    report = {}
    for clip in a.clips:
        video = Path("data/raw") / f"{clip}.mp4"
        report[clip] = {}
        for name, opts in VARIANTS.items():
            r = run_variant(video, a.seconds, opts, a.out if name == "+recovery" else None)
            report[clip][name] = r
            print(clip, name, r, flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "hands_variants.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
