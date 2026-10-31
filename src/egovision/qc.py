"""Stage 0 quality control: per processed frame sharpness, brightness and a lens-covered flag.

Blur is relative to the clip's own median sharpness, because absolute Laplacian variance is
dominated by scene texture (sawdust, granite) and differs 4x between sources (docs/DECISIONS.md D2).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from egovision.config import QCConfig


def frame_qc(img: np.ndarray, size: int) -> tuple[float, float, float]:
    """(sharpness, mean brightness, brightness std) on a downscaled grey copy."""
    h, w = img.shape[:2]
    g = cv2.cvtColor(cv2.resize(img, (size, int(round(h * size / w))), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var()), float(g.mean()), float(g.std())


def summarize(frames: list[int], fps: float, raw: list[tuple[float, float, float]], cfg: QCConfig) -> dict[str, Any]:
    if not raw:
        return {"frames": [], "summary": {"processed_frames": 0}}
    a = np.array(raw)
    med = float(np.median(a[:, 0])) or 1.0
    rel = a[:, 0] / med
    blurry = rel < cfg.blur_rel_threshold
    covered = (a[:, 1] < cfg.covered_max_brightness) | (a[:, 2] < cfg.covered_max_std)
    per = [{"frame": f, "t": round(f / fps, 3), "sharp_rel": round(float(r), 3), "brightness": round(float(b), 1),
            "blurry": bool(bl), "covered": bool(cv)}
           for f, r, b, bl, cv in zip(frames, rel, a[:, 1], blurry, covered, strict=True)]
    return {
        "frames": per,
        "summary": {
            "processed_frames": len(per), "sharpness_median": round(med, 1),
            "blurry_pct": round(100 * float(blurry.mean()), 1), "covered_pct": round(100 * float(covered.mean()), 1),
            "brightness_mean": round(float(a[:, 1].mean()), 1),
            "blur_rule": f"sharpness < {cfg.blur_rel_threshold} x clip median",
        },
    }
