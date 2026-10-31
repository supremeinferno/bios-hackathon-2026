"""NVIDIA RT-DETR Hand Detector v1 (ONNX) as a second source for missed-hand recovery (DECISIONS D16).

Model: Hugging Face `nvidia/RT-DETR-Hand-Detector-v1`, file `hand_detector.onnx` (OpenMDW-1.1), the 2D box
detector Isaac Capture's EgoHand uses. Only this 2D detector is used: no EgoHand 3D reconstruction, no
ViPE, nothing that needs calibration or depth. Preprocessing follows Isaac Capture's
`inference/hand_detector_2d/rtdetr/onnx_model.py`: plain (non-letterbox) resize to 640x640 RGB in [0, 1];
the graph rescales boxes to `orig_target_sizes`; label 1 = left, 2 = right (anatomical, the wearer's).

The detector only proposes boxes. Keypoints still come from MediaPipe, run on a padded square crop
around each box; a box without a MediaPipe hand inside is discarded (mediapipe_backend.py).
`onnxruntime` is imported lazily so MediaPipe-only installs work.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from egovision.schema import Hand

log = logging.getLogger(__name__)

INPUT_SIZE = 640  # the ONNX graph was traced at 640x640
LABEL_TO_SIDE = {1: "left", 2: "right"}


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def square_crop(box: list[float], pad: float, w: int, h: int) -> tuple[int, int, int, int]:
    """Square crop around `box` with side max(bw, bh) * (1 + pad), shifted (not shrunk) to stay inside
    the image; only shrunk if it is larger than the image. Returns integer pixel bounds (x1, y1, x2, y2)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = min(max(box[2] - box[0], box[3] - box[1]) * (1 + pad), float(w), float(h))
    x1, y1 = cx - side / 2, cy - side / 2
    x1 = min(max(x1, 0.0), w - side)
    y1 = min(max(y1, 0.0), h - side)
    return int(round(x1)), int(round(y1)), int(round(x1 + side)), int(round(y1 + side))


def crop_to_frame(points: list[tuple[float, float]], crop: tuple[int, int, int, int],
                  scale: float) -> list[list[float]]:
    """Normalised (0..1) crop coordinates -> original-video pixels.

    `crop` is in processing-resolution pixels; `scale` = processing width / original width (Frame.scale).
    """
    x1, y1, x2, y2 = crop
    cw, ch = x2 - x1, y2 - y1
    return [[(x1 + min(max(px, 0.0), 1.0) * cw) / scale, (y1 + min(max(py, 0.0), 1.0) * ch) / scale]
            for px, py in points]


def accept(candidate: Hand, existing: list[Hand], max_iou: float, min_score: float) -> bool:
    """A recovered hand is added only for a side not already present, with little overlap, and confident."""
    if candidate.side in {h.side for h in existing}:
        return False
    if any(iou(candidate.bbox, h.bbox) > max_iou for h in existing):
        return False
    return candidate.score >= min_score


class RTDetrHandDetector:
    """Thin onnxruntime wrapper. `detect` takes a BGR image and returns (side, score, box) in its pixels."""

    def __init__(self, weights: Path, score_threshold: float):
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError("hands.recover_detector='rtdetr' needs onnxruntime: "
                              "pip install -e .[rtdetr]") from e
        if not Path(weights).is_file():
            raise FileNotFoundError(f"{weights} missing; run `python scripts/download_weights.py "
                                    f"rtdetr_hand_detector_v1.onnx`")
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            try:
                ort.preload_dlls()  # CUDA/cuDNN DLLs from the installed PyTorch or nvidia-* wheels
            except Exception as e:  # noqa: BLE001 - fall back to CPU, never fail the run
                log.warning("rtdetr: could not preload CUDA DLLs (%s); CPU may be used", e)
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(weights), providers=providers)
        self.score_threshold = score_threshold
        log.info("rtdetr: %s on %s", Path(weights).name, self.session.get_providers()[0])

    @staticmethod
    def preprocess(bgr: np.ndarray) -> np.ndarray:
        from PIL import Image  # same resize as the reference implementation (PIL bilinear)

        rgb = Image.fromarray(np.ascontiguousarray(bgr[..., ::-1]))
        arr = np.asarray(rgb.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        return arr.transpose(2, 0, 1)[None]

    def detect(self, bgr: np.ndarray) -> list[tuple[str, float, list[float]]]:
        h, w = bgr.shape[:2]
        labels, boxes, scores = self.session.run(
            ["labels", "boxes", "scores"],
            {"images": self.preprocess(bgr), "orig_target_sizes": np.array([[w, h]], dtype=np.int64)})
        out = [(LABEL_TO_SIDE.get(int(lab), "unknown"), float(sc), [float(v) for v in box])
               for lab, box, sc in zip(labels[0], boxes[0], scores[0], strict=True) if sc >= self.score_threshold]
        return sorted(out, key=lambda d: -d[1])
