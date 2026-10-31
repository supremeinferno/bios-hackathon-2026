"""Render one annotated MP4 for the presentation from finished runs (PDF: demo videos <= 2 min total).

    python scripts/make_demo.py --clip MTL-SKL00089-005:20:30 --clip mytronlabs-000060-010:40:30 --out artifacts/demo.mp4

Each --clip is CLIP:START_S:DURATION_S and needs outputs/CLIP/ from `egovision run`. Each part
starts with a 1.5 s title card. Output goes to the git-ignored artifacts/ (data-use rules).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from egovision.config import Config
from egovision.export import load_result
from egovision.ingest import iter_all_frames
from egovision.media import write_video
from egovision.render import draw_frame, timeline_image, verb_colors

MAX_TOTAL_S = 120.0


def part_frames(out_root: Path, clip: str, start: float, dur: float, width: int) -> Iterator[np.ndarray]:
    r = load_result(out_root / clip)
    meta, hands, objects, actions = r["meta"], r["hands"], r["objects"], r["actions"]
    cfg = Config().render.model_copy(update={"width": width})
    verbs = Config().steps.verbs
    s = width / meta.width
    tl = timeline_image(actions, meta.duration_s, width, cfg.timeline_height, verb_colors(verbs))
    f0, f1 = int(start * meta.fps), min(len(hands), int((start + dur) * meta.fps))
    keys = [fo.frame for fo in objects]
    title = None
    for i, img in iter_all_frames(meta.path, width, f1 - 1):
        if i < f0:
            continue
        if title is None:  # 1.5 s title card in the clip's own frame size
            card = np.zeros((img.shape[0] + tl.shape[0], img.shape[1], 3), np.uint8)
            cv2.putText(card, clip, (30, card.shape[0] // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
            cv2.putText(card, f"{start:.0f}s - {start + dur:.0f}s  |  hands, held objects, task steps",
                        (30, card.shape[0] // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            title = card
            for _ in range(int(1.5 * meta.fps)):
                yield card
        t = i / meta.fps
        import bisect

        j = bisect.bisect_right(keys, i) - 1
        draw_frame(img, hands[i], objects[j] if j >= 0 else None, [a for a in actions if a.start_s <= t < a.end_s],
                   t, s, cfg)
        strip = tl.copy()
        x = int(t / meta.duration_s * width)
        cv2.line(strip, (x, 0), (x, strip.shape[0]), (255, 255, 255), 2)
        yield np.vstack([img, strip])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", action="append", required=True, help="CLIP:START_S:DURATION_S")
    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/demo.mp4"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()
    parts = []
    for spec in a.clip:
        clip, start, dur = spec.rsplit(":", 2)
        parts.append((clip, float(start), float(dur)))
    total = sum(d for _, _, d in parts) + 1.5 * len(parts)
    if total > MAX_TOTAL_S:
        print(f"warning: demo is {total:.0f}s; the PDF allows 2 minutes in total", file=sys.stderr)

    def all_frames() -> Iterator[np.ndarray]:
        for clip, start, dur in parts:
            yield from part_frames(a.outputs, clip, start, dur, a.width)

    enc = write_video(all_frames(), a.out, a.fps, bitrate=6_000_000)
    print(f"wrote {a.out} ({total:.0f}s, encoder {enc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
