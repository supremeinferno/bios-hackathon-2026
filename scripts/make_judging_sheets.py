"""Hand-tracking judging kit (PDF p.4: "label hands on 100 frames yourself and report how many your
system got right").

Samples frames from finished runs, draws our hand predictions large and numbered, tiles them into
sheets under artifacts/judging/ (git-ignored), and writes a CSV to fill in, one row per frame:
  gt_hands     wearer hands visible in the frame (0-2; ignore other people's hands)
  correct      predicted hands sitting on a real wearer hand
  false_pos    predicted hands on nothing / on another person / on a foot
  side_errors  correct hands with the wrong left/right label
  kp_bad       correct hands whose keypoints are clearly off the joints
Then `egovision eval` reports recall / precision / side accuracy / keypoint-OK.

    python scripts/make_judging_sheets.py --clips MTL-SKL00089-007 mytronlabs-000060-016 --per-clip 25
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from egovision.config import Config
from egovision.export import load_result
from egovision.ingest import decode_frames
from egovision.render import draw_frame

TILE_W = 640


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--per-clip", type=int, default=25)
    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    ap.add_argument("--sheets", type=Path, default=Path("artifacts/judging"))
    ap.add_argument("--csv", type=Path, default=Path("labels/hands_judged.csv"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    a.sheets.mkdir(parents=True, exist_ok=True)
    cfg = Config().render.model_copy(update={"width": TILE_W, "draw_objects": False})
    rows = []
    for clip in a.clips:
        r = load_result(a.outputs / clip)
        meta, hands = r["meta"], r["hands"]
        # Stratified: half uniform in time, half from "hard" frames (no hand, interpolated, low score).
        n = len(hands)
        hard = [i for i in range(0, n, 2) if not hands[i].hands or any(h.interp or h.score < 0.7 for h in hands[i].hands)]
        uni = list(range(0, n, max(1, n // a.per_clip)))[: a.per_clip - a.per_clip // 2]
        pick = sorted(set(uni) | set(rng.sample(hard, min(len(hard), a.per_clip // 2))))
        tiles = []
        for i, img in decode_frames(meta.path, (TILE_W, int(round(meta.height * TILE_W / meta.width))), set(pick)):
            draw_frame(img, hands[i], None, [], i / meta.fps, TILE_W / meta.width, cfg)
            cv2.putText(img, f"#{len(rows) + len(tiles)}  {clip} f{i}", (8, img.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            tiles.append((i, img))
        for k in range(0, len(tiles), 6):
            grid = [t for _, t in tiles[k:k + 6]]
            while len(grid) < 6:
                grid.append(np.zeros_like(grid[0]))
            sheet = np.vstack([np.hstack(grid[0:2]), np.hstack(grid[2:4]), np.hstack(grid[4:6])])
            cv2.imwrite(str(a.sheets / f"{clip}_{k // 6:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        rows += [{"clip": clip, "frame": i, "gt_hands": "", "correct": "", "false_pos": "", "side_errors": "",
                  "kp_bad": "", "notes": ""} for i, _ in tiles]
    a.csv.parent.mkdir(parents=True, exist_ok=True)
    if a.csv.is_file():
        print(f"{a.csv} exists; not overwriting. New template written next to it.", file=sys.stderr)
        a.csv = a.csv.with_suffix(".template.csv")
    with open(a.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} frames -> sheets in {a.sheets}/, fill in {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
