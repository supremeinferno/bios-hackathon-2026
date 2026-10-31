# Decision log

Short entries: what was decided, the evidence, and what would change it.

## D1 — Data and frames never enter git (Phase 0)
`data/`, `artifacts/`, `cache/`, `*.mp4` are git-ignored. Repo docs describe failures in text
only. **Why:** PDF p.2 and p.15. The repo must be public and must contain no videos, frames or
screenshots, and breaking the rule disqualifies the team.

## D2 — Blur is measured relative to each clip (Phase 0)
A fixed Laplacian-variance threshold (60) flagged 0% of frames as blurry. Visibly blurred frames
exist, and medians range from 1.8k (000003) to 8.4k (SKL), driven by texture. We switched to
"below 35% of the clip median", which flags 7–20% in SGF and 0–3% elsewhere, matching visual
inspection. **Revisit** when hand-level blur is measured in Phase 1.

## D3 — Decode once, cache downscaled frames (Phase 0)
Measured: 163 fps decode-only vs 57 fps decode+convert+resize, and 280 ms per random seek (HEVC
1080p, i5-12450H). So stages read from a frame cache, skipped frames are `grab()`ed rather than
decoded and converted, and random seeks stay out of the hot path.

## D4 — Do not swap MediaPipe handedness (M1)
PDF p.11 says MediaPipe labels hands as if the image were a mirrored selfie, so labels "may need
swapping". On `MTL-SKL00089-005` (full clip, 4,039 detections) with the swap ON, only 20.6% of
"left" hands had their wrist on the left half of the image (and "right" on the right half).
Visual check of 4 frames with both hands showed every label reversed. With MediaPipe 1.2.0 the
raw labels are already correct for our non-mirrored head-camera video. `swap_handedness=False`.
`run.json` keeps reporting `side_matches_image_half_pct` per clip as a regression check.

## D5 — Ingest decodes with PyAV, not OpenCV (Block 1)
Same clip, decode + scale to 960x540: PyAV (threaded decode, swscale resize in `to_ndarray`)
357 fps vs OpenCV 5.0 (`read` + `cv2.resize`) 106 fps. PyAV also gives exact frame timestamps.
