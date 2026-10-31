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
