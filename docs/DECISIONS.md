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

## D6 — Verb list frozen at 17 verbs; `press` added (Block 1)
The PDF lists `press` among example verbs (p.13); our Phase 0 draft dropped it. While drafting
validation labels, the stone-plant worker repeatedly operates a hand-held hoist remote, which
no other verb describes. `press` was added before any predictions were scored. The `hand truck`
vocabulary entry and the `hand tool -> press/adjust` cue were added at the same time; both
objects were already noted in dev clips (DATA_REPORT §3, 000060-009 and -011).

## D7 — Our own labels: provenance and splits (Block 2)
No labels exist (PDF p.1). See `labels/README.md`. Held-out for reporting: `mytronlabs-000060-016`
(0-180 s) and `MTL-SKL00089-007` (0-60 s). Dev for the improvement pass: `mytronlabs-000060-010`
(0-60 s). Labels were drafted by Claude Code from 1 fps frame sheets and are NOT human-verified;
every score in RESULTS.md says so.

## D8 — Open-vocabulary detector: OWLv2 base, threshold 0.2 (Block 1)
Same 8 keyframes (2 per source), same per-source prompts, RTX 3050 4 GB, fp16, batch 1:

| model | speed | peak VRAM | what we saw |
|---|---|---|---|
| `google/owlv2-base-patch16-ensemble` | **7.4 img/s** | 0.42 GB | held tools found (drill 0.63, planer in hand ~0.2, cloth 0.49-0.52, tape rolls); many low-score boxes 0.15-0.25; misses big surfaces (granite slab 0.21) |
| `IDEA-Research/grounding-dino-tiny` | 2.1 img/s | 1.06 GB | finds slab (0.43), end caps (0.59), profiles; but labels from text spans often land on the wrong box ("hand truck", "metal bar" everywhere) |

OWLv2 chosen: 3.5x faster and its per-prompt scores are cleaner. Threshold raised 0.15 -> 0.2
to cut the noise floor. Licence: Apache-2.0 for both. Grounding DINO stays selectable via
`objects.model`.

## D9 — Stage 1 privacy uses OWLv2 "person" boxes, not a face detector (Stage 1)
Benchmark on 10 frames with known coworker faces, a company sign, a document and phone screens:

| detector | speed | result |
|---|---|---|
| MediaPipe Face (short-range) | 141 img/s | found 1 close face; missed all distant faces; **false boxes on hands/arms** in 3 frames |
| OWLv2 "human face" | 7.4 img/s (shared pass) | found 2 coworker faces + a face on a phone screen; missed most |
| OWLv2 "person" / "human head" | same pass | "person" found nearly every coworker (incl. small/distant); "head" found several heads |
| OWLv2 sign / document / screen | same pass | "SHARP QUEEN" sign 0.17-0.19, paper document 0.41, phone screens 0.28-0.37 |

Chosen: one OWLv2 pass per keyframe with object + privacy prompts; person box -> head band (top
35%), wearer's own arms filtered, pad 20%, hold +/-0.5 s, hands cut out of the mask, pixelation
block 14 px. Keyframes every 0.33 s (was 0.5 s) for privacy coverage. No new model downloads.
Gap: no dedicated OCR for small printed text.

## D10 — Smaller privacy regions (Stage 1)
Per-category breakdown of the first privacy run showed heads dominate and 15% start in the bottom
30% of the frame (the wearer's own knees and feet). Changes: drop head/face regions with top edge
below 70% of frame height; pad 20% -> 10%; person head band 35% -> 25%; heads/faces as ellipses;
hold +/-0.5 s -> +/-0.35 s. Mean pixelated area (when blurring), first 40 s: SGF-021 15.3% -> 3.1%,
000060-006 12.5% -> 8.7%. Checked frames: coworker faces still covered. See ERRORS_AND_FIXES #19.
