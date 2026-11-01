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

## D11 — Privacy limited to faces and company names (Stage 1)
User decision: blur only faces and company names. Removed the `text`, `document`, `screen` and
`monitor` prompts, which matched tools and parts (printed templates, tape rolls, shiny metal parts).
Kept: `person` (as head band), `head`, `face`, `sign`, `logo`. Check on SGF-023, first 30 s:
regions = 64 heads, 2 faces, 3 signs; mean pixelated area 6.5% of the frame when blurring;
the nail gun, printed template and planer are no longer pixelated.
Trade-off: company names on small printed items are only caught if OWLv2 reads them as a sign or
logo; OCR with a company-name list would close that gap.

## D13 — Gemini: offline always runs; model 3.8-flash with fallbacks (Stage 5)
`gemini-3.5-flash` (from the PDF) returned 503 overload on two runs; `gemini-2.5-flash` is listed for
our key but 404s ("not available to new users", recommends 3.8-flash). Main model `gemini-3.8-flash`,
fallbacks 3.7 -> 3.5 -> flash-latest, 300 s timeout, 4xx not retried. The offline exemplar matcher
always runs and is used when Gemini fails, so a VLM outage never loses a run.

## D14 — Gemini on 30 s windows, main model gemini-3.5-flash (Stage 5)
Measured: full 3-min clips -> 503 on all models (two runs, ~1 h apart); 10 s sanitized clip -> 3.5-flash
3.6 s, flash-lite-latest 2.9 s, 3.8-flash 178 s. So pass 1 uses 30 s windows; main model 3.5-flash,
fallbacks flash-lite-latest then 3.8-flash; pass 2 re-asks windows around actions with verb "other" or
confidence < 0.5 at 4 fps. First held-out result (000060-016, our draft labels): Gemini raw F1@50 45.9
vs offline matcher 0.0 (the offline matcher depends on visible hands; ~15% of frames on this clip).

## D12 — Stage 3 recovery and coworker filter: OFF by default (measured)
`scripts/measure_hands.py`, first 60 s of SKL-005, 000060-016, SGF-036, 000003-002 (frames with a hand,
baseline -> +recovery): 92.0->93.0, 11.8->11.8, 56.3->55.6, 90.4->90.4 %; hand-model fps roughly halves.
Recovered "hands" include rollers, a foot and forearms. The coworker filter dropped 12 of the wearer's
own hands on 000003-002. Both stay available in config; neither is used for results.

## D15 — Step defaults chosen on the dev clip: Gemini pass 1 + verification + snapping; pass 2 off
Dev clip `mytronlabs-000060-010` (0-60 s, 14 draft segments, labelled before any prediction was seen),
`scripts/ablate_steps.py`:

| variant | F1@10 | F1@25 | F1@50 | edit | hand acc | frame acc |
|---|---|---|---|---|---|---|
| offline matcher | 26.1 | 8.7 | 0.0 | 18.2 | 66.7 | 6.5 |
| Gemini pass 1 | 35.3 | 17.6 | 11.8 | 33.2 | 41.7 | 22.6 |
| Gemini pass 1 + pass 2 | 16.7 | 16.7 | 11.1 | 28.2 | 25.0 | 21.4 |
| pass 1 + verify | 35.3 | 17.6 | 11.8 | 34.8 | 41.7 | 31.9 |
| pass 1 + verify + snap | 35.3 | 17.6 | 11.8 | 34.8 | 41.7 | 31.9 |

Pass 2 off; verification and snapping on (never worse on dev). Tiny sample: one minute, 14 segments.
Disclosure: the snapping bug (ERRORS #28) was noticed on the held-out clip; the fix is a logic
correction (missing != zero), not a tuned threshold. Held-out numbers are reported for raw and final.

## D16 — RT-DETR hand detector as a recovery source: better than CLAHE, kept OFF by default (for now)
NVIDIA RT-DETR Hand Detector v1 (`nvidia/RT-DETR-Hand-Detector-v1`, `hand_detector.onnx`, 80.4 MB,
OpenMDW-1.1, sha256-pinned in `scripts/download_weights.py`; `onnxruntime-gpu==1.26.0`, extra `rtdetr`).
On this laptop: 22 img/s on the RTX 3050 (CUDA), 2.3 img/s on CPU. Only its 2D boxes are used: when
MediaPipe finds fewer hands than expected, each box is cropped (padded square, side x1.5), MediaPipe runs
on the crop, keypoints are mapped back; boxes with no MediaPipe hand are discarded. MediaPipe's own hands
are never changed (regression test: default output identical to before).

`scripts/measure_hands.py`, first 60 s, every 2nd frame at 960 px (run while the 40-clip batch used the
same CPU/GPU, so fps are comparable between variants but low in absolute terms):

| clip | frames with >=1 hand: base / CLAHE / RT-DETR | both hands | recovered: CLAHE / RT-DETR | hand fps: base / CLAHE / RT-DETR |
|---|---|---|---|---|
| SKL-005 | 91.0 / 93.0 / 93.0 | 67.6 / 69.6 / 70.8 | 36 / 47 | 23.3 / 23.0 / 16.8 |
| 000003-002 | 89.2 / 90.4 / 91.3 | 44.2 / 46.2 / 48.4 | 29 / 57 | 26.7 / 19.8 / 9.5 |
| SGF-036 | 54.4 / 55.6 / 61.1 | 3.2 / 3.3 / 6.1 | 11 / 86 | 37.3 / 12.7 / 5.2 |
| 000060-016 | 10.2 / 11.8 / 14.7 | 0.7 / 1.0 / 1.1 | 17 / 44 | 44.2 / 20.7 / 10.8 |

RT-DETR boxes with no MediaPipe hand inside (discarded): 212/415, 459/1014, 995/1540, 661/786.
Visual check of all 234 RT-DETR-recovered hands (crops in artifacts/measure/, git-ignored): ~207 real
wearer hands (~88%); wrong: 4 bare feet and 1 knee (SGF, SKL), 1 pink roller (stone plant; CLAHE had 7 of
12), ~6 coworker hands (000003), ~5 forearm-only crops, 5 crops with no hand (blurred planks, objects);
~8 too blurred to tell. Baselines here are 1-2 pts below D12's because these runs read the JPEG frame
cache while D12 decoded directly; comparisons within this table use identical frames.

Decision: RT-DETR recovery beats CLAHE on gain (+2.0 to +6.7 pts vs +1.2 to +2.0) and on precision.
It stays OFF by default because (a) it slows the hand stage 1.4-7x on clips where MediaPipe often misses
hands, (b) ~8% of its additions are wrong (feet, coworker hands), and (c) switching the default changes
every output and needs a re-run of all clips. Enable per run with
`--set hands.recover=true --set hands.recover_detector=rtdetr` (e.g. for the demo clips).

## D17 — Cycle consistency score, labour cost and automation candidate indicator (2026-11-01)

New `egovision sop-cycles` (egovision/sop_cycles.py, config `sop_cycles`); `egovision sop` and the default pipeline
are unchanged. Describes tasks and steps, never individual people; the indicator is not a replacement decision.

- **Cycles.** analytics.py:65-73 uses the most frequent step per clip as the cycle marker; on our data that is a
  `hold:*` step (a background state) for 3 of 4 sources, chosen per clip, and it counts unverified actions. So:
  per task group, the anchor is a non-background step (not hold/other/wait/walk) that repeats in >= 2 clips and
  maximises intervals x (1 - min(CV of its period, 1)). A cycle runs from one anchor start to the next, so the
  partial segments at the clip edges are never counted. Fewer than 3 complete cycles: "insufficient data".
- **Task groups.** One source can hold several tasks (MTL-SKL00089: battery packs vs aluminium profiles), so
  clips are grouped by the overlap of their verified step sets (Jaccard >= 0.2), with a config override.
- **Score.** `100 * (w_order * (1 - mean order distance) + w_timing * (1 - min(CV of cycle time / 0.5, 1)))
  / (w_order + w_timing)`, w_order = w_timing = 0.5. Order distance = Levenshtein distance between a cycle's
  step sequence and the task template, divided by the longer length. Equal weights because neither part is
  measured as more important; a cycle-time CV of 0.5 or more already means no timing consistency.
- **Indicator.** `candidate`: step in >= 80% of cycles, duration CV <= 25%, task order distance <= 0.3 and a verb
  for repeatable gross motion (pick, place, push, carry). `keep manual`: judgement or fine-manipulation verbs
  (adjust, inspect, write, attach, fasten) or CV > 50%. Otherwise `needs review`.
- **Wage.** Team-declared (`sop_cycles.wage.currency`, `rate_per_hour`), never inferred; without it every cost
  is null and only labour hours are reported. No wage is declared in this repository.
- **Measured on outputs/** (RESULTS section 6): 14, 15, 3 and 7 complete cycles in four task groups, stone plant
  none; scores 18-33, mainly from label noise and short clips, so no step is a `candidate`.

