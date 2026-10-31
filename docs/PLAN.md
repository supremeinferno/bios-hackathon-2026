# Build plan (time-boxed)

H0 = start of the build session. The PDF's own 24h plan (SPEC.md §4) is the outer frame. Hand
tracking gets PS1-level effort because it is 30/100 for every PS and the tie-breaker.

| # | Hours | Milestone | Done when |
|---|---|---|---|
| M0 | H0–1 | **Phase 0: spec and data** | `SPEC.md`, `docs/DATA_REPORT.md`, `.gitignore` in place ✅ |
| M1 | H1–3 | **Thin end-to-end path** | `src/egovision/` skeleton with typed config; ingest with a cached, strided decode; MediaPipe hands → `hands.jsonl`; Gemini whole-clip → `actions.csv`; overlay video; 10-second smoke test passing; first commit |
| M2 | H3–6 | **Measure it (MVP)** | Labelling tool (hand boxes + side; action segments); ~120 labelled hand frames + ≥2 held-out action clips; `scripts/evaluate.py` gives baseline numbers. **This is the MVP: video in, PS3 file out, validation score.** |
| M3 | H6–11 | **Hand tracking quality** | Compare ≥2 hand stacks on our labelled frames (MediaPipe full-frame vs. detector→crop→landmarks cascade, e.g. RTMDet/RTMPose-hand via rtmlib or a YOLO hand detector; resolution; fisheye undistortion on/off). Add: own-hand filter (rejects coworker hands and feet), IoU/Hungarian tracking with stable IDs, L/R swap fix, One-Euro smoothing, gap interpolation. Report recall/precision/L-R accuracy/ID switches/FPS per variant in `docs/DECISIONS.md` |
| M4 | H11–15 | **Task-step quality (PS3 bonuses)** | Contact and grab/release events (hands + open-vocab objects on keyframes); boundary refinement from those events; per-window VLM labelling with the fixed verb list; temporal clean-up (min length, merge, per-hand overlap rules); per-source step template giving missing/out-of-order/repeated steps and worker comparison; offline path measured; ablations |
| M5 | H15–19 | **Product layer + full run** | CLI (`run`, `batch`, `eval`), FastAPI job service, dashboard (video + overlays + synced timeline + stats), docker-compose; batch over all 40 clips; per-stage throughput table |
| M6 | H19–22 | **Submission** | README (Mermaid diagram, one-command run, laptop/GPU, every library/model/API with licence), `docs/RESULTS.md` (honest numbers, ablations, text-only failure cases), demo videos ≤ 2 min on private Drive, 5–6 slides, push before freeze |
| — | H22–24 | Buffer + rehearsal | Everyone can explain every stage |

## Key technical bets (to be validated by measurement, not assumed)

1. **Hands:** MediaPipe is the baseline. The likely gain comes from a cascade: hand detector →
   crop → landmarks. Crops make close/edge hands bigger in the landmark model's input, and the
   detector can be filtered to the wearer's hands.
2. **Boundaries come from hands, labels come from a VLM.** Local hand/contact signals give precise
   start/end; Gemini (or a local VLM offline) names verb + object per window. This beats
   whole-clip prompting, which samples about 1 fps and gives loose times.
3. **Repetition is structure.** Each source repeats one task family, so a per-source template
   supports step-order checks and worker comparison without labels.

## Not applicable from the original brief (and why)

- *Supervised temporal model (MS-TCN/ASFormer)*: needs labels, and there are none. We use the
  zero/few-shot branch the brief describes for that case.
- *Official metric*: none exists (judged by a panel). `evaluate.py` uses our own disclosed
  labels.
- *Frame gallery in repo docs*: forbidden by the data rules; galleries go in the private slides only.
