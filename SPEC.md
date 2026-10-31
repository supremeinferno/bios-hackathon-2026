# SPEC — Egocentric Vision Hackathon, PS3 (Task-step timeline)

Source of truth: `docs/problem_statement.pdf` ("Egocentric Vision Hackathon – Problem Statements",
Oct 30 2026, 17 pages). Where this file and any other instruction disagree, the PDF wins.
Page numbers below refer to the PDF.

---

## 1. PS3 — goal, inputs, outputs (p.4)

**Why it matters.** A factory task is a sequence of steps. A timeline such as
"0:12–0:20 pick up screwdriver, 0:20–0:41 fasten screws" lets a company check that steps happen
in order and measure how long each takes.

**What to build.** A program that splits each clip into actions. Each action has:

| field | meaning |
|---|---|
| start time | seconds from clip start |
| end time | seconds from clip start |
| verb | e.g. `pick`, `place`, `fasten`, `inspect` (p.4), `press` (p.13 prompt) |
| object | what is handled |
| hand | `left`, `right` or `both` (p.13 prompt) |

**Input.** One egocentric (head-mounted camera) factory clip, `.mp4`.

**Minimum goal.** Send a clip to a video-understanding API (e.g. Gemini), ask for a JSON
timeline, display it next to the video.

**Bonus ideas (PDF list, verbatim intent).**
1. Use grab and release moments from PS2 to make start/end times more precise.
2. A version that runs **fully offline** with a small local model.
3. Compare two workers doing the same task: who is faster, which steps differ.
4. Spot steps that are missing or out of order.

**Suggested output format — recommended, not mandatory (p.3–4).**

```
actions.csv : clip, start_s, end_s, verb, object, hand
```

Step 6 (p.13) also suggests improving the API path by (a) querying short windows around each
grab from PS2 instead of the whole clip and (b) giving the model a **fixed verb list** so every
clip uses the same words.

## 2. Hand tracking is required for PS3 too (p.3, p.15)

> "Hand-tracking accuracy is judged in every problem statement … Even if you choose PS3 or PS5,
> show on the demo clips how reliably your system finds the hands." (p.3)

Hand tracking is the single largest rubric item (30/100) and the tie-breaker. So our PS3
system must also emit PS1-quality hand tracks. PS1 suggested format (p.3):

```
hands.jsonl, one line per frame:
{"frame": 120, "t": 4.0, "hands": [{"side": "left", "bbox": [x1,y1,x2,y2], "score": 0.93, "kp2d": [[x,y], ...21 points]}]}
```

PS1 asks for: every hand found, left/right correct, 21 points on the right joints, **same ID
frame to frame**. PS1 bonuses we will reuse: smoothing (One-Euro), fast recovery when a hand
re-enters, fixing L/R swaps, better than baseline on gloves/blur/dark, report % frames with
hands found and FPS.

Two gotchas the PDF calls out (p.11):
- MediaPipe labels handedness **as if the image were a mirrored selfie**; head-camera video is
  not mirrored, so labels may need swapping. Verify on our footage.
- MediaPipe "world" 3D points are relative to the hand centre, not the camera.

PS2 contact (p.4, p.12), used as a PS3 bonus input: per hand per frame, touching
**nothing / portable object / fixed object**; baseline rule = ≥2 fingertips (MediaPipe points
4, 8, 12, 16, 20) inside an object box padded by 15% of hand size; contacts lasting only 1–2
frames are noise. Suggested `contacts.csv : clip, t_start, t_end, hand, state, object_name`.

## 3. Evaluation — how results are judged (p.14–16)

**There is no official metric and no automatic leaderboard.** "A panel of judges scores every
team … Judges score what they see in your demo, your code and your answers to their questions."
The clips **have no labels** (p.1).

| Criterion | Points | What judges look for |
|---|---|---|
| Hand-tracking accuracy | **30** | Every hand found, points on the right joints, L/R correct, stable through blur, gloves and occlusion **on the demo clips** |
| Results for your problem statement | 20 | Task steps working correctly on real factory footage |
| Technical approach | 15 | Sensible model choices, clearly explained; real work beyond the starter code |
| Usefulness to industry | 10 | Would a factory/robotics team use it? Clean, easy-to-use output |
| Innovation | 10 | New idea, clever tool combination, smart fix for a hard case |
| Efficiency | 5 | Speed and cost **for the hardware used** |
| Presentation and honesty | 10 | Clear demo, failure cases shown honestly, good Q&A answers |
| **Total** | **100** | Ties broken by hand-tracking accuracy |

How hand accuracy is checked (p.3–4): judges watch output on the demo clips frame by frame,
focusing on motion blur, gloves, hands holding tools, overlapping hands, hands leaving and
re-entering. "Measured numbers strengthen your case. For example, label hands on 100 frames
yourself and report how many your system got right."

Numbers that help (p.16): share of frames where hands were found, accuracy on frames we
labelled ourselves, frames per second. FAQ: "label a few minutes of one clip yourself and
measure how often you are right." Missed hands "count directly against your accuracy score".

**Consequence for `scripts/evaluate.py`:** it cannot implement an official metric because none
exists. It will implement standard metrics against **our own hand-made labels** (see §7, A1).

## 4. Deliverables and deadline (p.14–15)

Submit **before the code freeze** (exact time not stated in the PDF):

| What | Where | Requirements |
|---|---|---|
| Code | **Public** GitHub repo | README with setup steps, **the one command that runs the pipeline**, the problem statement(s) chosen, laptop vs cloud GPU. **No videos, frames or screenshots from the data in the repo.** List every open-source library, pretrained model and AI API used (p.16). |
| Demo videos | Google Drive folder shared during the event | Processed demo clips with results drawn on them, **≤ 2 minutes total**. Upload only there, never publicly. |
| Presentation | Same Drive location | 5–6 slides: problem, approach, results, what failed, what next. **GitHub link on slide 1.** |
| Live pitch | On stage | 5 min presentation + live demo, then 3 min judge Q&A. Every member must be able to explain any part of the code. |

Some clips "will be marked as demo clips"; every team must show results on them (p.2).

Suggested 24h plan from the PDF (p.14): 0–3h setup + minimum goal + first demo video;
3–10h main improvement; 10–16h all clips, failure cases, fix biggest; 16–21h speed/polish, demo
clips, numbers for slides; 21–23h README/slides/demo video, push before freeze; 23–24h rehearse.

## 5. Constraints (p.2, p.5–8, p.16)

- **Compute:** organisers provide no GPUs or API credits. Laptop CPU path or free Colab/Kaggle
  GPU path; mixing is allowed; tell the judges which was used. Efficiency is scored against
  hardware. PDF laptop guidance: use 480p, process every 2nd/3rd frame, interpolate gaps.
  **Our hardware:** Windows 11 laptop, NVIDIA RTX 3050 Laptop GPU **4 GB VRAM**, CUDA 13.1 driver.
- **Allowed tools:** any open-source library, pretrained model or AI API. List them all in the
  README.
- **Online vs offline:** cloud AI APIs are allowed, including sending clips (p.2). Fully-offline
  is a PS3 bonus. API keys must come from environment variables, never in code or repo (p.8).
  Save every API response to disk; expect free-tier rate limits; process clips one at a time.
  Gemini samples ~1 frame/s by default and can miss fast hand motions (p.8).
- **Code provenance:** all project code written during the 24 hours; prep (installs, weights,
  reading) allowed beforehand. AI coding assistants allowed, but judges will ask us to explain.
- **Team:** 2–4 students.
- **Licensing:** PDF imposes no licence restriction beyond the data rules. We will still record
  each model's licence in the README.

### Data-use rules (p.2) — breaking these means disqualification

- Use the data **only for this hackathon**.
- Do **not** post clips, frames or screenshots publicly (GitHub, LinkedIn, YouTube, Instagram).
- Do **not** commit video files; keep them in a `.gitignore`d folder.
- Clips **may** be sent to cloud AI APIs.
- **Delete every copy** of the data when the event ends, including Colab/Kaggle/API file storage.
- Clips showing faces, screens or company documents must not appear in public posts.

## 6. Data actually received vs. what the PDF describes

| PDF says (p.1–2) | What we received |
|---|---|
| `egohack-data/clips/factory_001.mp4 …`, **20 clips, ~3 min each** | Drive folder `Hackathon_vidoes/Sample/` with **40** `.mp4` clips named by source: `MTL-SGF00106-*` (10), `MTL-SKL00089-*` (10), `mytronlabs-000003-*` (10), `mytronlabs-000060-*` (10) |
| `clips_480p/` downscaled copies | **Not provided** |
| `README.txt` (resolution, fps, camera) | **Not provided** |
| Some clips marked as demo clips | **No marking yet** |
| No labels | Confirmed: **no label/annotation files** |

Per-clip technical details are in `docs/DATA_REPORT.md`.

## 7. Ambiguities and the assumptions we make

| # | Ambiguity | Assumption / decision |
|---|---|---|
| A1 | No labels and no official metric, yet we must "measure" accuracy. | We create a small, clearly-disclosed **in-house validation set**: (a) hand boxes + side on ≥100 frames sampled across sources, emphasising hard cases (blur, gloves, overlap, re-entry); (b) action segments (start, end, verb, object, hand) for ≥2 **held-out** clips from different sources. `evaluate.py` reports hand recall/precision @IoU0.5, L/R accuracy, ID-switch count, % frames with hands, FPS; for actions: segmental F1@{10,25,50}, edit score, frame-wise verb accuracy, and verb/object/hand accuracy on matched segments. Labels are made by the team with our labelling tool (model pre-labels allowed, every label human-checked). Validation clips are never used for tuning prompts/thresholds. |
| A2 | Verb vocabulary is open ("pick, place, fasten, inspect…"). | A **fixed, closed verb list** (PDF step 6 recommends it), drafted from what is visible in the 4 sources (`docs/DATA_REPORT.md` §4) and frozen before validation labelling. Draft (16): `pick, place, hold, carry, insert, attach, fasten, cut, sand, grind, wipe, push, adjust, inspect, write, other`. `sand` covers plane/sand; `grind` covers grind/polish; `attach` covers end caps, tape and clips; `push` covers sliding/tilting slabs. |
| A3 | Object names are free text. | Free text from the model, normalised to a canonical per-source vocabulary (lower-case, singular, synonym map) so "screw driver"/"screwdriver" match. |
| A4 | `hand` values. | `left`, `right`, `both`; plus `none` only for camera-only actions like `inspect` with no hand in contact. |
| A5 | Can actions overlap? (left hand holds while right fastens) | Actions of **different hands may overlap**; actions of the same hand may not. A `both` action blocks both hands. Gaps (idle) are simply not listed. |
| A6 | Time precision. | Float seconds, 2 decimals, `0 ≤ start_s < end_s ≤ clip duration`; sorted by `start_s`. |
| A7 | Output format is only "suggested". | We emit exactly the suggested `actions.csv` (plus `hands.jsonl` in the PS1 format, and `contacts.csv` in the PS2 format) so judges can compare, plus a richer internal JSON. |
| A8 | Demo clips not yet marked. | Pipeline must run on **any** clip without per-clip tuning. Until organisers announce demo clips, we pick 1 representative clip per source for demos. |
| A9 | No 480p copies provided. | We downscale ourselves (configurable width, default 640 px) at decode time; full-res stays available for the hand model if it helps. |
| A10 | API vs offline. | **Primary path: hybrid** — local hand tracking + contact events give precise boundaries; a video-language API (Gemini, free tier) names verb/object per segment. **Offline path** as bonus/fallback: local models only (4 GB VRAM limits us to small/quantised models; to be measured). No `GEMINI_API_KEY` is currently set — the team must create one (Google AI Studio) and export it. |
| A11 | Prompt asks for a frame-based failure gallery in `docs/RESULTS.md` and sampled frames in the data report. | The PDF forbids frames/screenshots from the data **in the repo** (and the repo must be public). So all frames and rendered videos go to the git-ignored `artifacts/` folder; repo docs describe failures **in text** and point to local paths. Image galleries go in the slides (private Drive) only. |
| A12 | "Known procedure" for step-order checks. | Not given. Clips appear to come from 4 recording sources, likely repeated tasks per source. We derive a **procedure template per source** from our timelines (consensus alignment), then flag missing / out-of-order / repeated steps against it and compare workers (two PS3 bonuses). |
| A13 | Code-freeze time not specified. | Treat end of the 24h window as the freeze; aim to be submission-ready by hour 21 per the PDF plan. |
| A14 | Hand tracking is 30/100 even for PS3. | Hand tracking gets **PS1-level investment** (model comparison, tracking, smoothing, L/R fix, gap recovery, measured accuracy), not just a feature for PS3. |
| A15 | The repo lives under `OneDrive\Desktop`. | OneDrive is not running right now. If it starts syncing, 4.4 GB of company video in `data/raw/` would be copied to the cloud, which the data rules make us delete later. Recommendation: keep the data outside OneDrive, or exclude the folder from sync. |
