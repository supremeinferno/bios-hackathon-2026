# Results

Every number here was measured on our laptop (i5-12450H, RTX 3050 Laptop 4 GB) on the real sample
clips. **No official labels exist** (PDF p.1). Step scores use **our own draft labels, made by Claude Code
from 1 frame/s contact sheets and not yet checked by a person** ([labels/README.md](../labels/README.md)).
Treat them as indicative. Scripts that reproduce each table are named under it.

## 1. Task-step timeline (PS3)

### 1.1 Held-out clips (never used to choose anything)

`mytronlabs-000060-016` (stone plant, 0–180 s, 20 labelled segments):

| Timeline | Actions | Frame acc. | F1@10 | F1@25 | F1@50 | Edit | Verb acc.* | Object acc.* | Hand acc.* | Boundary error | Label segments found |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Offline labeller (no API) | 9 | 1.8 | 6.1 | 6.1 | 0.0 | 25.0 | 66.7 | 0.0 | 0.0 | 2.03 s | 15% |
| Gemini, 30 s windows (raw) | 30 | **45.1** | **49.2** | **49.2** | **45.9** | **40.4** | 72.2 | 44.4 | **66.7** | **0.80 s** | **90%** |
| **Final pipeline** (Gemini + Stage 7 verification + Stage 8 snapping) | 30 | 43.5 | 45.9 | 45.9 | 42.6 | 38.7 | 72.2 | 44.4 | 55.6 | 0.83 s | 90% |

`MTL-SKL00089-007` (LED-light bench, 0–60 s, 14 labelled segments):

| Timeline | Actions | Frame acc. | F1@10 | F1@25 | F1@50 | Edit | Verb acc.* | Object acc.* | Hand acc.* | Boundary error | Label segments found |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Offline labeller | 39 | 2.9 | 11.1 | 11.1 | 0.0 | 11.2 | 42.9 | 0.0 | 42.9 | 0.84 s | 50% |
| Gemini, 30 s windows (raw) | 24 | 3.0 | 9.0 | 9.0 | 9.0 | 16.6 | 14.3 | 14.3 | **78.6** | 0.91 s | **100%** |
| **Final pipeline** | 24 | 3.9 | 9.0 | 9.0 | 9.0 | 16.6 | 14.3 | 21.4 | 78.6 | 1.15 s | 100% |

\* on labelled segments matched in time (IoU ≥ 0.25). F1@k: segment F1 at IoU threshold k% (same verb required).
Reproduce: `python scripts/ablate_steps.py --runs outputs_gemini --labels labels`.

**Reading the bench result honestly.** Gemini's segment *times* agree closely with the labels (e.g. 11.2–17.7 s
vs 11.5–17.5 s; 45.5–52.5 vs 45.5–51.5 s) and every labelled segment was found, but the *words* differ:
Gemini calls the part a "battery pack" with "green wrap" and the verb "fasten"; the draft labels say
"circuit board" and "attach". From the frames, Gemini's reading is plausible and the draft label may be
wrong. We did **not** change held-out labels after seeing predictions; the team check will settle it.

### 1.2 How the defaults were chosen (dev clip only)

Dev clip `mytronlabs-000060-010`, 0–60 s, 14 segments, labelled before any prediction was looked at:

| Variant | F1@10 | F1@25 | F1@50 | Edit | Hand acc. | Frame acc. |
|---|---|---|---|---|---|---|
| Offline labeller | 26.1 | 8.7 | 0.0 | 18.2 | 66.7 | 6.5 |
| Gemini pass 1 (30 s windows) | 35.3 | 17.6 | 11.8 | 33.2 | 41.7 | 22.6 |
| + pass 2 (re-ask uncertain windows at 4 fps) | 16.7 | 16.7 | 11.1 | 28.2 | 25.0 | 21.4 |
| pass 1 + verification | 35.3 | 17.6 | 11.8 | 34.8 | 41.7 | 31.9 |
| pass 1 + verification + snapping (**default**) | 35.3 | 17.6 | 11.8 | 34.8 | 41.7 | 31.9 |

Decisions: pass 2 off (it hurt), verification and snapping on (never worse on dev) — DECISIONS D15.
One minute and 14 segments is a small sample; on the held-out stone-plant clip the same verification
costs ~3 F1 points, which we report rather than tune away.

### 1.3 What we learned

- **Gemini on short windows is the strongest labeller** on both held-out clips and the dev clip. Whole
  3-minute clips were refused with 503 on every model for over an hour; 30 s windows answer in seconds.
- **The offline labeller depends on seeing the hands.** On the walking stone-plant clip MediaPipe finds a
  hand in ~12–15% of frames, and the offline labeller finds only 15% of the labelled segments.
- **Refinement stages need their own measurements.** The first version of boundary snapping made F1@50
  drop from 45.9 to 26.2 (a missing hand was read as a still hand); fixed, it is now neutral (ERRORS #28).

## 2. Hand tracking (30/100 of the judging, every PS)

### 2.1 Label-free numbers (first 60 s, MediaPipe, every 2nd frame at 960 px)

| Clip | Setting | Processed frames with ≥1 hand | with both hands | Hand-model speed |
|---|---|---|---|---|
| MTL-SKL00089-005 | bench, seated | 92.0% | 68.6% | 34.8 fps |
| mytronlabs-000003-002 | grinding machine, seated | 90.4% | 45.4% | 34.9 fps |
| MTL-SGF00106-036 | woodwork, two people | 56.3% | 3.7% | 34.7 fps |
| mytronlabs-000060-016 | stone plant, walking | 11.8% | 1.1% | 50.4 fps |

Reproduce: `python scripts/measure_hands.py --clips <clips> --seconds 60`. Left/right labels: MediaPipe's raw
labels are correct for this non-mirrored footage; the PDF's suggested swap made every label wrong (D4).

### 2.2 Ideas we measured and switched off

| Change | Effect on frames with a hand | Speed | What the visual check showed | Kept? |
|---|---|---|---|---|
| Retry missed frames on a contrast-enhanced copy | +1.0 / 0.0 / −0.7 / 0.0 pts | roughly halves | 7 of 12 recovered "hands" on the stone plant were pink rollers; a foot and forearms elsewhere | No |
| Drop hands pointing down at the camera (coworkers) | −0.8 pts on the coworker clip | — | all 12 drops there were the **wearer's own** hand | No |

### 2.3 Accuracy on judged frames

**Pending.** `python scripts/make_judging_sheets.py --clips MTL-SKL00089-007 mytronlabs-000060-016 --per-clip 50`
creates sheets and a CSV for ~100 frames (half of them hard frames); a teammate fills in hands found,
false hands, side errors and bad keypoints, then `egovision eval` reports recall / precision / side accuracy.

## 3. Privacy (Stage 1)

| Clip (full length) | Regions pixelated | Avg. pixelated area when blurring |
|---|---|---|
| MTL-SGF00106-021 | 1,554 heads · 47 faces · 34 signs · 5 logos | 5.6% |
| MTL-SGF00106-023 | 442 heads · 8 faces · 8 signs · 2 logos | 2.9% |
| mytronlabs-000060-006 | 880 heads · 3 faces · 838 signs · 96 logos | 7.8% |
| MTL-SKL00089-005 | 370 heads · 24 faces · 11 signs · 1 logo | 3.4% |

Blurring only faces and company names (tools and parts stay sharp) followed a user decision (D11). Known
misses: part of a sign at a frame edge; the wearer's own knees when squatting (ERRORS #21, #22). Gemini
file storage was checked after every run: 0 files left.

## 4. Speed (laptop, per 60 s of video, `MTL-SKL00089-007`)

| Stage | Time | Throughput |
|---|---|---|
| Decode (PyAV, every 2nd frame, 960 px) | 5.9 s | 154 kept frames/s |
| Hand model (MediaPipe, CPU) | 25.8 s | 34.8 frames/s |
| Keyframe pass (OWLv2 + SigLIP, GPU fp16, every 0.33 s) | 45.3 s | 7.7 keyframes/s |
| Privacy write (sanitized copy) | 22.7 s | |
| Steps (Gemini, 2 windows, network-bound) | 85.6 s | |

## 5. Failure cases (described in text; frames stay on the private Drive)

1. **Hands far away or out of view** (stone plant): few hand detections → weak contacts, weak offline labels,
   and Stage 7 marks many correct Gemini actions "rejected" for lack of hand evidence.
2. **Fine-grained bench work** (SKL): verbs and objects are ambiguous at this level of detail; labels and
   model disagree on words more than on timing.
3. **Coworker hands** can be tracked as the wearer's; the rule we tried to filter them was wrong (§2.2).
4. **False hands on skin-coloured objects** (pink rollers) appear when thresholds are lowered (§2.2).
5. **Gemini availability**: whole clips 503; short windows needed; a hung request once took 27 minutes
   before the timeout fix (ERRORS #24–26, #29).

Full list with fixes: [ERRORS_AND_FIXES.md](ERRORS_AND_FIXES.md).
