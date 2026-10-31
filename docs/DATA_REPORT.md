# Data report

Generated in Phase 0 from the files in `data/raw/` (git-ignored). Per the data-use rules
(SPEC.md §5), **this report contains no frames or screenshots**: the contact sheets it refers to
live in the git-ignored `artifacts/data_inspect/` folder on the team laptop only.

## 1. What we received

- Source: shared Drive folder `Hackathon_vidoes/Sample/`, 40 `.mp4` files, 4.1 GB total.
  Each file's size was checked against the server's `Content-Length`; four were
  truncated on the first download and re-fetched.
- **No label or annotation files**, no `README.txt`, no `clips_480p/`, no demo-clip marking
  (the PDF describes all three, see SPEC.md §6).
- Every file is technically identical in format:

| property | value (all 40 clips) |
|---|---|
| resolution | 1920 × 1080 |
| frame rate | 30.000 fps (constant) |
| duration | 180.00 s = 5,400 frames |
| video codec | HEVC (H.265), ≈ 4.8 Mbit/s |
| audio | an audio track (`soun` handler) is present in the checked files; content not yet inspected |
| container metadata | no camera make/model tags; files appear re-encoded before distribution |
| integrity | first and last frames decode in every file |

- **Lens:** strong fisheye / ultra-wide (barrel distortion clearly visible; straight benches and
  conveyors bend into arcs). Horizontal FOV estimated at 150°+. No intrinsics are available.
- **Mounting:** head-mounted and pitched steeply down. The wearer's own torso/lap occupies the
  bottom 15–30% of most frames, and arms enter from the bottom edge.

### Decode throughput on our laptop (i5-12450H, 8C/12T; RTX 3050 Laptop 4 GB)

Measured with OpenCV 4.11 (FFmpeg backend, CPU decode) on `MTL-SKL00089-005.mp4`:

| operation | speed |
|---|---|
| sequential decode only (`grab`) | 163 fps |
| sequential decode + BGR convert + resize to 640 px | 57 fps |
| random seek + read | ≈ 280 ms per seek |

So decode is not free: a full 1080p decode of one 3-minute clip takes about 33 s of CPU just to
grab frames. Consequences for the design: decode once and cache downscaled frames, `grab()`
skipped frames and only `retrieve()` the ones we keep, and avoid random seeks in the hot path.

## 2. Recording sources

File names follow `<source>-<chunk>.mp4`. Chunks are consecutive 3-minute pieces of one long
recording, so neighbouring numbers are continuous footage and actions can cross a chunk boundary.

| source | clips (chunk numbers) | continuous runs | setting | worker posture |
|---|---|---|---|---|
| `MTL-SGF00106` | 021, 023, 027, 028, 029, 031, 032, 035, 036, 040 | 027–029, 031–032, 035–036 | Woodworking shop: building curved plywood furniture frames (sofa/chair carcasses) on a sawdust-covered floor | Squatting/kneeling on the floor, moving around the frame |
| `MTL-SKL00089` | 005, 006, 007, 008, 011, 012, 013, 014, 022, 023 | 005–008, 011–014, 022–023 | Electronics/lighting assembly bench: LED tube lights (aluminium profiles, white diffuser strips, green LED/driver PCBs, end caps) | Seated at a cluttered bench |
| `mytronlabs-000003` | 002, 003, 004, 005, 006, 007, 008, 012, 013, 015 | 002–008, 012–013 | Metal workshop: finishing small metal parts at a round grinding/polishing machine, using a hand-held clamp/jig with a cream handle | Seated at the machine, hands very close to the camera |
| `mytronlabs-000060` | 006, 008, 009, 010, 011, 012, 013, 014, 015, 016 | 008–016 | Stone plant: granite slabs on roller conveyors/racks; wiping, pushing, tilting and clamping slabs | Standing and walking a lot |

Within a source the same worker and task family repeat across clips. This is useful for
building a per-source step template and for the PS3 bonus "compare … same task".

## 3. Per-clip observations

The contents column comes from looking at 8 frames per clip (t = 5, 25, 50, 75, 100, 125, 150, 175 s).
Quality columns come from 90 frames per clip (every 2 s, downscaled to 640 px):
- `blur%`: share of sampled frames whose Laplacian variance is below 35% of that clip's median. It is a relative measure of camera-motion blur. Absolute values are not comparable across sources because texture (sawdust, granite) dominates them.
- `overexp`: % of pixels > 245.
- `dark`: % of pixels < 20.

These are whole-frame metrics: they catch camera shake, **not** fast hand motion over a static
background (seen e.g. in 000003-012/013). Hand-level blur will be measured in Phase 1.

### MTL-SGF00106 — furniture frame workshop

Common to all clips: bare hands (wristwatch/bracelet), **bare feet often in view** while
squatting (a false-positive risk for hand detectors), sawdust floor, power tools: red cordless
drill/nail gun, blue-white electric planer/sander, green jigsaw. Coworkers walk through the scene.

| clip | blur% | overexp | dark | what is visibly happening |
|---|---|---|---|---|
| 021 | 10.0 | 0.43 | 0.28 | Walking around the shop; coworkers' faces in view; folding/holding hands; later kneels at a curved frame and handles a small part next to a seated coworker |
| 023 | 10.0 | 0.13 | 0.22 | Squatting on the floor shaping plywood pieces with a hand-held electric planer/sander; picking up and placing offcuts |
| 027 | 6.7 | 0.29 | 0.20 | Inside a large curved frame, fitting slats; using the red power tool (nailing/drilling); planing a part on the floor |
| 028 | 13.3 | 0.61 | 0.19 | Working inside/around the curved frame; fitting and fastening pieces; motion-blurred handling of a plywood part |
| 029 | 13.3 | 0.49 | 0.17 | Red power tool on the frame; holding/aligning frame parts; marking/cutting templates on the floor |
| 031 | 20.0 | 0.47 | 0.24 | Planing/sanding plywood templates on the floor with the blue-white tool; walking back to the frame (highest blur in the set) |
| 032 | 11.1 | 0.71 | 0.20 | Carrying and turning frame parts (strong blur); planing a plywood template |
| 035 | 14.4 | 0.47 | 0.27 | Sorting plywood strips; planing a strip; coworker's face in view at ~150 s |
| 036 | 10.0 | 1.27 | 0.52 | **Two people** working together on a frame: a coworker's hands and body are close to the wearer's hands; red drill in use |
| 040 | 7.8 | 1.56 | 0.15 | Holding a curved frame while grinding/sanding it with a disc tool (blue disc); coworker in view |

### MTL-SKL00089 — LED tube-light assembly bench

Common: seated woman worker, bare hands with red bangles, a bright yellow-green dress fills the
lower frame (skin-like and high-contrast clothing next to the hands), cluttered bench with
tape rolls, a white bowl, red tape dispenser, boxes and crates. Brightest and sharpest source,
but with large dark regions under the bench (5–7% dark pixels).

| clip | blur% | overexp | dark | what is visibly happening |
|---|---|---|---|---|
| 005 | 0.0 | 1.42 | 7.23 | Picking small green parts (PCBs/sleeves) from a pile and working on them with both hands; reaching to the box on the left |
| 006 | 0.0 | 1.03 | 6.75 | Repetitive handling of small green parts: holding with one hand, working with the other (wiring/sleeving), placing back |
| 007 | 0.0 | 0.98 | 6.40 | Same repetitive green-part task; both hands at the bench centre |
| 008 | 0.0 | 1.29 | 6.88 | Same task; another worker appears beside the bench at ~150 s; looks up/around at the end |
| 011 | 0.0 | 1.46 | 5.55 | Handling long white diffuser strips / aluminium profiles: sliding strips into profiles, holding long parts diagonally |
| 012 | 0.0 | 1.45 | 5.44 | Same long-profile task; **writing on a paper sheet** at ~25 s (document in view); inserting strip into profile |
| 013 | 1.1 | 1.56 | 5.25 | Long white strips/profiles: threading, bending, inspecting; gathering a bundle of profiles at the end |
| 014 | 1.1 | 1.23 | 5.52 | Unpacking from a plastic bag/box at the start; then long-profile assembly at the bench |
| 022 | 7.8 | 0.81 | 1.66 | Fetching a box of grey end caps (turning around, blurred); fitting end caps onto aluminium profiles |
| 023 | 5.6 | 0.76 | 1.41 | Fitting end caps onto aluminium profiles, repeated; a coworker's face in view at ~150 s |

### mytronlabs-000003 — small metal parts at a grinding machine

Common: seated man, bare hands (ring, wrist band), forearms large in the frame, hands often
**very close to the lens and cut off by the frame edges**, frequent two-hand manipulation of
small shiny parts, a cream-handled clamp/jig, a centre grinding wheel, a green paint/oil tin,
a smartphone lying on the machine, a coworker sitting opposite (face often visible).

| clip | blur% | overexp | dark | what is visibly happening |
|---|---|---|---|---|
| 002 | 1.1 | 0.84 | 0.77 | Picking small parts, filing/finishing them on the machine top; clamping a part in the cream-handled jig; inspecting |
| 003 | 1.1 | 0.52 | 0.48 | Holding parts against the grinding wheel; placing finished parts on the table; looking at coworker |
| 004 | 3.3 | 0.33 | 0.62 | Forearms resting/working on the machine top; reaching to the wheel; two-hand part manipulation at the end |
| 005 | 0.0 | 0.28 | 0.55 | Close-up two-hand work on a metal assembly (lock-like body with spring/pin); fitting and adjusting |
| 006 | 0.0 | 0.40 | 0.66 | Same assembly with the cream jig: inserting, adjusting, inspecting; placing on table |
| 007 | 1.1 | 0.62 | 0.64 | Grinding at the wheel; jig work; a large out-of-focus hand covering most of the frame at ~150 s |
| 008 | 0.0 | 0.73 | 0.70 | Jig and part handling; grinding at the wheel; worker leans back at the end (lap and hands only) |
| 012 | 1.1 | 0.74 | 0.78 | Jig assembly with both hands; a strongly blurred turn of the head at ~50 s |
| 013 | 0.0 | 0.93 | 0.85 | Jig assembly, close-up; fast hand motion (local blur) |
| 015 | 2.2 | 0.40 | 0.67 | Picking parts; assembling with jig; holding an orange plastic comb-like part at the end |

### mytronlabs-000060 — granite slab plant

Common: standing/walking worker, bare hands and forearms (pink shirt sleeves), hands often
**small, at the bottom edge, or out of view** for long stretches; large overexposed windows
(4–5% blown pixels, the most of any source); red-white roller conveyor; yellow slab clamp; a
yellow hand-held tool; cloth/rag for wiping; **company signage readable** in 006; other workers.

| clip | blur% | overexp | dark | what is visibly happening |
|---|---|---|---|---|
| 006 | 0.0 | 4.59 | 0.21 | Walking around the plant; looking at slabs and the conveyor; hands mostly out of view |
| 008 | 0.0 | 3.63 | 0.21 | Slab on conveyor; wiping slab with a cloth; walking; another worker in view |
| 009 | 0.0 | 3.67 | 0.14 | Pushing a hand truck; wiping; tilting/holding a slab upright with both hands; operating the clamp |
| 010 | 1.1 | 3.76 | 0.13 | Wiping a slab on the conveyor with a cloth (repeated); tilting a slab; yellow clamp |
| 011 | 0.0 | 4.40 | 0.15 | Operating a yellow hand tool next to a slab; tilting/holding a slab; walking along conveyor |
| 012 | 1.1 | 4.28 | 0.17 | Holding slab edges; pushing slabs; walking |
| 013 | 1.1 | 3.81 | 0.14 | Wiping slab; checking slab stacks; clamp in view |
| 014 | 1.1 | 4.46 | 0.15 | Walking; pushing slab on rollers; touching/inspecting slab edge |
| 015 | 0.0 | 4.54 | 0.16 | Wiping slab on conveyor; holding a long metal bar; walking |
| 016 | 0.0 | 4.62 | 0.17 | Wiping a slab with a cloth with one hand while the other rests on it; walking |

## 4. Labels

**There are no labels.** Nothing to parse, so there are no class/step counts. In place of a label
summary, the action vocabulary we expect from what is visible:

| source | likely verbs | likely objects |
|---|---|---|
| SGF00106 | pick, place, hold, align, fasten (nail/drill), plane/sand, cut, carry, inspect | plywood frame, plywood panel/template, wooden strip, drill/nail gun, planer/sander, jigsaw |
| SKL00089 | pick, place, hold, insert, attach (end cap), wire/sleeve, tape, write, inspect, unpack | aluminium profile, diffuser strip, LED/driver PCB, end cap, tape, paper sheet, box/bag |
| 000003 | pick, place, hold, grind/polish, clamp/adjust (jig), assemble, inspect | small metal part, jig/clamp, grinding wheel, file, comb-like plastic part |
| 000060 | wipe, push, tilt/lift, hold, clamp, carry, walk, inspect | granite slab, cloth/rag, roller conveyor, slab clamp, hand tool, hand truck, metal bar |

This feeds the fixed verb list in SPEC.md (A2).

## 5. Implications for the system

1. **Fisheye distortion everywhere.** Hands near the frame edges are stretched and bent. Hand
   models trained on normal-lens images may degrade there. Test whether a fixed approximate
   undistortion (or running the detector on a centre crop plus edge crops) helps.
2. **Distractor hands and feet.** Coworkers' hands (SGF-036, 000003, SKL-008, 000060) and the
   wearer's bare feet (SGF) appear regularly. PS1 asks for the *worker's* hands, so we need an
   "own hand" filter. The egocentric prior helps: own forearms enter from the bottom edge.
3. **Hands cut off by the frame edge or very close to the lens** (000003) and **small or missing
   hands** (000060). Both stress detection recall and need recovery and interpolation across gaps.
4. **No gloves are visible in any sampled frame.** The PDF warns about gloves, but our footage has
   bare hands with jewellery (watches, bangles, rings). We should still say this in the
   presentation, and check whether a demo clip from organisers adds gloves.
5. **Skin-coloured surroundings.** Plywood and wood (SGF), pink/red granite (000060) and orange
   parts (000003) are close to skin tones, so naive skin-colour heuristics will fail.
6. **Blur** is mainly camera motion in SGF (up to 20% of sampled frames) and fast hand motion in
   000003. Temporal tracking and interpolation matter more than single-frame accuracy there.
7. **Privacy for slides and public material:** coworkers' faces (SGF-021/035/040, SKL-023,
   000003-*), a paper document (SKL-012), phone screens (000003) and readable company signage
   (000060-006). Use these only on the private Drive, and blur them if any image goes anywhere else.
8. **Continuous runs** of chunks let us test step templates across chunk boundaries and compare
   repeated cycles of the same task (e.g. SKL 005–008 green-part cycles, 000060 slab-wiping cycles).

## 6. Proposed splits (to confirm before labelling)

- **Hand-tracking validation set:** about 120 frames, 30 per source, stratified so at least half are
  hard (blur, edge-truncated, overlap, coworker hands, re-entry). Each frame is labelled with own-hand
  boxes plus side; keypoints are spot-checked visually, not fully labelled.
- **Action validation set (held out, never used for prompt or threshold tuning):** one clip per
  source, `MTL-SGF00106-040`, `MTL-SKL00089-007`, `mytronlabs-000003-013`, `mytronlabs-000060-016`,
  fully labelled for at least two of them and partially for the other two, depending on team time.
- **Development clips** (free to tune on): all others.
- **Provisional demo candidates** until organisers mark demo clips: one per source, chosen for
  hard hand cases, e.g. SGF-036 (two people + blur), SKL-022 (blur + end caps), 000003-005
  (close two-hand manipulation), 000060-010 (wiping, small hands).
