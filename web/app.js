// FORGE dashboard: plain ES module, no build step. Talks to egovision.api (FastAPI).
const $ = (s) => document.querySelector(s);
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
};
const HAND_EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],
  [14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];
const SIDE = ["#50dc50", "#ff9600"];

const state = { clips: [], clip: null, res: null, hands: [], keys: [], keyFrames: [], colors: {},
  layers: { hands: true, keypoints: true, objects: false, held: true }, jobs: {} };

// ---------- helpers ----------
function verbColors(verbs) {  // same golden-ratio hues as render.py
  const out = {};
  verbs.forEach((v, i) => {
    const h = (i * 0.618) % 1, s = 0.65, val = 0.95, k = (n) => (n + h * 6) % 6;
    const f = (n) => val - val * s * Math.max(0, Math.min(k(n), 4 - k(n), 1));
    out[v] = `rgb(${Math.round(f(5) * 255)},${Math.round(f(3) * 255)},${Math.round(f(1) * 255)})`;
  });
  out.other = "rgb(150,150,150)";
  return out;
}
const fmt = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function lanes(a) { return a.hand === "left" ? ["left"] : a.hand === "right" ? ["right"] : ["left", "right"]; }

// ---------- clips ----------
async function loadClips() {
  state.clips = await api("/api/clips");
  renderClips();
}
function renderClips() {
  const q = $("#clipFilter").value.toLowerCase();
  $("#clipList").innerHTML = state.clips.filter((c) => c.name.toLowerCase().includes(q)).map((c) => `
    <li data-clip="${esc(c.name)}" class="${c.name === state.clip ? "sel" : ""}">
      <span>${esc(c.name)}${c.source === "upload" ? " ⬆" : ""}</span>
      <span class="dots" title="blue = results, green = sanitized copy, amber = labels">
        <i class="dot ${c.has_results ? "res" : ""}"></i><i class="dot ${c.has_sanitized ? "san" : ""}"></i>
        ${c.has_labels ? '<i class="dot lab"></i>' : ""}</span>
    </li>`).join("") || '<li class="note">No clips in data/raw — upload one.</li>';
}
$("#clipList").addEventListener("click", (e) => {
  const li = e.target.closest("li[data-clip]");
  if (li) selectClip(li.dataset.clip);
});
$("#clipFilter").addEventListener("input", renderClips);

async function selectClip(name) {
  state.clip = name;
  $("#runBtn").disabled = false;
  $("#clipTitle").textContent = name;
  renderClips();
  const c = state.clips.find((x) => x.name === name);
  if (!c?.has_results) { clearResults("No results yet — press ▶ Run."); return; }
  try {
    const [res, hands, keys] = await Promise.all([api(`/api/results/${name}`), api(`/api/results/${name}/hands`),
      api(`/api/results/${name}/keyframes`)]);
    Object.assign(state, { res, hands, keys, keyFrames: keys.map((k) => k.frame), colors: verbColors(res.verbs) });
    showResults();
  } catch (err) { clearResults(`Could not load results: ${err.message}`); }
}

function clearResults(msg) {
  state.res = null;
  const v = $("#video"); v.removeAttribute("src"); v.load();
  $("#emptyMsg").textContent = msg; $("#emptyMsg").style.display = "grid";
  for (const id of ["stats", "legend", "tab-actions", "tab-review", "tab-analytics", "tab-contacts", "tab-tracking", "tab-privacy", "tab-eval", "tab-downloads"]) $("#" + id).innerHTML = "";
  drawTimeline();
}

// ---------- results ----------
function showResults() {
  const { res } = state;
  const v = $("#video");
  if (res.media.sanitized) {
    v.src = res.media.sanitized; $("#emptyMsg").style.display = "none";
  } else {
    v.removeAttribute("src"); v.load();
    $("#emptyMsg").textContent = "No sanitized copy for this clip (privacy was off). Originals are never streamed.";
    $("#emptyMsg").style.display = "grid";
  }
  const st = res.stats, tm = res.timings;
  const card = (val, label) => `<div class="stat"><b>${val ?? "–"}</b><span>${label}</span></div>`;
  $("#stats").innerHTML = [
    card(`${st.all_frames_with_hand_pct ?? "–"}%`, "frames with a hand"),
    card(`${st.processed_with_two_hands_pct ?? "–"}%`, "both hands (processed)"),
    card(st.track_ids, "hand track IDs"),
    card(res.actions.length, "actions"),
    card(res.contacts.length, "contact segments"),
    card(`${tm.hands_fps ?? "–"} fps`, "hand model (CPU)"),
    card(`${tm.realtime_factor ?? "–"}×`, "real-time factor"),
    card(st.device, "device"),
  ].join("");
  const used = [...new Set(res.actions.map((a) => a.verb))];
  $("#legend").innerHTML = used.map((v) => `<span><i style="background:${state.colors[v]}"></i>${esc(v)}</span>`).join("");
  renderActions(); renderContacts(); renderTracking(); renderPrivacy(); renderEval(); renderDownloads();
  renderReview(); renderAnalytics();
  drawTimeline();
}

function renderActions() {
  const rows = state.res.actions.map((a, i) => `<tr data-t="${a.start_s}" data-i="${i}">
    <td>${fmt(a.start_s)}</td><td>${fmt(a.end_s)}</td>
    <td><span class="pill" style="background:${state.colors[a.verb] || "#999"}">${esc(a.verb)}</span></td>
    <td>${esc(a.object)}</td><td>${esc(a.hand)}</td><td>${statusPill(a.status)}<div class="flags">${esc((a.flags || []).join(", "))}</div></td>
    <td>${a.confidence ?? "–"}</td><td>${esc(a.source)}</td></tr>`).join("");
  $("#tab-actions").innerHTML = `<p class="note">Click a row to jump to it. PS3 file: actions.csv (clip, start_s, end_s, verb, object, hand).
    Status comes from hand-grounded verification (Stage 7).</p>
    <table><thead><tr><th>start</th><th>end</th><th>verb</th><th>object</th><th>hand</th><th>status</th><th>conf.</th><th>source</th></tr></thead>
    <tbody>${rows || '<tr><td colspan="8">No actions.</td></tr>'}</tbody></table>`;
}
const statusPill = (st) => st ? `<span class="st st-${esc(st)}">${esc(st)}</span>` : "–";

// ---------- Stage 9 review ----------
async function renderReview() {
  const el = $("#tab-review");
  let rv;
  try { rv = await api(`/api/results/${state.clip}/review`); } catch (e) { el.innerHTML = `<p class="note">${esc(e.message)}</p>`; return; }
  const who = localStorage.getItem("reviewer") || "";
  const rows = rv.queue.map((a) => `<tr data-t="${a.start_s}" data-id="${esc(a.id)}">
    <td>${a.review ? statusPill(a.review.status) : statusPill(a.status)}<div class="flags">${esc((a.flags || []).join(", "))}</div></td>
    <td>${fmt(a.start_s)}–${fmt(a.end_s)}</td><td>${esc(a.verb)}</td><td>${esc(a.object)}</td><td>${esc(a.hand)}</td>
    <td class="rv"><button data-op="approve" title="Approve">✓</button><button data-op="edit" title="Edit">✎</button>
      <button data-op="split" title="Split at playhead">✂</button><button data-op="merge" title="Merge with next">⊕</button>
      <button data-op="reject" title="Reject">✗</button></td></tr>`).join("");
  const tp = rv.trim_plan;
  el.innerHTML = `<div class="rvhead"><label>Reviewer <input id="rvName" value="${esc(who)}" placeholder="your name"></label>
      <span class="note">${rv.progress.reviewed}/${rv.progress.total} reviewed · most uncertain first · edits are stored in review.json, model output is never overwritten</span></div>
    <table><thead><tr><th>status</th><th>time</th><th>verb</th><th>object</th><th>hand</th><th>decide</th></tr></thead>
    <tbody>${rows || '<tr><td colspan="6">No actions.</td></tr>'}</tbody></table>
    <h2 style="margin-top:16px">Trim plan (Stage 10)</h2>
    <p class="note">Keeps every non-rejected action ± padding at full rate (${tp.kept_full_rate_pct}% of the clip); the rest is kept at 1 fps so timestamps stay valid. Runs on the sanitized copy only.</p>
    <button id="approveTrim" ${tp.approved ? "disabled" : ""}>${tp.approved ? "✓ Plan approved" : "Approve trim plan"}</button>
    <button id="runTrim" class="primary" style="width:auto" ${tp.approved ? "" : "disabled"}>Run trim + compress</button>
    ${state.res.trim_map ? `<p class="note">Last trim: ${esc(JSON.stringify(state.res.trim_map.size_mb))} MB</p>` : ""}`;
}
async function decide(body) {
  const by = ($("#rvName")?.value || "").trim() || "reviewer";
  localStorage.setItem("reviewer", by);
  await api(`/api/results/${state.clip}/review`, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...body, by }) });
  await renderReview();
}
$("#tab-review").addEventListener("click", async (e) => {
  const b = e.target.closest("button");
  try {
    if (b?.id === "approveTrim") return await decide({ op: "approve_trim" });
    if (b?.id === "runTrim") { await api(`/api/results/${state.clip}/trim`, { method: "POST" }); return pollJobs(); }
    const tr = e.target.closest("tr[data-id]");
    if (!tr) return;
    if (!b) return seek(parseFloat(tr.dataset.t));
    const id = tr.dataset.id, op = b.dataset.op;
    if (op === "edit") {
      const verb = prompt(`verb (one of: ${state.res.verbs.join(", ")})`); if (verb === null) return;
      const object = prompt("object"); if (object === null) return;
      const hand = prompt("hand: left | right | both | unknown"); if (hand === null) return;
      const fields = {};
      if (verb.trim()) fields.verb = verb.trim(); if (object.trim()) fields.object = object.trim(); if (hand.trim()) fields.hand = hand.trim();
      return await decide({ action_id: id, op, fields });
    }
    if (op === "split") return await decide({ action_id: id, op, at_s: $("#video").currentTime });
    await decide({ action_id: id, op });
  } catch (err) { alert(`Review failed: ${err.message}`); }
});

// ---------- Stage 11 analytics (+ Stage 0 QC, Stage 8 step check) ----------
function bars(obj, scale = 1, unit = "") {
  const max = Math.max(...Object.values(obj).map(Number), 1e-9);
  return `<div class="bars">${Object.entries(obj).map(([k, v]) => `<div><span>${esc(k)}</span>
    <i style="width:${(100 * v) / max}%;background:${state.colors[k] || "var(--accent)"}"></i><span>${(v * scale).toFixed(scale === 100 ? 0 : 1)}${unit}</span></div>`).join("")}</div>`;
}
function renderAnalytics() {
  const a = state.res.analytics, qc = state.res.qc, sc = state.res.steps_check;
  if (!a) { $("#tab-analytics").innerHTML = "<p class='note'>No analytics for this run (re-run the pipeline).</p>"; return; }
  const verbs = Object.fromEntries(Object.entries(a.time_per_verb_s).map(([v, d]) => [v, d.total]));
  let html = `<p class="note">Positions are camera-relative (the camera is on the worker's head).</p>
    <h2>Time per verb (s)</h2>${bars(verbs)}
    <div class="kv" style="margin-top:12px"><div>busy share L / R</div><div>${Math.round(a.busy_share.left * 100)}% / ${Math.round(a.busy_share.right * 100)}%</div>
    <div>two-handed share</div><div>${Math.round(a.two_handed_share * 100)}%</div>`;
  if (a.idle) html += `<div>left idle (visible, no action / not visible)</div><div>${Math.round(a.idle.left.visible_no_action_share * 100)}% / ${Math.round(a.idle.left.not_visible_share * 100)}%</div>
    <div>right idle (visible, no action / not visible)</div><div>${Math.round(a.idle.right.visible_no_action_share * 100)}% / ${Math.round(a.idle.right.not_visible_share * 100)}%</div>`;
  if (a.cycles) html += `<div>most repeated step</div><div>${esc(a.cycles.step)} × ${a.cycles.occurrences}, median cycle ${a.cycles.median_cycle_s ?? "–"} s${a.cycles.slow_cycles?.length ? `, slow at ${a.cycles.slow_cycles.join(", ")} s` : ""}</div>`;
  if (qc) html += `<div>QC: blurry / lens covered</div><div>${qc.blurry_pct}% / ${qc.covered_pct}% of processed frames</div>`;
  if (sc) html += `<div>step template (Stage 8)</div><div>completeness ${sc.completeness ?? "–"} · missing ${esc(sc.missing.join(", ") || "none")} · out of order ${esc(sc.out_of_order.join(", ") || "none")} · repeated ${esc(sc.repeated.join(", ") || "none")}</div>`;
  html += "</div>";
  if (a.zone_share) html += `<h2 style="margin-top:12px">Work zones, right hand</h2>${bars(a.zone_share.right, 100, "%")}
    <h2>Work zones, left hand</h2>${bars(a.zone_share.left, 100, "%")}`;
  $("#tab-analytics").innerHTML = html;
}

function renderContacts() {
  const rows = state.res.contacts.map((c) => `<tr data-t="${c.t_start}"><td>${fmt(c.t_start)}</td><td>${fmt(c.t_end)}</td>
    <td>${esc(c.hand)}</td><td>${esc(c.state)}</td><td>${esc(c.object_name)}</td></tr>`).join("");
  $("#tab-contacts").innerHTML = `<p class="note">Grab → release timeline (PS2 format). A contact = ≥2 fingertips inside the padded object box.</p>
    <table><thead><tr><th>start</th><th>end</th><th>hand</th><th>state</th><th>object</th></tr></thead>
    <tbody>${rows || '<tr><td colspan="5">No contacts.</td></tr>'}</tbody></table>`;
}
function kv(obj) {
  return `<div class="kv">${Object.entries(obj).map(([k, v]) => `<div>${esc(k)}</div><div>${esc(typeof v === "object" ? JSON.stringify(v) : v)}</div>`).join("")}</div>`;
}
function renderTracking() {
  const s = state.res.stats;
  $("#tab-tracking").innerHTML = `<p class="note">Label-free numbers from this run. Accuracy on judged frames is in docs/RESULTS.md.</p>` + kv({
    "processed frames": s.processed_frames, "processed frames with a hand": `${s.processed_with_hand_pct}%`,
    "all frames with a hand (after interpolation)": `${s.all_frames_with_hand_pct}%`,
    "left / right detections": `${s.left_detections} / ${s.right_detections}`, "track IDs": s.track_ids,
    "mean hand score": s.mean_score, "side matches image half (sanity)": `${s.side_matches_image_half_pct}%`,
    ...Object.fromEntries(Object.entries(state.res.timings).map(([k, v]) => [`time: ${k}`, v])),
  });
}
function renderPrivacy() {
  const p = state.res.privacy;
  $("#tab-privacy").innerHTML = p ? `<p class="note">${esc(p.note)}</p>` + kv({
    "frames pixelated": `${p.frames_blurred_pct}%`, "mean blurred area (when blurred)": `${p.mean_blurred_area_pct}%`,
    "regions by category": p.regions_by_category, "keyframes with regions": `${p.keyframes_with_regions} / ${p.keyframes}`,
    "frames where hands were kept sharp": p.frames_with_hands_protected, "encoder / width": `${p.encoder} / ${p.width}px`,
  }) + `<p class="note">Reviewer check: scrub the video; any visible face, company name or screen = a miss to report.</p>`
    : "<p class='note'>Privacy sanitization was off for this run.</p>";
}
function renderEval() {
  const e = state.res.evaluation;
  $("#tab-eval").innerHTML = e && e.steps ? `<p class="note">${esc(e.labels_note)}</p>` + kv(e.steps)
    : "<p class='note'>No labels for this clip (labels/&lt;clip&gt;.actions.csv).</p>";
}
function renderDownloads() {
  $("#tab-downloads").innerHTML = state.res.downloads.map((n) =>
    `<a class="dl ${n === "actions.csv" ? "main" : ""}" href="/api/results/${state.clip}/download/${n}">⬇ ${n}</a>`).join("");
}
for (const id of ["#tab-actions", "#tab-contacts"]) {
  $(id).addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-t]");
    if (tr) seek(parseFloat(tr.dataset.t));
  });
}
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tabs button, .tab").forEach((x) => x.classList.remove("active"));
  b.classList.add("active"); $("#tab-" + b.dataset.tab).classList.add("active");
}));

// ---------- timeline ----------
const tl = $("#timeline");
function duration() { return state.res?.video.duration_s || 1; }
function drawTimeline() {
  const dpr = window.devicePixelRatio || 1, w = tl.clientWidth, h = tl.clientHeight;
  tl.width = w * dpr; tl.height = h * dpr;
  const g = tl.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
  const top = 6, lane = (h - 22) / 2, x0 = 18, W = w - x0 - 6;
  g.fillStyle = "#8b98a5"; g.font = "11px Segoe UI";
  g.fillText("L", 4, top + lane / 2 + 4); g.fillText("R", 4, top + lane * 1.5 + 4);
  if (!state.res) return;
  const D = duration();
  for (const a of state.res.actions) {
    g.fillStyle = state.colors[a.verb] || "#999";
    for (const ln of lanes(a)) {
      const y = top + (ln === "left" ? 0 : lane);
      g.fillRect(x0 + (a.start_s / D) * W, y + 2, Math.max(2, ((a.end_s - a.start_s) / D) * W), lane - 4);
    }
  }
  g.fillStyle = "#8b98a5";
  for (let s = 0; s <= D; s += D > 120 ? 30 : 10) g.fillText(fmt(s).replace(/\.0$/, ""), x0 + (s / D) * W - 10, h - 4);
  const t = $("#video").currentTime || 0;
  g.strokeStyle = "#fff"; g.lineWidth = 2; g.beginPath(); g.moveTo(x0 + (t / D) * W, 0); g.lineTo(x0 + (t / D) * W, h - 16); g.stroke();
}
function tlHit(e) {
  const r = tl.getBoundingClientRect(), x0 = 18, W = r.width - x0 - 6;
  const t = Math.max(0, Math.min(1, (e.clientX - r.left - x0) / W)) * duration();
  const lane = e.clientY - r.top < 6 + (r.height - 22) / 2 ? "left" : "right";
  const a = state.res?.actions.find((x) => x.start_s <= t && t < x.end_s && lanes(x).includes(lane));
  return { t, a, x: e.clientX - r.left };
}
tl.addEventListener("click", (e) => { if (state.res) { const { t, a } = tlHit(e); seek(a ? a.start_s : t); } });
tl.addEventListener("mousemove", (e) => {
  const tip = $("#tooltip"); if (!state.res) return;
  const { t, a, x } = tlHit(e);
  tip.style.display = "block"; tip.style.left = `${x + 10}px`; tip.style.top = "0px";
  tip.textContent = a ? `${a.verb} ${a.object} · ${a.hand} · ${fmt(a.start_s)}–${fmt(a.end_s)}` : fmt(t);
});
tl.addEventListener("mouseleave", () => { $("#tooltip").style.display = "none"; });
window.addEventListener("resize", () => { drawTimeline(); drawOverlay(); });

// ---------- video overlay ----------
const video = $("#video"), ov = $("#overlay");
function seek(t) { video.currentTime = t; video.play().catch(() => {}); }
function keyAt(frame) {  // last keyframe at or before `frame`
  let lo = 0, hi = state.keyFrames.length - 1, best = -1;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (state.keyFrames[m] <= frame) { best = m; lo = m + 1; } else hi = m - 1; }
  return best >= 0 ? state.keys[best] : null;
}
function drawOverlay() {
  const dpr = window.devicePixelRatio || 1, w = ov.clientWidth, h = ov.clientHeight;
  if (ov.width !== w * dpr || ov.height !== h * dpr) { ov.width = w * dpr; ov.height = h * dpr; }
  const g = ov.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  if (!state.res || !video.src) return;
  const { width: VW, height: VH, fps } = state.res.video, s = Math.min(w / VW, h / VH);
  const ox = (w - VW * s) / 2, oy = (h - VH * s) / 2, X = (x) => ox + x * s, Y = (y) => oy + y * s;
  const frame = Math.min(state.hands.length - 1, Math.floor(video.currentTime * fps + 1e-3));
  const key = keyAt(frame), L = state.layers;
  g.font = "12px Segoe UI"; g.lineWidth = 2;
  if (key && L.objects) {
    g.strokeStyle = "rgba(60,200,255,.8)"; g.fillStyle = "rgba(60,200,255,.95)";
    for (const [label, score, b] of key.objects) { g.strokeRect(X(b[0]), Y(b[1]), (b[2] - b[0]) * s, (b[3] - b[1]) * s); g.fillText(`${label} ${score}`, X(b[0]) + 3, Y(b[3]) - 4); }
  }
  const held = new Map((key?.contacts || []).filter((c) => c[2] !== "none").map((c) => [c[0], c]));
  for (const hd of state.hands[frame] || []) {
    const [id, side, score, interp, x1, y1, x2, y2, ...kp] = hd, col = SIDE[side];
    g.strokeStyle = col; g.fillStyle = col; g.lineWidth = interp ? 1 : 2;
    let tag = `${side ? "R" : "L"}${id >= 0 ? id : ""} ${(score / 100).toFixed(2)}`;
    const c = held.get(id);
    if (c && L.held) {
      tag += ` · ${c[2] === "portable" ? "holds" : "touches"} ${c[3]}`;
      if (c[4]) { g.setLineDash([5, 3]); g.strokeRect(X(c[4][0]), Y(c[4][1]), (c[4][2] - c[4][0]) * s, (c[4][3] - c[4][1]) * s); g.setLineDash([]); }
    }
    if (L.hands) { g.strokeRect(X(x1), Y(y1), (x2 - x1) * s, (y2 - y1) * s); }
    if (L.keypoints) {
      g.beginPath();
      for (const [a, b] of HAND_EDGES) { g.moveTo(X(kp[2 * a]), Y(kp[2 * a + 1])); g.lineTo(X(kp[2 * b]), Y(kp[2 * b + 1])); }
      g.stroke(); g.fillStyle = "#fff";
      for (let i = 0; i < 21; i++) g.fillRect(X(kp[2 * i]) - 1.5, Y(kp[2 * i + 1]) - 1.5, 3, 3);
      g.fillStyle = col;
    }
    if (L.hands || L.held) { g.fillStyle = "rgba(0,0,0,.6)"; const tw = g.measureText(tag).width; g.fillRect(X(x1), Y(y1) - 18, tw + 8, 16); g.fillStyle = col; g.fillText(tag, X(x1) + 4, Y(y1) - 6); }
  }
  const t = video.currentTime;
  const now = state.res.actions.filter((a) => a.start_s <= t && t < a.end_s);
  $("#nowStep").innerHTML = now.map((a) => `<span style="border-left-color:${state.colors[a.verb]}">${esc(a.hand)}: <b>${esc(a.verb)}</b> ${esc(a.object)}</span>`).join("");
  document.querySelectorAll("#tab-actions tr[data-i]").forEach((tr) => {
    const a = state.res.actions[+tr.dataset.i]; tr.classList.toggle("now", a.start_s <= t && t < a.end_s);
  });
}
function loop() { drawOverlay(); drawTimeline(); }
if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
  const tick = () => { loop(); video.requestVideoFrameCallback(tick); };
  video.requestVideoFrameCallback(tick);
}
video.addEventListener("seeked", loop); video.addEventListener("loadeddata", loop); video.addEventListener("timeupdate", loop);
document.querySelectorAll("[data-layer]").forEach((cb) => cb.addEventListener("change", () => { state.layers[cb.dataset.layer] = cb.checked; loop(); }));

// ---------- run + jobs ----------
$("#runBtn").addEventListener("click", async () => {
  if (!state.clip) return;
  const body = { clip: state.clip, stride: +$("#stride").value, objects: $("#optObjects").checked,
    privacy: $("#optPrivacy").checked, max_seconds: +$("#maxSec").value || null };
  try { const { id } = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); state.jobs[id] = true; pollJobs(); }
  catch (err) { alert(`Could not start: ${err.message}`); }
});
async function pollJobs() {
  const jobs = await api("/api/jobs").catch(() => []);
  $("#jobList").innerHTML = jobs.slice(0, 6).map((j) => `<li><b>${esc(j.clip)}</b> · <span class="status-${j.status}">${j.status}</span>
    ${j.status === "running" ? ` · ${esc(j.stage)} ${Math.round(j.progress * 100)}%` : ""}${j.error ? `<div class="status-failed">${esc(j.error)}</div>` : ""}
    <div class="bar"><i style="width:${(j.status === "done" ? 1 : j.progress) * 100}%"></i></div></li>`).join("");
  const active = jobs.some((j) => j.status === "queued" || j.status === "running");
  for (const j of jobs) {
    if (j.status === "done" && state.jobs[j.id]) { delete state.jobs[j.id]; await loadClips(); if (j.clip === state.clip) selectClip(j.clip); }
  }
  if (active) setTimeout(pollJobs, 1500);
}

// ---------- upload ----------
$("#uploadInput").addEventListener("change", async (e) => {
  const f = e.target.files[0]; if (!f) return;
  const fd = new FormData(); fd.append("file", f);
  try { const { name } = await api("/api/upload", { method: "POST", body: fd }); await loadClips(); selectClip(name); }
  catch (err) { alert(`Upload failed: ${err.message}`); }
  e.target.value = "";
});

// ---------- boot ----------
api("/api/health").then((h) => { $("#health").textContent = `server ok · queue ${h.queue}`; }).catch(() => { $("#health").textContent = "server offline"; });
loadClips().then(() => { const first = state.clips.find((c) => c.has_results); if (first) selectClip(first.name); });
pollJobs();
