// FORGE Studio frontend: plain ES module, no build step.
// The browser uploads, displays and downloads. Every event, metric and report comes from the backend
// (REST for the initial state, a WebSocket for live updates); nothing is computed or stored here.
import { Hands3DViewer } from "./hands3d.js";

const $ = (s) => document.querySelector(s);
const SVGNS = "http://www.w3.org/2000/svg";
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `${r.status} ${r.statusText}`);
  return r.json();
}
const fmtT = (s) => (s == null ? "–" : `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`);
const fmtD = (s) => (s == null ? "–" : s < 60 ? `${s.toFixed(1)} s` : `${Math.floor(s / 60)} m ${String(Math.round(s % 60)).padStart(2, "0")} s`);
const fmtMB = (b) => `${(b / 1e6).toFixed(1)} MB`;
const pct = (x) => (x == null ? "–" : `${Math.round(100 * x)}%`);
const when = (t) => (t ? new Date(t * 1000).toLocaleTimeString() : "–");
const css = getComputedStyle(document.documentElement);
const PALETTE = [1, 2, 3, 4, 5, 6, 7, 8].map((i) => css.getPropertyValue(`--w${i}`).trim());
const OTHER = css.getPropertyValue("--muted").trim();
const WARN = css.getPropertyValue("--warn").trim();
const BAD = css.getPropertyValue("--bad").trim();
// Fixed slot per worker number (W1 = slot 1 ...); beyond 8 workers fold to grey instead of cycling hues.
const workerColor = (wid) => { const n = parseInt(String(wid).slice(1), 10); return n >= 1 && n <= 8 ? PALETTE[n - 1] : OTHER; };
const KEY_FIELDS = ["event_id", "worker_id", "action", "object", "tool", "start_ts", "end_ts", "duration_s", "confidence",
  "verification_status", "completion_status", "frames_seen", "sop_step", "source"];

const state = { settings: null, videos: [], vid: null, video: null, events: new Map(), workers: [], analytics: null,
  reports: [], batches: new Map(), ws: null, wsTimer: null, wsRetry: 0, file: null, localUrl: null, compare: null,
  compareIds: new Set(), renderQueued: false, hands2d: null };

const viewer3d = new Hands3DViewer({ canvas: $("#h3dCanvas"), panel: $("#h3dPanel"), message: $("#h3dMsg"), player: $("#player"),
  colors: { left: css.getPropertyValue("--left").trim(), right: css.getPropertyValue("--right").trim(),
    camera: css.getPropertyValue("--accent").trim(), ax: "#f85149", ay: "#3fb950", az: "#4cc2ff",
    grid: "#3b4654", muted: OTHER, bg: "#0b0e13" } });

// ---------------------------------------------------------------- small UI helpers
function toast(msg, err = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = `toast${err ? " err" : ""}`;
  t.style.display = "block";
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.style.display = "none"; }, err ? 9000 : 4500);
}
const tip = $("#tooltip");
function showTip(html, e) {
  tip.innerHTML = html;
  tip.style.display = "block";
  const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  const y = e.clientY + 16 + tip.offsetHeight > window.innerHeight ? e.clientY - tip.offsetHeight - 10 : e.clientY + 16;
  tip.style.left = `${x}px`; tip.style.top = `${y}px`;
}
const hideTip = () => { tip.style.display = "none"; };
function svgEl(tag, attrs = {}, parent = null) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}
function niceStep(range, target = 8) {
  const raw = range / target, p = 10 ** Math.floor(Math.log10(raw || 1));
  return [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw) || raw;
}
function seek(t) {
  const p = $("#player");
  if (!p.src) return;
  p.currentTime = t;
  p.play().catch(() => {});
}
function eventTip(e) {
  const long = (state.analytics?.long_tasks || []).find((x) => x.event_id === e.event_id);
  return `<b>${esc(e.event_id)} · ${esc(e.worker_id)} · ${esc(e.action)} ${esc(e.object)}</b>`
    + `<span>${fmtT(e.start_ts)} → ${fmtT(e.end_ts)} · ${fmtD(e.duration_s)}${e.tool ? ` · tool: ${esc(e.tool)}` : ""}</span><br>`
    + `<span>confidence ${e.confidence.toFixed(2)} · ${esc(e.verification_status)} · ${esc(e.completion_status)}</span>`
    + (long ? `<br><span style="color:${WARN}">⚠ unusually long: ${esc(long.reason)}</span>` : "")
    + (e.evidence ? `<br><span>${esc(e.evidence)}</span>` : "");
}

// ---------------------------------------------------------------- health + settings + video list
async function loadHealth() {
  try {
    const h = await api("/api/health");
    $("#healthBadge").textContent = h.analysing ? `server ok · analysing` : "server ok";
    $("#healthBadge").className = "badge ok";
    const cb = $("#aiBadge");
    cb.textContent = h.gemini_configured ? `Gemini: ${h.model}` : "Gemini: no key · offline labeller";
    cb.className = `badge ${h.gemini_configured ? "ok" : "warn"}`;
    cb.title = h.gemini_configured ? `Live analysis with ${h.model} (fallbacks: ${h.fallback_models.join(", ")}); offline FORGE labeller if Gemini fails`
      : "No GEMINI_API_KEY in .env: analyses use the offline FORGE labeller (local GPU, lower accuracy)";
  } catch {
    $("#healthBadge").textContent = "server unreachable";
    $("#healthBadge").className = "badge bad";
  }
}
async function loadSettings() {
  state.settings = await api("/api/settings");
  $("#everyS").value = state.settings.sample_every_s;
  $("#batchN").value = state.settings.batch_frames;
  $("#analyzerSel").value = state.settings.analyzer;
}
async function loadForgeClips() {
  let clips = [];
  try { clips = await api("/api/forge/clips"); } catch (e) { $("#importMsg").textContent = `Cannot list FORGE results: ${e.message}`; }
  $("#forgeClip").innerHTML = clips.map((c) => `<option value="${esc(c.clip)}">${esc(c.clip)} · ${c.processed_until_s ? fmtT(c.processed_until_s) : "?"}`
    + ` · ${esc(c.steps_backend || "forge")}${c.imported_video_id ? " · imported" : ""}</option>`).join("")
    || "<option value=''>no FORGE results found</option>";
  $("#importBtn").disabled = !clips.length;
  state.forgeClips = clips;
}
$("#importBtn").addEventListener("click", async () => {
  const clip = $("#forgeClip").value;
  if (!clip) return;
  $("#importBtn").disabled = true;
  $("#importMsg").textContent = `Importing ${clip}…`;
  try {
    const v = await api("/api/forge/import", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ clip }) });
    $("#importMsg").textContent = `${clip} → ${v.video_id}`;
    await Promise.all([loadVideos(), loadForgeClips()]);
    select(v.video_id);
  } catch (e) { $("#importMsg").textContent = `Import failed: ${e.message}`; }
  $("#importBtn").disabled = false;
});
async function loadVideos() {
  try { state.videos = await api("/api/videos"); } catch (e) { toast(`Cannot load videos: ${e.message}`, true); return; }
  renderVideoList();
  renderCompareChips();
}
function renderVideoList() {
  $("#videoList").innerHTML = state.videos.map((v) => `
    <li data-id="${v.video_id}" class="${v.video_id === state.vid ? "sel" : ""}" title="${esc(v.filename)}">
      <span class="name">${esc(v.label || v.filename)}</span>
      <span class="sub"><span class="chip ${v.status}">${v.status}</span><span>${fmtD(v.duration_s)}</span>
        <span>${v.n_events ?? 0} events</span><span>${v.video_id}</span></span>
    </li>`).join("") || '<li class="note">No videos yet. Upload one above.</li>';
}
$("#videoList").addEventListener("click", (e) => { const li = e.target.closest("li[data-id]"); if (li) select(li.dataset.id); });
$("#refreshVideos").addEventListener("click", loadVideos);

// ---------------------------------------------------------------- upload
function pickFile(f) {
  if (!f) return;
  state.file = f;
  $("#fileInfo").textContent = `${f.name} · ${fmtMB(f.size)}`;
  $("#uploadBtn").disabled = false;
  if (!$("#labelInput").value) $("#labelInput").value = f.name.replace(/\.[^.]+$/, "");
  if (state.localUrl) URL.revokeObjectURL(state.localUrl);
  state.localUrl = URL.createObjectURL(f);
  closeLive();
  state.vid = null;
  renderVideoList();
  $("#videoTitle").textContent = `${f.name} (not uploaded yet)`;
  $("#videoMeta").innerHTML = "";
  $("#player").src = state.localUrl;
  $("#playerMsg").textContent = "Local preview from your computer. Upload it to analyse it on the backend.";
}
$("#fileInput").addEventListener("change", (e) => pickFile(e.target.files[0]));
const drop = $("#drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => pickFile(e.dataTransfer.files[0]));

$("#uploadBtn").addEventListener("click", () => {
  if (!state.file) return;
  const fd = new FormData();
  fd.append("file", state.file);
  fd.append("label", $("#labelInput").value);
  fd.append("sop", $("#sopInput").value);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/videos/upload");
  $("#uploadBar").classList.remove("hidden");
  $("#uploadBtn").disabled = true;
  $("#uploadMsg").className = "note";
  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    $("#uploadBar i").style.width = `${(100 * e.loaded) / e.total}%`;
    $("#uploadMsg").textContent = e.loaded < e.total ? `Uploading ${fmtMB(e.loaded)} / ${fmtMB(e.total)}…`
      : "Upload complete; backend is checking the video with OpenCV…";
  };
  xhr.onload = async () => {
    $("#uploadBtn").disabled = false;
    let body = {};
    try { body = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
    if (xhr.status !== 200) {
      $("#uploadMsg").textContent = `Upload failed: ${body.detail || xhr.statusText}`;
      $("#uploadMsg").className = "note err";
      return;
    }
    $("#uploadMsg").textContent = `Stored on the backend as ${body.video_id} (${body.stored_file}).`;
    state.file = null;
    $("#fileInput").value = "";
    $("#labelInput").value = "";
    $("#uploadBtn").disabled = true;
    await loadVideos();
    select(body.video_id, { keepLocal: true });
  };
  xhr.onerror = () => {
    $("#uploadBtn").disabled = false;
    $("#uploadMsg").textContent = "Upload failed: network error";
    $("#uploadMsg").className = "note err";
  };
  xhr.send(fd);
});

// ---------------------------------------------------------------- select a video
async function select(vid, { keepLocal = false } = {}) {
  closeLive();
  Object.assign(state, { vid, video: null, events: new Map(), workers: [], analytics: null, reports: [], batches: new Map(),
    compare: null, compareIds: new Set() });
  history.replaceState(null, "", `#v=${vid}`);
  renderVideoList();
  let v;
  try { v = await api(`/api/videos/${vid}`); } catch (e) { toast(e.message, true); return; }
  if (state.vid !== vid) return;
  state.video = v;
  state.reports = v.reports || [];
  $("#sopInput").value = (v.sop || []).join("\n");
  if (v.sample_every_s) $("#everyS").value = v.sample_every_s;
  if (v.batch_frames) $("#batchN").value = v.batch_frames;
  $("#maxSec").value = v.analysis_limit_s || 0;
  renderMeta();
  setPlayer(keepLocal);
  viewer3d.clear("");
  loadHands2d();
  renderH3dStatus();
  if (v.hands3d_status === "ready") loadH3d();
  renderStatus();
  renderDownloads();
  renderCompareChips();
  renderAll();
  openLive(vid);
}
function setPlayer(keepLocal) {
  const v = state.video, p = $("#player");
  const pending = v.preview_status === "pending";
  if (keepLocal && state.localUrl && pending) {
    p.src = state.localUrl;
  } else {
    p.src = v.media_url;
  }
  $("#playerMsg").textContent = pending
    ? `The ${v.codec} codec may not play in every browser; the backend is making an H.264 preview…`
    : v.preview_status === "failed" ? "Preview conversion failed; showing the original file." : "";
}
$("#player").addEventListener("error", () => {
  const v = state.video;
  if (!v) return;
  $("#playerMsg").textContent = v.preview_status === "pending"
    ? `This browser cannot play the ${v.codec} original; the backend is converting it to H.264 (the analysis does not need to wait).`
    : `This browser cannot play this file (${v.codec}).`;
});
$("#player").addEventListener("timeupdate", movePlayhead);
function renderMeta() {
  const v = state.video;
  $("#videoTitle").textContent = v.label || v.filename;
  $("#deleteBtn").classList.remove("hidden");
  $("#videoMeta").innerHTML = [
    ["id", v.video_id], ["file", v.filename], ["length", fmtD(v.duration_s)], ["size", fmtMB(v.size_bytes)],
    ["video", `${v.width}×${v.height} @ ${v.fps.toFixed(2)} fps · ${v.codec}`], ["backend file", v.stored_file],
    ["uploaded", new Date(v.created_at * 1000).toLocaleString()],
  ].map(([k, val]) => `<span>${k}: <b>${esc(val)}</b></span>`).join("");
}
$("#deleteBtn").addEventListener("click", async () => {
  if (!state.vid || !confirm("Delete this video, its events and its reports from the backend?")) return;
  try {
    await api(`/api/videos/${state.vid}`, { method: "DELETE" });
    closeLive();
    state.vid = null; state.video = null;
    history.replaceState(null, "", "#");
    location.reload();
  } catch (e) { toast(e.message, true); }
});

// ---------------------------------------------------------------- live connection
function closeLive() {
  clearTimeout(state.wsTimer);
  if (state.ws) { state.ws.onclose = null; state.ws.close(); state.ws = null; }
  setLive("–", "");
}
function setLive(text, cls) { const b = $("#liveBadge"); b.textContent = `live: ${text}`; b.className = `badge ${cls}`; }
function openLive(vid) {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/videos/${vid}/live`);
  state.ws = ws;
  setLive("connecting…", "warn");
  ws.onopen = () => { state.wsRetry = 0; setLive("connected", "ok"); };
  ws.onmessage = (m) => { if (state.vid === vid) handle(JSON.parse(m.data)); };
  ws.onclose = () => {
    if (state.vid !== vid) return;
    const wait = Math.min(15000, 1000 * 2 ** state.wsRetry++);
    setLive(`reconnecting in ${Math.round(wait / 1000)} s`, "bad");
    state.wsTimer = setTimeout(() => { if (state.vid === vid) openLive(vid); }, wait);
  };
}
function handle(msg) {
  switch (msg.type) {
    case "snapshot":
      state.events = new Map(msg.events.map((e) => [e.event_id, e]));
      state.workers = msg.workers;
      state.analytics = msg.analytics;
      state.reports = msg.reports;
      state.batches = new Map(msg.batches.map((b) => [b.batch_index, b]));
      Object.assign(state.video, msg.status);
      renderStatus(); renderDownloads(); renderBatches(); renderViewer(); renderAll();
      break;
    case "status":
      if (msg.status.status === "completed" && state.video.status !== "completed" && !state.hands2d) loadHands2d();
      Object.assign(state.video, msg.status);
      renderStatus();
      renderH3dStatus();
      updateListItem();
      break;
    case "hands3d":
      state.video.hands3d_status = msg.status;
      renderH3dStatus();
      if (msg.status === "ready") { loadH3d(); if (state.hands2d?.source !== "forge") loadHands2d(); toast("3D hands ready."); }
      if (msg.status === "failed") toast(`3D hands failed: ${state.video.hands3d_error || ""}`, true);
      break;
    case "event":
      upsertEvent(msg.event, msg.op);
      break;
    case "workers":
      state.workers = msg.workers;
      break;
    case "analytics":
      state.analytics = msg.analytics;
      queueRender();
      break;
    case "batch":
      state.batches.set(msg.batch.batch_index, { ...state.batches.get(msg.batch.batch_index), ...msg.batch });
      renderBatches();
      break;
    case "report":
      api(`/api/videos/${state.vid}/reports`).then((r) => { state.reports = r; renderDownloads(); }).catch(() => {});
      break;
    case "done":
      toast("Analysis complete. Final CSV and JSON reports are stored on the backend.");
      loadVideos();
      break;
    case "error":
      toast(`Analysis stopped: ${msg.error}`, true);
      loadVideos();
      break;
    case "preview":
      state.video.preview_status = msg.preview_status;
      if (msg.preview_status === "ready") {
        const p = $("#player"), t = p.currentTime || 0;
        p.src = state.video.media_url;
        p.addEventListener("loadedmetadata", () => { p.currentTime = t; }, { once: true });
        $("#playerMsg").textContent = "";
        toast("Browser preview ready (H.264).");
      }
      break;
    default:
      break;  // ping
  }
}
function updateListItem() {
  const item = state.videos.find((x) => x.video_id === state.vid);
  if (item && state.video) {
    item.status = state.video.status;
    item.n_events = state.events.size;
    renderVideoList();
  }
}
function queueRender() {
  if (state.renderQueued) return;
  state.renderQueued = true;
  requestAnimationFrame(() => { state.renderQueued = false; renderAll(); });
}

// ---------------------------------------------------------------- analysis controls + progress
function renderStatus() {
  const v = state.video;
  if (!v) return;
  const running = v.status === "processing" || v.status === "queued";
  const resumable = ["failed", "cancelled", "interrupted"].includes(v.status) && v.batches_done > 0;
  $("#startBtn").disabled = running;
  $("#startBtn").textContent = running ? "Analysis running…" : v.status === "completed" ? "▶ Analyse again"
    : resumable ? "▶ Resume analysis" : "▶ Start analysis";
  $("#restartBtn").classList.toggle("hidden", !resumable);
  $("#cancelBtn").classList.toggle("hidden", !running);
  for (const id of ["#maxSec", "#everyS", "#batchN", "#sopInput", "#analyzerSel"]) $(id).disabled = running || resumable;
  if (v.analyzer && (running || resumable)) $("#analyzerSel").value = v.analyzer;
  $("#analysisNote").textContent = v.analysis_note ? `Analysis: ${v.analysis_note}` : "";
  $("#progressFill").style.width = `${100 * (v.progress || 0)}%`;
  $("#progressFill").style.background = v.status === "failed" ? BAD : v.status === "completed" ? css.getPropertyValue("--ok") : "";
  $("#statusText").textContent = v.status;
  $("#stageText").textContent = v.error ? `${v.stage || ""} · ${v.error}` : v.stage || "";
  const end = v.analysis_limit_s || v.duration_s;
  $("#progressStats").innerHTML = [
    ["batches", `${v.batches_done} / ${v.batches_total || "–"}`], ["frames sent", `${v.frames_done} / ${v.frames_total || "–"}`],
    ["analysed", `${fmtT(v.processed_until_s)} of ${fmtT(end)}`],
    ["tokens in / out", `${(v.input_tokens || 0).toLocaleString()} / ${(v.output_tokens || 0).toLocaleString()}`],
    ["analyser", v.analyzer || "–"], ["model", v.model || "–"], ["frames from", v.frames_source || "–"],
  ].map(([k, val]) => `<span>${k}: <b>${esc(val)}</b></span>`).join("");
}
async function startAnalysis(restart) {
  const v = state.video;
  if (!v) return;
  if (v.status === "completed" && !confirm("Analyse again? The stored events and reports of this video are replaced.")) return;
  if (restart && !confirm("Restart from the beginning? Stored events of this video are deleted first.")) return;
  const maxSec = parseFloat($("#maxSec").value);
  const body = { restart, analyzer: $("#analyzerSel").value, max_seconds: maxSec > 0 ? maxSec : null, sop: $("#sopInput").value,
    sample_every_s: parseFloat($("#everyS").value) || null, batch_frames: parseInt($("#batchN").value, 10) || null };
  try {
    const st = await api(`/api/videos/${v.video_id}/analyze`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body) });
    Object.assign(state.video, st);
    if (restart || ["completed", "uploaded"].includes(v.status)) {
      state.events = new Map(); state.batches = new Map(); state.analytics = null;
      renderViewer(); renderBatches(); renderAll();
    }
    renderStatus();
    updateListItem();
  } catch (e) { toast(`Cannot start: ${e.message}`, true); }
}
$("#startBtn").addEventListener("click", () => startAnalysis(false));
$("#restartBtn").addEventListener("click", () => startAnalysis(true));
$("#cancelBtn").addEventListener("click", async () => {
  try { Object.assign(state.video, await api(`/api/videos/${state.vid}/cancel`, { method: "POST" })); renderStatus(); }
  catch (e) { toast(e.message, true); }
});
function renderBatches() {
  const rows = [...state.batches.values()].sort((a, b) => b.batch_index - a.batch_index);
  $("#batchLog").innerHTML = rows.map((b) => b.status === "offline"
    ? `<div><span>offline</span><span>${fmtT(b.start_ts)}–${fmtT(b.end_ts)}</span><span>${b.n_new} events from the FORGE labeller <span class="muted">· ${esc(b.notes || "")}</span></span></div>`
    : b.status === "failed"
    ? `<div class="failed"><span>#${b.batch_index + 1}</span><span>${fmtT(b.start_ts)}–${fmtT(b.end_ts)}</span><span>failed: ${esc(b.error)}</span></div>`
    : `<div><span>#${b.batch_index + 1}</span><span>${fmtT(b.start_ts)}–${fmtT(b.end_ts)}</span><span>${b.n_new} new · ${b.n_updated} updated`
      + `${b.n_rejected ? ` · <b style="color:${WARN}">${b.n_rejected} rejected by validation</b>` : ""}`
      + ` <span class="muted">· ${b.latency_s ?? "–"} s · ${(b.input_tokens || 0).toLocaleString()} tok in</span></span></div>`
  ).join("") || '<div><span></span><span class="muted">no batches yet</span><span></span></div>';
}

// ---------------------------------------------------------------- 3D hands (computed on the backend)
function renderH3dStatus() {
  const v = state.video, b = $("#h3dBtn");
  if (!v) { b.disabled = true; return; }
  const st = v.hands3d_status || "none";
  b.disabled = st === "queued" || st === "running";
  b.textContent = st === "running" ? `Computing… ${Math.round(100 * (v.hands3d_progress || 0))}%`
    : st === "queued" ? "Queued…" : st === "ready" ? "↻ Recompute" : "Compute 3D hands";
  if (st !== "ready" && !viewer3d.data) {
    $("#h3dMsg").textContent = st === "running" || st === "queued"
      ? "The backend is running the hand model on every 2nd frame and fitting each hand in 3D (about 2× real time)…"
      : st === "failed" ? `3D hands failed: ${v.hands3d_error || ""}`
      : "Press “Compute 3D hands”: the backend runs MediaPipe on this video and fits each hand in 3D (local CPU, no API).";
  }
  renderDownloads();
}
async function loadH3d() {
  const vid = state.vid;
  try {
    const data = await api(`/api/videos/${vid}/hands3d`);
    if (state.vid !== vid) return;
    viewer3d.setData(data);
    const s = data.summary;
    $("#h3dNote").textContent = `Approximate monocular 3D (${s.coords}). ${s.note} Lens assumed: ${s.camera.lens}, `
      + `${s.camera.hfov_deg}° horizontal FOV (STUDIO_HANDS3D_HFOV_DEG).`;
  } catch (e) { viewer3d.clear(`Cannot load 3D hands: ${e.message}`); }
}
$("#h3dBtn").addEventListener("click", async () => {
  if (!state.vid) return;
  try { Object.assign(state.video, await api(`/api/videos/${state.vid}/hands3d`, { method: "POST" })); renderH3dStatus(); }
  catch (e) { toast(e.message, true); }
});
$("#h3dViews").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-view]");
  if (!b) return;
  document.querySelectorAll("#h3dViews button").forEach((x) => x.classList.toggle("active", x === b));
  viewer3d.setView(b.dataset.view);
});
$("#h3dTrails").addEventListener("change", (e) => { viewer3d.trails = e.target.checked; viewer3d.render(); });

// ---------------------------------------------------------------- 2D hand tracking on the video
// Tracks come from the backend (FORGE hands.jsonl, or the 3D job's 2D keypoints); the browser only draws them.
const HAND_EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],
  [14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];
const HAND_COLORS = [css.getPropertyValue("--left").trim(), css.getPropertyValue("--right").trim()];
const FINGERTIPS = [4, 8, 12, 16, 20];
async function loadHands2d() {
  const vid = state.vid;
  state.hands2d = null;
  drawHands();
  $("#handsNote").textContent = "";
  let d;
  try { d = await api(`/api/videos/${vid}/hands2d`); } catch (e) {
    if (state.vid === vid) $("#handsNote").textContent = "No hand tracks for this video yet: run the offline labeller or press “Compute 3D hands”.";
    return;
  }
  if (state.vid !== vid) return;
  const gaps = d.t.slice(1).map((t, i) => t - d.t[i]).sort((a, b) => a - b);
  d.maxGap = 1.5 * (gaps[gaps.length >> 1] || 0.04) + 0.02;  // a frame further away than this has no tracks
  state.hands2d = d;
  const withHands = d.frames.filter((f) => f.length).length;
  $("#handsNote").textContent = `${d.source === "forge" ? "FORGE hand tracker (MediaPipe + tracking)" : "3D hands job (MediaPipe, every 2nd frame)"}`
    + ` · hands in ${pct(withHands / Math.max(d.frames.length, 1))} of ${d.frames.length} frames`;
  drawHands();
}
function handsAt(t) {
  const d = state.hands2d;
  let lo = 0, hi = d.t.length - 1;
  if (hi < 0) return null;
  while (lo < hi) { const m = (lo + hi) >> 1; if (d.t[m] < t) lo = m + 1; else hi = m; }
  if (lo > 0 && t - d.t[lo - 1] < d.t[lo] - t) lo -= 1;
  return Math.abs(d.t[lo] - t) <= d.maxGap ? d.frames[lo] : null;
}
function drawHands() {
  const c = $("#handsOverlay"), p = $("#player");
  const w = c.clientWidth, h = c.clientHeight, dpr = window.devicePixelRatio || 1;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const d = state.hands2d;
  if (!d || !$("#handsToggle").checked || !p.videoWidth) return;
  const hands = handsAt(p.currentTime);
  if (!hands) return;
  // The video is letterboxed inside the stage (object-fit: contain); tracks are in the original video's pixels.
  const s = Math.min(w / p.videoWidth, h / p.videoHeight);
  const ox = (w - s * p.videoWidth) / 2, oy = (h - s * p.videoHeight) / 2;
  const kx = s * p.videoWidth / (d.width || p.videoWidth), ky = s * p.videoHeight / (d.height || p.videoHeight);
  const X = (x) => ox + x * kx, Y = (y) => oy + y * ky;
  for (const hd of hands) {
    const [id, side, score, interp, x1, y1, x2, y2] = hd, kp = hd.slice(8), col = HAND_COLORS[side];
    g.globalAlpha = interp ? 0.55 : 1;
    g.strokeStyle = col; g.lineWidth = 2; g.setLineDash(interp ? [6, 4] : []);
    g.strokeRect(X(x1), Y(y1), X(x2) - X(x1), Y(y2) - Y(y1));
    g.setLineDash([]); g.lineWidth = 2.5; g.beginPath();
    for (const [a, b] of HAND_EDGES) { g.moveTo(X(kp[2 * a]), Y(kp[2 * a + 1])); g.lineTo(X(kp[2 * b]), Y(kp[2 * b + 1])); }
    g.stroke();
    for (let i = 0; i < 21; i++) {
      g.fillStyle = FINGERTIPS.includes(i) ? col : "#fff";
      g.beginPath(); g.arc(X(kp[2 * i]), Y(kp[2 * i + 1]), FINGERTIPS.includes(i) ? 4 : 2.5, 0, 2 * Math.PI); g.fill();
    }
    const tag = `${side ? "R" : "L"}${id >= 0 ? ` #${id}` : ""} ${score}%${interp ? " (interp.)" : ""}`;
    g.font = "600 12px system-ui, sans-serif";
    const tw = g.measureText(tag).width, ty = Math.max(Y(y1) - 18, 0);
    g.fillStyle = col; g.fillRect(X(x1), ty, tw + 8, 17);
    g.fillStyle = "#000"; g.fillText(tag, X(x1) + 4, ty + 13);
  }
  g.globalAlpha = 1;
}
(function handsLoop() {
  const p = $("#player");
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    const tick = () => { drawHands(); p.requestVideoFrameCallback(tick); };
    p.requestVideoFrameCallback(tick);
  }
  for (const ev of ["seeked", "loadeddata", "pause", "timeupdate"]) p.addEventListener(ev, drawHands);
  $("#handsToggle").addEventListener("change", drawHands);
  new ResizeObserver(drawHands).observe($("#handsOverlay"));
})();

// ---------------------------------------------------------------- downloads
function renderDownloads() {
  const v = state.video;
  if (!v) return;
  const analysed = v.status !== "uploaded" || state.events.size > 0;
  const items = [["csv", "events", "Events CSV", "every validated event with worker, timing, verification and flags"],
    ["csv", "workers", "Workers CSV", "per-worker statistics computed by the backend"],
    ["json", "report", "JSON report", "video, workers, events, analytics, anomalies and the analysis batch log"]];
  const running = v.status !== "completed";
  $("#downloads").innerHTML = items.map(([kind, table, title, what]) => {
    const rep = state.reports.find((r) => r.kind === kind && r.table_name === table);
    const href = kind === "csv" ? `/api/videos/${v.video_id}/export/csv?table=${table}` : `/api/videos/${v.video_id}/export/json`;
    const info = rep ? `<span class="tag ${rep.status}">${rep.status}</span> ${rep.n_events} events · stored ${when(rep.created_at)}`
      + `<br><code>${esc(rep.path)}</code>` : analysed ? "generated on download" : "available once analysis has started";
    return `<div class="dl"><div class="what"><b>${title}</b>${what}<br>${info}</div>`
      + `<a class="${analysed ? "" : "off"}" href="${href}" download>⬇ ${kind.toUpperCase()}${running && analysed ? " (partial)" : ""}</a></div>`;
  }).join("") + (v.hands3d_status === "ready" ? `<div class="dl"><div class="what"><b>3D hands (JSONL)</b>one line per processed frame: `
    + `21 keypoints per hand in metres (camera/head frame), wrist axes, aperture, speed</div>`
    + `<a href="/api/videos/${v.video_id}/export/hands3d" download>⬇ JSONL</a></div>` : "")
    + (running && analysed ? '<div class="note">While the analysis is not complete, downloads are partial: '
    + "the file name ends in _PARTIAL and every row says report_status=partial.</div>" : "");
}

// ---------------------------------------------------------------- live JSON viewer
function eventJson(e) {
  const obj = $("#fullRecord").checked ? e : Object.fromEntries(KEY_FIELDS.filter((k) => !(k === "sop_step" && e[k] == null)).map((k) => [k, e[k]]));
  return esc(JSON.stringify(obj, null, 2))
    .replace(/(&quot;[^&]*?&quot;)(\s*:)/g, '<span class="k">$1</span>$2')
    .replace(/:\s(&quot;.*?&quot;)/g, ': <span class="s">$1</span>')
    .replace(/:\s(-?\d+(?:\.\d+)?)/g, ': <span class="n">$1</span>')
    .replace(/:\s(true|false|null)/g, ': <span class="b">$1</span>');
}
const sortedEvents = () => [...state.events.values()].sort((a, b) => a.start_ts - b.start_ts || a.seq - b.seq);
function passes(e) {
  const w = $("#filterWorker").value, s = $("#filterStatus").value;
  return (!w || e.worker_id === w) && (!s || e.verification_status === s);
}
function eventNode(e) {
  const d = document.createElement("div");
  d.className = "ev";
  d.dataset.id = e.event_id;
  d.style.borderLeftColor = workerColor(e.worker_id);
  d.innerHTML = eventJson(e);
  return d;
}
function renderViewer() {
  const box = $("#jsonViewer");
  const evs = sortedEvents().filter(passes);
  box.replaceChildren(...evs.map(eventNode));
  if (!evs.length) box.innerHTML = `<div class="empty">${state.events.size ? "No events match the filters."
    : "Events appear here as soon as the backend has validated and stored them."}</div>`;
  renderViewerMeta();
}
function renderViewerMeta() {
  $("#eventCount").textContent = state.events.size ? `· ${state.events.size}` : "";
  const sel = $("#filterWorker"), cur = sel.value;
  const ids = [...new Set([...state.events.values()].map((e) => e.worker_id))].sort((a, b) => parseInt(a.slice(1)) - parseInt(b.slice(1)));
  sel.innerHTML = '<option value="">all workers</option>' + ids.map((w) => `<option ${w === cur ? "selected" : ""}>${w}</option>`).join("");
}
function upsertEvent(e, op) {
  const isNew = !state.events.has(e.event_id);
  state.events.set(e.event_id, e);
  const box = $("#jsonViewer");
  box.querySelector(".empty")?.remove();
  let node = box.querySelector(`.ev[data-id="${e.event_id}"]`);
  if (!passes(e)) { node?.remove(); renderViewerMeta(); queueRender(); return; }
  const fresh = eventNode(e);
  if (node) node.replaceWith(fresh);
  else {
    const after = sortedEvents().filter(passes);
    const idx = after.findIndex((x) => x.event_id === e.event_id);
    const next = after.slice(idx + 1).map((x) => box.querySelector(`.ev[data-id="${x.event_id}"]`)).find(Boolean);
    box.insertBefore(fresh, next || null);
  }
  fresh.classList.add(isNew || op === "created" ? "flash-new" : "flash-upd");
  if ($("#followNew").checked) {  // scroll the viewer only, never the page
    const top = fresh.offsetTop - box.offsetTop, bottom = top + fresh.offsetHeight;
    if (top < box.scrollTop || bottom > box.scrollTop + box.clientHeight) box.scrollTo({ top: Math.max(0, bottom - box.clientHeight + 6), behavior: "smooth" });
  }
  renderViewerMeta();
  queueRender();
}
$("#jsonViewer").addEventListener("click", (ev) => { const d = ev.target.closest(".ev"); if (d) seek(state.events.get(d.dataset.id).start_ts); });
for (const id of ["#filterWorker", "#filterStatus", "#fullRecord"]) $(id).addEventListener("change", renderViewer);

// ---------------------------------------------------------------- analytics rendering (numbers from the backend)
function renderAll() {
  renderTiles(); renderAlerts(); renderWorkers(); renderComparison(); renderTimeline(); renderTaskChart();
  $("#timelineNote").textContent = state.analytics?.is_partial && state.video?.status !== "uploaded"
    ? `analysed up to ${fmtT(state.video.processed_until_s)}` : "";
}
function renderTiles() {
  const s = state.analytics?.summary;
  const tiles = s ? [["Workers", s.n_workers], ["Events", s.n_events], ["Observed task time", fmtD(s.total_s)],
    ["Average duration", fmtD(s.avg_s)], ["Median duration", fmtD(s.median_s)], ["Unusually long", s.n_long_tasks, s.n_long_tasks > 0],
    ["Incomplete", s.n_incomplete, s.n_incomplete > 0], ["In progress", s.n_in_progress],
    ["Low confidence", s.verification?.low_confidence || 0]] : [];
  $("#tiles").innerHTML = tiles.map(([k, v, warn]) => `<div class="tile ${warn ? "warn" : ""}"><b>${esc(v)}</b><span>${k}</span></div>`).join("")
    || '<span class="note">Statistics appear when the first batch has been analysed.</span>';
}
function alertList(title, icon, items, cls, render, emptyText) {
  return `<div><h3>${icon} ${title} <span class="count">${items.length}</span></h3>`
    + (items.length ? `<ul>${items.map((x) => `<li class="${cls}${x.start_ts != null ? " click" : ""}" ${x.start_ts != null ? `data-t="${x.start_ts}"` : ""}>${render(x)}</li>`).join("")}</ul>`
      : `<div class="empty">${emptyText}</div>`) + "</div>";
}
function renderAlerts() {
  const a = state.analytics;
  if (!a) { $("#alerts").innerHTML = '<div class="empty note">No analysis results yet.</div>'; return; }
  const partial = a.is_partial ? `<div class="note">Live figures: ${state.video?.status === "processing" ? "analysis running" : `analysis ${state.video?.status}`}, `
    + `${fmtT(a.processed_until_s)} analysed. Alerts update after every batch.</div>` : "";
  const ev = (x) => `<b>${esc(x.event_id)} · ${esc(x.worker_id)} · ${esc(x.task)}</b> ${fmtT(x.start_ts)}–${fmtT(x.end_ts)} (${fmtD(x.duration_s)})`;
  const sop = a.sop;
  const sopHtml = !sop ? '<div><h3>📋 SOP deviations</h3><div class="empty">No SOP provided for this video.</div></div>'
    : alertList("SOP deviations", "📋", sop.deviations, "warn", (d) => `<b>${esc(d.worker_id)} · step ${d.step}: ${esc(d.step_text)}</b>`
      + `<span class="why">${esc(d.type.replaceAll("_", " "))}: ${esc(d.detail)}${d.event_ids.length ? ` (${d.event_ids.join(", ")})` : ""}</span>`,
    "No deviations from the SOP.") + sop.workers.map((w) => `<div class="note">${esc(w.worker_id)}: ${w.n_cycles} cycle${w.n_cycles === 1 ? "" : "s"}`
      + ` (${w.complete_cycles} complete${w.median_cycle_s != null ? `, median ${fmtD(w.median_cycle_s)}` : ""})`
      + ` · ${w.compliance_pct ?? "–"}% of expected steps in order`
      + `${w.cycles.some((c) => c.partial_start || c.partial_end) ? " · first/last cycle cut off by the clip edges (not counted as skipped)" : ""}</div>`).join("");
  // offline-labeller events (imported FORGE results may have been labelled by Gemini: those are not flagged)
  const offlineN = [...state.events.values()].filter((e) => /^(offline|import)/.test(e.source || "") && !/gemini/.test(e.source)).length;
  const warn = offlineN ? `<div class="source-warn">⚠ ${offlineN} of ${state.events.size} events come from FORGE's offline labeller, `
    + "whose measured accuracy is far below Gemini's (docs/RESULTS.md). Treat these alerts as hints, and check them in the video.</div>" : "";
  $("#alerts").innerHTML = warn + partial
    + alertList("Unusually long tasks", "⚠", a.long_tasks, "warn", (x) => `${ev(x)}<span class="why">${esc(x.reason)}</span>`, "None flagged.")
    + alertList("Incomplete tasks", "⛔", a.incomplete_tasks, "bad", (x) => `${ev(x)}<span class="why">${esc(x.reason)}</span>`, "None.")
    + alertList("Potential bottlenecks", "⏱", a.bottlenecks, "info", (b) => `<b>${esc(b.type.replaceAll("_", " "))}${b.worker_id ? ` · ${esc(b.worker_id)}` : ""}</b>`
      + `<span class="why">${esc(b.reason)}</span>`, "None found.")
    + sopHtml;
}
$("#alerts").addEventListener("click", (e) => { const li = e.target.closest("li[data-t]"); if (li) seek(parseFloat(li.dataset.t)); });

function renderWorkers() {
  const ws = (state.analytics?.workers || []).filter((w) => w.n_actions > 0);
  $("#workers").innerHTML = ws.map((w) => `
    <div class="worker" style="border-top-color:${workerColor(w.worker_id)}">
      <header><b><span class="swatch" style="background:${workerColor(w.worker_id)}"></span>${esc(w.worker_id)}</b>
        <span>${esc(w.description)}${w.is_camera_wearer ? " · camera wearer" : ""}</span></header>
      <div class="kv small">${[["actions", w.n_actions], ["average", fmtD(w.avg_s)], ["median", fmtD(w.median_s)],
        ["total observed", fmtD(w.total_s)], ["productive", fmtD(w.productive_s)], ["waiting", `${fmtD(w.wait_s)} (${pct(w.wait_share)})`],
        ["unusually long", w.n_long_tasks], ["incomplete", w.n_incomplete], ["in progress", w.n_in_progress],
        ["low confidence", w.n_low_confidence], ["seen", `${fmtT(w.first_seen_ts)}–${fmtT(w.last_seen_ts)}`]]
        .map(([k, v]) => `<span>${k}: <b>${esc(v)}</b></span>`).join("")}</div>
      <table><thead><tr><th>task</th><th class="num">n</th><th class="num">median</th><th class="num">total</th><th class="num">max</th></tr></thead>
        <tbody>${w.tasks.map((t) => `<tr><td>${esc(t.task)}</td><td class="num">${t.count}</td><td class="num">${fmtD(t.median_s)}</td>`
          + `<td class="num">${fmtD(t.total_s)}</td><td class="num">${fmtD(t.max_s)}</td></tr>`).join("")}</tbody></table>
    </div>`).join("") || '<span class="note">No workers detected yet.</span>';
}

// ---------------------------------------------------------------- worker comparison (small multiples, one metric each)
function renderCompareChips() {
  const others = state.videos.filter((v) => v.video_id !== state.vid && (v.n_events || 0) > 0);
  $("#compareWith").innerHTML = others.map((v) => `<label><input type="checkbox" value="${v.video_id}" ${state.compareIds.has(v.video_id) ? "checked" : ""}>`
    + `${esc(v.label || v.filename)}</label>`).join("") || '<span class="note">none analysed yet</span>';
}
$("#compareWith").addEventListener("change", async (e) => {
  const id = e.target.value;
  if (e.target.checked) state.compareIds.add(id); else state.compareIds.delete(id);
  await loadCompare();
});
async function loadCompare() {
  if (!state.compareIds.size || !state.vid) { state.compare = null; renderComparison(); return; }
  try { state.compare = await api(`/api/compare?video_ids=${[state.vid, ...state.compareIds].join(",")}`); }
  catch (e) { toast(e.message, true); state.compare = null; }
  renderComparison();
}
function comparisonRows() {
  if (state.compare) {
    return state.compare.workers.map((w, i) => ({ ...w, label: `${w.video_label} · ${w.worker_id}`,
      color: i < 8 ? PALETTE[i] : OTHER }));
  }
  return (state.analytics?.comparison?.metrics || []).map((w) => ({ ...w, key: w.worker_id, label: w.worker_id, color: workerColor(w.worker_id) }));
}
function barMultiple(rows, field, title, fmt) {
  const lab = 132, w = Math.max(240, ($("#comparison").clientWidth - 18) / 2), h = 8 + rows.length * 22;
  const max = Math.max(...rows.map((r) => r[field] || 0), 1e-9);
  const wrap = document.createElement("div");
  wrap.innerHTML = `<h3>${title}</h3>`;
  const svg = svgEl("svg", { viewBox: `0 0 ${w} ${h}`, height: h });
  rows.forEach((r, i) => {
    const y = 4 + i * 22, bw = Math.max(r[field] ? 2 : 0, ((w - lab - 70) * (r[field] || 0)) / max);
    svgEl("text", { x: lab - 6, y: y + 12, "text-anchor": "end", class: "lbl" }, svg).textContent = r.label.length > 22 ? `${r.label.slice(0, 21)}…` : r.label;
    svgEl("rect", { x: lab, y: y + 2, width: bw, height: 14, rx: 3, fill: r.color }, svg);
    svgEl("text", { x: lab + bw + 6, y: y + 13, class: "val" }, svg).textContent = fmt(r[field]);
    const hit = svgEl("rect", { x: 0, y, width: w, height: 20, class: "hit" }, svg);
    hit.addEventListener("mousemove", (e) => showTip(`<b>${esc(r.label)}</b><span>${title}: ${fmt(r[field])}</span>`, e));
    hit.addEventListener("mouseleave", hideTip);
  });
  wrap.appendChild(svg);
  wrap.classList.add("chart");
  return wrap;
}
function renderComparison() {
  const rows = comparisonRows();
  const box = $("#comparison");
  if (!rows.length) { box.innerHTML = '<div class="note">Worker comparison appears once workers have been detected.</div>'; return; }
  const metrics = [["n_actions", "Actions", (v) => String(v ?? "–")], ["avg_s", "Average task duration", fmtD],
    ["median_s", "Median task duration", fmtD], ["total_s", "Total observed task time", fmtD]];
  const shared = state.compare ? state.compare.shared_tasks : state.analytics?.comparison?.shared_tasks || [];
  const labelOf = (key) => rows.find((r) => r.key === key)?.label || key;
  const keys = rows.map((r) => r.key);
  const multiples = document.createElement("div");
  multiples.className = "small-multiples";
  multiples.append(...metrics.map(([f, t, fmt]) => barMultiple(rows, f, t, fmt)));
  box.replaceChildren(multiples);
  box.insertAdjacentHTML("beforeend", `<table><thead><tr><th>worker</th><th class="num">actions</th><th class="num">avg</th><th class="num">median</th>`
    + `<th class="num">total</th><th class="num">waiting</th><th class="num">long</th><th class="num">incomplete</th></tr></thead><tbody>`
    + rows.map((r) => `<tr><td><span class="swatch" style="background:${r.color}"></span>${esc(r.label)}</td><td class="num">${r.n_actions}</td>`
      + `<td class="num">${fmtD(r.avg_s)}</td><td class="num">${fmtD(r.median_s)}</td><td class="num">${fmtD(r.total_s)}</td>`
      + `<td class="num">${pct(r.wait_share)}</td><td class="num">${r.n_long_tasks}</td><td class="num">${r.n_incomplete}</td></tr>`).join("")
    + "</tbody></table>"
    + (shared.length ? `<h3>Same task, different workers (median duration)</h3><table><thead><tr><th>task</th>`
      + keys.map((k) => `<th class="num">${esc(labelOf(k))}</th>`).join("") + "<th>fastest</th></tr></thead><tbody>"
      + shared.map((t) => `<tr><td>${esc(t.task)}</td>` + keys.map((k) => `<td class="num">${t.workers[k] ? `${fmtD(t.workers[k].median_s)} (n=${t.workers[k].count})` : "–"}</td>`).join("")
        + `<td>${esc(labelOf(t.fastest))}</td></tr>`).join("") + "</tbody></table>"
      : '<div class="note">No task was done by more than one worker yet. Tick another analysed video above to compare workers across videos.</div>'));
}

// ---------------------------------------------------------------- timeline (Gantt per worker)
function renderTimeline() {
  const box = $("#timeline");
  const v = state.video;
  const evs = sortedEvents();
  if (!v || !evs.length) { box.innerHTML = `<div class="empty">${v ? "Events will be drawn here as they arrive." : ""}</div>`; return; }
  const longIds = new Set((state.analytics?.long_tasks || []).map((x) => x.event_id));
  const workers = [...new Set(evs.map((e) => e.worker_id))].sort((a, b) => parseInt(a.slice(1)) - parseInt(b.slice(1)));
  const gut = 40, W = Math.max(300, box.clientWidth), rowH = 26, top = 4, H = top + workers.length * rowH + 22;
  const dur = v.analysis_limit_s || v.duration_s, x = (t) => gut + ((W - gut - 6) * t) / dur;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, height: H });
  svgEl("rect", { x: gut, y: top, width: Math.max(0, x(v.processed_until_s || 0) - gut), height: workers.length * rowH, class: "analysed" }, svg);
  const g = svgEl("g", { class: "axis" }, svg), step = niceStep(dur);
  for (let t = 0; t <= dur + 1e-6; t += step) {
    svgEl("line", { x1: x(t), x2: x(t), y1: top, y2: top + workers.length * rowH, "stroke-dasharray": "2 4" }, g);
    svgEl("text", { x: x(t), y: H - 6, "text-anchor": "middle" }, g).textContent = fmtT(t).replace(/\.0$/, "");
  }
  workers.forEach((w, i) => {
    svgEl("text", { x: 4, y: top + i * rowH + 17, class: "lbl" }, svg).textContent = w;
  });
  for (const e of evs) {
    const i = workers.indexOf(e.worker_id), y = top + i * rowH + 4, x0 = x(e.start_ts), bw = Math.max(2, x(e.end_ts) - x0 - 2);
    const r = svgEl("rect", { x: x0, y, width: bw, height: 18, rx: 4, fill: workerColor(e.worker_id),
      "fill-opacity": e.completion_status === "in_progress" ? 0.55 : 1 }, svg);
    if (longIds.has(e.event_id)) { r.setAttribute("stroke", WARN); r.setAttribute("stroke-width", 2); }
    if (e.completion_status === "incomplete") { r.setAttribute("stroke", BAD); r.setAttribute("stroke-width", 2); r.setAttribute("stroke-dasharray", "4 3"); }
    if (bw > 46) {
      const t = svgEl("text", { x: x0 + 5, y: y + 13, class: "val", "pointer-events": "none" }, svg);
      t.textContent = e.action.length * 6.5 > bw - 8 ? e.action.slice(0, Math.max(1, Math.floor((bw - 8) / 6.5))) : e.action;
      t.setAttribute("fill", "#ffffff");
    }
    const hit = svgEl("rect", { x: x0 - 2, y: y - 3, width: bw + 4, height: 24, class: "hit" }, svg);
    hit.addEventListener("mousemove", (m) => showTip(eventTip(e), m));
    hit.addEventListener("mouseleave", hideTip);
    hit.addEventListener("click", () => seek(e.start_ts));
  }
  state.timelineX = x;
  svgEl("line", { id: "playhead", class: "playhead", x1: gut, x2: gut, y1: top - 2, y2: top + workers.length * rowH + 2 }, svg);
  box.replaceChildren(svg);
  movePlayhead();
}
function movePlayhead() {
  const ph = document.getElementById("playhead");
  if (!ph || !state.timelineX) return;
  const xx = state.timelineX(Math.min($("#player").currentTime || 0, state.video?.analysis_limit_s || state.video?.duration_s || 0));
  ph.setAttribute("x1", xx); ph.setAttribute("x2", xx);
}

// ---------------------------------------------------------------- task durations (dot strip per task)
function renderTaskChart() {
  const box = $("#taskChart");
  const a = state.analytics;
  if (!a || !a.tasks.length) { box.innerHTML = '<div class="empty">Task durations appear after the first events.</div>'; return; }
  const tasks = a.tasks.slice(0, 14), longIds = new Set(a.long_tasks.map((x) => x.event_id));
  const byTask = new Map(tasks.map((t) => [t.task, []]));
  for (const e of state.events.values()) byTask.get(a.event_task?.[e.event_id])?.push(e);
  const lab = 150, right = 96, W = Math.max(320, box.clientWidth), rowH = 26, H = 8 + tasks.length * rowH + 20;
  const max = Math.max(...tasks.map((t) => t.max_s || 0), 1), x = (s) => lab + ((W - lab - right) * s) / max;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, height: H });
  const g = svgEl("g", { class: "axis" }, svg), step = niceStep(max, 5);
  for (let s = 0; s <= max + 1e-6; s += step) {
    svgEl("line", { x1: x(s), x2: x(s), y1: 4, y2: H - 18, "stroke-dasharray": "2 4" }, g);
    svgEl("text", { x: x(s), y: H - 4, "text-anchor": "middle" }, g).textContent = `${+s.toFixed(1)} s`;
  }
  tasks.forEach((t, i) => {
    const cy = 8 + i * rowH + 10;
    svgEl("text", { x: lab - 8, y: cy + 4, "text-anchor": "end", class: "lbl" }, svg).textContent = t.task.length > 24 ? `${t.task.slice(0, 23)}…` : t.task;
    svgEl("line", { x1: x(t.min_s), x2: x(t.max_s), y1: cy, y2: cy, stroke: OTHER, "stroke-width": 2, "stroke-linecap": "round" }, svg);
    for (const e of byTask.get(t.task) || []) {
      const long = longIds.has(e.event_id);
      const c = svgEl("circle", { cx: x(e.duration_s), cy, r: 4.5, fill: workerColor(e.worker_id), stroke: long ? WARN : css.getPropertyValue("--panel").trim(),
        "stroke-width": long ? 2.5 : 1.5 }, svg);
      c.addEventListener("mousemove", (m) => showTip(eventTip(e), m));
      c.addEventListener("mouseleave", hideTip);
      c.addEventListener("click", () => seek(e.start_ts));
      c.style.cursor = "pointer";
    }
    svgEl("line", { x1: x(t.median_s), x2: x(t.median_s), y1: cy - 8, y2: cy + 8, stroke: css.getPropertyValue("--text").trim(), "stroke-width": 2 }, svg);
    svgEl("text", { x: W - right + 8, y: cy + 4, class: "val" }, svg).textContent = `n=${t.count} · ${fmtD(t.median_s)}`;
    const hit = svgEl("rect", { x: 0, y: cy - 10, width: lab, height: 20, class: "hit" }, svg);
    hit.addEventListener("mousemove", (m) => showTip(`<b>${esc(t.task)}</b><span>${t.count} events · median ${fmtD(t.median_s)} · `
      + `range ${fmtD(t.min_s)}–${fmtD(t.max_s)} · total ${fmtD(t.total_s)}${t.n_long ? ` · ${t.n_long} unusually long` : ""}</span>`, m));
    hit.addEventListener("mouseleave", hideTip);
  });
  box.replaceChildren(svg);
}

// ---------------------------------------------------------------- boot
let resizeTimer;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderAll, 150); });
window.addEventListener("hashchange", () => { const id = location.hash.match(/v=([0-9a-f]{12})/)?.[1]; if (id && id !== state.vid) select(id); });
(async () => {
  viewer3d.clear("Select or upload a video, then compute its 3D hands.");
  await Promise.all([loadHealth(), loadSettings(), loadVideos(), loadForgeClips()]);
  setInterval(loadHealth, 10000);
  setInterval(loadVideos, 15000);
  const id = location.hash.match(/v=([0-9a-f]{12})/)?.[1];
  if (id && state.videos.some((v) => v.video_id === id)) select(id);
  renderAll();
})();
