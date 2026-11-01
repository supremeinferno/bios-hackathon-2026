// 3D hand viewer: plain canvas, no libraries. Data and every number shown come from the backend
// (/api/videos/{id}/hands3d, egovision/hands3d.py); the browser only projects them to the screen.
// Frame: metres, origin = camera on the head, X right, Y forward (out of the lens), Z up.
const EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];
// Viewing angles; the distance is chosen to fit the scene (camera + where this clip's hands are) into the canvas.
const VIEWS = { "3d": { yaw: 1.0, pitch: 0.35 }, head: { yaw: 0, pitch: 0.15 }, top: { yaw: 0, pitch: 1.45 },
  side: { yaw: Math.PI / 2, pitch: 0.1 } };
const step10 = (x, up) => (up ? Math.ceil(x * 10) : Math.floor(x * 10)) / 10;
const pct = (arr, q) => { const s = [...arr].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => mul(a, 1 / Math.max(Math.hypot(...a), 1e-9));
const f2 = (x) => (x == null || x < 0 ? "–" : x.toFixed(2));

function unpack(h) {  // [side, score, dist, aperture, speed, err, wrist(3), axes(9), kp(63)]
  const kp = [];
  for (let j = 0; j < 21; j++) kp.push(h.slice(18 + 3 * j, 21 + 3 * j));
  return { side: h[0] ? "right" : "left", score: h[1], dist: h[2], aperture: h[3], speed: h[4], err: h[5],
    wrist: h.slice(6, 9), axes: [h.slice(9, 12), h.slice(12, 15), h.slice(15, 18)], kp };
}

export class Hands3DViewer {
  constructor({ canvas, panel, message, player, colors }) {
    Object.assign(this, { canvas, panel, message, player, colors });
    this.ctx = canvas.getContext("2d");
    this.data = null;
    this.view = { ...VIEWS["3d"], dist: null };
    this.setBounds([]);
    this.trails = true;
    let drag = null;
    canvas.addEventListener("mousedown", (e) => { drag = { x: e.clientX, y: e.clientY, ...this.view }; });
    window.addEventListener("mouseup", () => { drag = null; });
    window.addEventListener("mousemove", (e) => {
      if (!drag) return;
      this.view.yaw = drag.yaw - (e.clientX - drag.x) * 0.008;
      this.view.pitch = Math.max(-1.45, Math.min(1.45, drag.pitch + (e.clientY - drag.y) * 0.008));
      this.render();
    });
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.view.dist = Math.max(0.15, Math.min(12, (this.view.dist || 1) * (e.deltaY > 0 ? 1.1 : 0.9)));
      this.fitSize = "manual";  // keep the user's zoom when the canvas is resized
      this.render();
    }, { passive: false });
    canvas.addEventListener("dblclick", () => this.setView("3d"));
    const loop = () => { if (this.data && !player.paused) this.render(); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
    ["seeked", "timeupdate", "pause"].forEach((ev) => player.addEventListener(ev, () => this.data && this.render()));
    window.addEventListener("resize", () => this.render());
  }

  setView(name) { this.view = { ...VIEWS[name], dist: null }; this.fitSize = null; this.render(); }

  setBounds(wrists) {
    // Grid and framing from where this clip's hands are (5th-95th percentile), always including the camera.
    const has = wrists.length > 0, q = (k, v) => (has ? pct(wrists.map((p) => p[k]), v) : null);
    const b = { x0: step10(Math.min(has ? q(0, 0.05) - 0.04 : -0.3, -0.1), false), x1: step10(Math.max(has ? q(0, 0.95) + 0.04 : 0.3, 0.1), true),
      y1: step10(Math.max(has ? q(1, 0.95) + 0.04 : 0.6, 0.3), true), floor: step10(Math.min(has ? q(2, 0.05) : -0.3, 0) - 0.04, false),
      top: Math.max(has ? q(2, 0.95) : 0, 0) + 0.06 };
    this.bounds = b;
    this.keys = [];
    for (const x of [b.x0, b.x1]) for (const y of [0, b.y1]) for (const z of [b.floor, b.top]) this.keys.push([x, y, z]);
    this.target = [(b.x0 + b.x1) / 2, b.y1 / 2, (b.floor + b.top) / 2];
    this.view.dist = null;
    this.fitSize = null;
  }

  fit(W, H) {
    // smallest viewing distance at which every key point projects inside the canvas (with a margin)
    const ok = (d) => {
      const P = this.camera(W, H, d);
      return this.keys.every((k) => { const q = P(k); return q && q[0] > 40 && q[0] < W - 40 && q[1] > 16 && q[1] < H - 16; });
    };
    let lo = 0.15, hi = 12;
    if (!ok(hi)) return hi;
    for (let n = 0; n < 30; n++) { const mid = (lo + hi) / 2; if (ok(mid)) hi = mid; else lo = mid; }
    return hi;
  }

  setData(data) {
    this.data = { summary: data.summary, t: data.frames.map((f) => f[0]), hands: data.frames.map((f) => f[1].map(unpack)) };
    this.setBounds(this.data.hands.flat().map((h) => h.wrist));
    this.message.textContent = "";
    this.render();
  }

  clear(text) { this.data = null; this.setBounds([]); this.message.textContent = text || ""; this.render(); }

  frameIndex(t) {
    const ts = this.data.t;
    let lo = 0, hi = ts.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (ts[mid] < t) lo = mid + 1; else hi = mid; }
    if (lo > 0 && Math.abs(ts[lo - 1] - t) < Math.abs(ts[lo] - t)) lo -= 1;
    return Math.abs(ts[lo] - t) <= 2.5 / (this.data.summary.rate_hz || 15) ? lo : -1;
  }

  camera(W, H, dist = this.view.dist) {
    const { yaw, pitch } = this.view;
    const eye = add(this.target, mul([Math.cos(pitch) * Math.sin(yaw), -Math.cos(pitch) * Math.cos(yaw), Math.sin(pitch)], dist));
    const fwd = norm(sub(this.target, eye));
    const right = norm(cross(fwd, [0, 0, 1]));
    const up = cross(right, fwd);
    const focal = Math.min(W, H) * 2.6;  // narrow virtual lens: farther eye, little perspective distortion
    return (p) => {
      const v = sub(p, eye), z = dot(v, fwd);
      if (z < 0.02) return null;
      return [W / 2 + (dot(v, right) * focal) / z, H / 2 - (dot(v, up) * focal) / z, z];
    };
  }

  render() {
    const c = this.canvas, dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth, H = c.clientHeight;
    if (!W || !H) return;
    if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
    const g = this.ctx;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    if (this.view.dist == null || (this.fitSize !== "manual" && this.fitSize !== `${W}x${H}`)) {
      this.view.dist = this.fit(W, H);  // refit when the view is reset or the canvas changes size
      this.fitSize = `${W}x${H}`;
    }
    const P = this.camera(W, H), C = this.colors, b = this.bounds;
    const line = (a, b, color, width = 1, dash = null) => {
      const pa = P(a), pb = P(b);
      if (!pa || !pb) return;
      g.strokeStyle = color; g.lineWidth = width; g.setLineDash(dash || []);
      g.beginPath(); g.moveTo(pa[0], pa[1]); g.lineTo(pb[0], pb[1]); g.stroke(); g.setLineDash([]);
    };
    const label = (p, text, color, dx = 4, dy = -4) => {
      const q = P(p); if (!q) return;
      g.fillStyle = color; g.font = "11px Segoe UI, system-ui, sans-serif"; g.fillText(text, q[0] + dx, q[1] + dy);
    };
    // floor grid (metres, every 10 cm) just below the lowest hands, with tick labels
    const F = b.floor, every = b.x1 - b.x0 > 0.8 || b.y1 > 0.8 ? 0.2 : 0.1;
    for (let x = b.x0; x <= b.x1 + 1e-6; x += 0.1) line([x, 0, F], [x, b.y1, F], C.grid);
    for (let y = 0; y <= b.y1 + 1e-6; y += 0.1) line([b.x0, y, F], [b.x1, y, F], C.grid);
    for (let x = b.x0; x <= b.x1 + 1e-6; x += every) label([x, b.y1, F], (Math.abs(x) < 1e-6 ? 0 : x).toFixed(1), C.muted, -8, 14);
    for (let y = every; y <= b.y1 + 1e-6; y += every) label([b.x0, y, F], y.toFixed(1), C.muted, -26, 4);
    label([b.x1, b.y1, F], "X (m)", C.muted, 14, 14);
    label([b.x0, b.y1, F], "Y (m)", C.muted, -30, -10);
    line([0, 0, F], [0, 0, 0], C.grid, 1, [3, 3]);
    label([0, 0, F], `Z ${F.toFixed(1)} m`, C.muted, 4, 12);
    // camera / head at the origin: small frustum + axes
    const corners = [[-0.05, 0.07, 0.03], [0.05, 0.07, 0.03], [0.05, 0.07, -0.03], [-0.05, 0.07, -0.03]];
    corners.forEach((k, i) => { line([0, 0, 0], k, C.camera, 1); line(k, corners[(i + 1) % 4], C.camera, 1); });
    [[[0.08, 0, 0], C.ax, "X"], [[0, 0.08, 0], C.ay, "Y"], [[0, 0, 0.08], C.az, "Z"]].forEach(([d, col, n]) => {
      line([0, 0, 0], d, col, 2); label(d, n, col);
    });
    const o = P([0, 0, 0]);
    if (o) { g.fillStyle = C.camera; g.beginPath(); g.arc(o[0], o[1], 6, 0, 7); g.fill(); label([0, 0, 0], "camera (head)", C.camera, 8, 14); }
    if (!this.data) { this.panelEmpty(); return; }
    const t = this.player.currentTime || 0, i = this.frameIndex(t);
    const hands = i >= 0 ? [...this.data.hands[i]] : [];
    // wrist trails over the last 2 s
    if (this.trails && i >= 0) {
      for (const side of ["left", "right"]) {
        let prev = null;
        for (let k = Math.max(0, i - 60); k <= i; k++) {
          if (this.data.t[k] < t - 2) { prev = null; continue; }
          const h = this.data.hands[k].find((x) => x.side === side);
          if (h && prev) line(prev, h.wrist, C[side] + "88", 1.5);
          prev = h ? h.wrist : null;
        }
      }
    }
    hands.sort((a, b) => (P(b.wrist)?.[2] || 0) - (P(a.wrist)?.[2] || 0));  // far hand first
    for (const h of hands) {
      const col = C[h.side];
      line(h.wrist, [h.wrist[0], h.wrist[1], F], col + "66", 1, [4, 4]);  // drop line to the floor
      const s = P([h.wrist[0], h.wrist[1], F]);
      if (s) { g.fillStyle = col + "55"; g.beginPath(); g.ellipse(s[0], s[1], 7, 3, 0, 0, 7); g.fill(); }
      EDGES.forEach(([a, b]) => line(h.kp[a], h.kp[b], col, 2));
      h.kp.forEach((p, j) => {
        const q = P(p); if (!q) return;
        g.fillStyle = col; g.strokeStyle = C.bg; g.lineWidth = 1.5;
        g.beginPath(); g.arc(q[0], q[1], j === 0 ? 6 : 3.2, 0, 7); g.fill(); g.stroke();
      });
      [[0, C.ax], [1, C.ay], [2, C.az]].forEach(([k, ac]) => line(h.wrist, add(h.wrist, mul(h.axes[k], 0.04)), ac, 2));
    }
    this.panelFrame(t, i, hands);
  }

  panelEmpty() {
    this.panel.innerHTML = '<div class="note">3D hand data appears here once it has been computed on the backend.</div>';
  }

  panelFrame(t, i, hands) {
    const s = this.data.summary, C = this.colors;
    const rows = ["left", "right"].map((side) => {
      const h = hands.find((x) => x.side === side);
      const head = `<h3><span class="swatch" style="background:${C[side]}"></span>${side} hand</h3>`;
      if (!h) return `${head}<div class="note">not detected at this moment</div>`;
      const n = h.axes[2];
      return `${head}<table class="spec"><tbody>
        <tr><td>wrist X / Y / Z</td><td>${h.wrist.map((v) => v.toFixed(2)).join(" / ")} m</td></tr>
        <tr><td>distance from head</td><td>${f2(h.dist)} m</td></tr>
        <tr><td>wrist speed</td><td>${f2(h.speed)} m/s</td></tr>
        <tr><td>grip aperture</td><td>${(h.aperture * 100).toFixed(1)} cm <span class="note">thumb–index</span></td></tr>
        <tr><td>palm normal</td><td>${n.map((v) => v.toFixed(2)).join(", ")}</td></tr>
        <tr><td>2D fit</td><td>${h.err.toFixed(2)}° <span class="note">consistency, not accuracy</span></td></tr>
        <tr><td>detection score</td><td>${h.score.toFixed(2)}</td></tr></tbody></table>`;
    }).join("");
    const md = s.median_dist_m || {};
    this.panel.innerHTML = `<div class="spec-head">t = ${t.toFixed(2)} s${i >= 0 ? ` · frame ${i + 1}/${this.data.t.length}` : " · no processed frame here"}</div>`
      + rows
      + `<h3><span class="swatch" style="background:${C.camera}"></span>camera (head)</h3>
         <table class="spec"><tbody><tr><td>position</td><td>0, 0, 0 (origin)</td></tr>
         <tr><td>axes</td><td>X right · Y forward · Z up</td></tr>
         <tr><td>lens (assumed)</td><td>${s.camera.lens.replace(" (assumed)", "")}, ${s.camera.hfov_deg}° FOV</td></tr></tbody></table>
         <h3>Clip</h3><table class="spec"><tbody>
         <tr><td>sample rate</td><td>${s.rate_hz} Hz · ${s.frames} frames</td></tr>
         <tr><td>frames with a hand</td><td>${s.frames_with_hand_pct}%</td></tr>
         <tr><td>median distance</td><td>${Object.entries(md).map(([k, v]) => `${k} ${v} m`).join(" · ") || "–"}</td></tr>
         <tr><td>median 2D fit</td><td>${s.median_fit_err_deg ?? "–"}°</td></tr></tbody></table>`;
  }
}
