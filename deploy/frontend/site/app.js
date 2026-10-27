const $ = (id) => document.getElementById(id);
const api = (p) => fetch("api/" + p, {cache: "no-store"}).then((r) => (r.ok ? r.json() : Promise.reject(r)));
const pct = (v) => (v == null ? "–" : (v * 100).toFixed(0) + "%");
const num = (v, d = 1) => (v == null ? "–" : Number(v).toFixed(d));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// links to the other tools on the same tailnet host
const L = window.ROOMSENSE_LINKS || {};
$("links").innerHTML = Object.entries(L)
  .map(([k, port]) => `<a href="https://${location.hostname}:${port}/" target="_blank" rel="noopener">${k}</a>`)
  .join("");

async function refreshLive() {
  try {
    const h = await api("health");
    $("health").textContent = h.model_loaded ? `model v${h.model_version}` : "no model yet";
    $("health").className = "pill " + (h.model_loaded && !h.camera_error ? "ok" : "bad");
    if (h.camera_error) $("health").textContent += " · camera error";
  } catch { $("health").textContent = "API offline"; $("health").className = "pill bad"; return; }
  try {
    const l = await api("latest");
    $("frame").src = "api/latest.jpg?" + Date.now();
    $("frameAge").textContent = Math.round(Date.now() / 1000 - l.timestamp) + " s ago";
    $("label").textContent = l.label.replaceAll("_", " ");
    $("conf").textContent = pct(l.confidence);
    $("confBar").style.width = pct(l.confidence);
    $("truth").textContent = l.truth ? `capture session says: ${l.truth.replaceAll("_", " ")} ${l.correct ? "✓" : "✗"}` : "";
    const probs = Object.entries(l.probabilities).sort((a, b) => b[1] - a[1]);
    $("probs").innerHTML = probs.map(([c, p], i) =>
      `<li class="${i === 0 ? "top" : ""}"><span>${c.replaceAll("_", " ")}</span><div class="bar"><i style="width:${p * 100}%"></i></div><span>${pct(p)}</span></li>`).join("");
  } catch { /* no camera prediction yet */ }
}

function drawHist(ref, cur) {
  const c = $("hist"), g = c.getContext("2d"), W = c.width, H = c.height;
  g.clearRect(0, 0, W, H);
  if (!ref) return;
  const n = ref.length, bw = W / n, max = Math.max(...ref, ...(cur || [0]), 0.01);
  for (let i = 0; i < n; i++) {
    const rh = (ref[i] / max) * (H - 20), ch = cur ? (cur[i] / max) * (H - 20) : 0;
    g.fillStyle = css("--mut"); g.globalAlpha = 0.45;
    g.fillRect(i * bw + 2, H - 16 - rh, bw / 2 - 2, rh);
    g.fillStyle = css("--acc"); g.globalAlpha = 1;
    g.fillRect(i * bw + bw / 2, H - 16 - ch, bw / 2 - 2, ch);
  }
  g.fillStyle = css("--mut"); g.font = "11px system-ui";
  g.fillText("dark", 2, H - 2); g.fillText("bright", W - 34, H - 2);
}

async function refreshDrift() {
  try {
    const d = await api("drift");
    $("ood").textContent = pct(d.ood_share);
    $("psi").textContent = num(d.psi, 2);
    $("thr").textContent = pct(d.threshold);
    $("win").textContent = d.window;
    const m = $("driftMsg");
    if (d.window < d.min_samples) { m.textContent = `Collecting frames (${d.window}/${d.min_samples})…`; m.className = "msg"; }
    else if (d.drifting) { m.textContent = "Drift: live frames no longer look like the training data."; m.className = "msg bad"; }
    else if (d.ood_share > 0.2) { m.textContent = "Some frames fall outside the training lighting."; m.className = "msg warn"; }
    else { m.textContent = "Stable: live frames match the training data."; m.className = "msg ok"; }
    $("bright").textContent = `${num(d.brightness_mean, 0)} / ${num(d.reference_brightness_mean, 0)}`;
    $("acc").textContent = d.live_accuracy == null ? "no capture session running" : `${pct(d.live_accuracy)} of ${d.labelled}`;
    drawHist(d.reference_hist, d.current_hist);
    $("confm").textContent = `${pct(d.confidence_mean)} / ${pct(d.reference_confidence_mean)}`;
    const r = await api("retrain");
    $("retrain").textContent = r.requested ? `requested: ${r.reason}` :
      (r.history && r.history.length ? `last done ${new Date(r.history.at(-1).done_at * 1000).toLocaleString()}` : "none");
  } catch { /* API offline */ }
}

async function refreshModel() {
  try {
    const m = await api("model");
    const rows = {
      "Version": m.version,
      "Classes": m.classes.join(", "),
      "Test macro-F1": num(m.metrics && m.metrics.test_macro_f1, 3),
      "Trained on": `${m.train_frames ?? "–"} frames ${m.data_include && Object.keys(m.data_include).length ? JSON.stringify(m.data_include) : "(all conditions)"}`,
      "Train brightness": num(m.reference_brightness_mean, 0),
      "Exported": m.created_at || "–",
      "Loaded": new Date(m.loaded_at * 1000).toLocaleString(),
    };
    $("model").innerHTML = Object.entries(rows).map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  } catch { $("model").innerHTML = "<dt>Status</dt><dd>No model loaded yet. Run the pipeline on the Legion.</dd>"; }
}

$("upload").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = $("file").files[0];
  if (!f) return;
  $("upPreview").src = URL.createObjectURL(f);
  const body = new FormData();
  body.append("file", f);
  const r = await fetch("api/predict", {method: "POST", body});
  const j = await r.json();
  $("upResult").textContent = r.ok
    ? `${j.label} (${pct(j.confidence)}) · ${num(j.latency_ms, 1)} ms · brightness ${num(j.brightness, 0)}`
    : JSON.stringify(j);
});

refreshLive(); refreshDrift(); refreshModel();
setInterval(refreshLive, 3000);
setInterval(refreshDrift, 5000);
setInterval(refreshModel, 30000);
