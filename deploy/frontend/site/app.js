// Live room page: latest camera prediction, drift monitor, serving model, photo upload.

// What each room state means, in everyday words.
const MEANING = {
  at_desk: "Someone is sitting at the desk.",
  lying_in_bed: "Someone is lying down in bed.",
  sitting_on_bed: "Someone is sitting up on the bed.",
  in_room: "Someone is in the room, standing or moving about.",
  empty: "Nobody is in the room and it is lit.",
  lights_off: "The room is dark.",
};
const MAX_UPLOAD = 15 * 1024 * 1024;

async function refreshLive() {
  const h = await refreshStatus();
  if (!h) return;
  if (!h.model_loaded) {
    $("label").textContent = "No model yet";
    $("meaning").innerHTML = 'RoomSense has not been trained yet. Once a model is trained it appears here by itself. <a href="pipeline.html">Open the Pipeline page</a>';
  }
  try {
    const l = await api("latest");
    $("frame").src = API_BASE + "latest.jpg?" + Date.now();
    $("frame").hidden = false;
    $("frameEmpty").hidden = true;
    const age = Math.max(0, Math.round(Date.now() / 1000 - l.timestamp));
    $("frameAge").hidden = false;
    $("frameAge").textContent = age < 15 ? "live" : age + " s ago";
    $("frameAge").className = age > 60 ? "stale" : "";
    $("label").textContent = nice(l.label);
    $("meaning").textContent = (MEANING[l.label] || "") + (l.confidence < 0.6 ? " The model is not very sure about this one." : "");
    $("conf").textContent = pct(l.confidence);
    $("confBar").style.width = pct(l.confidence);
    $("confBar").className = l.confidence < 0.6 ? "low" : "";
    $("truth").textContent = l.truth ? `You are recording "${nice(l.truth)}" right now: the model is ${l.correct ? "right ✓" : "wrong ✗"}` : "";
    const probs = Object.entries(l.probabilities).sort((a, b) => b[1] - a[1]);
    $("probs").innerHTML = probs.map(([c, p], i) =>
      `<li class="${i === 0 ? "top" : ""}"><span>${esc(nice(c))}</span><div class="bar"><i style="width:${p * 100}%"></i></div><span>${pct(p)}</span></li>`).join("");
  } catch (e) {
    if (e.offline) return;
    // 404: the model or the camera has not produced anything yet
    $("frameEmpty").textContent = h.camera_error
      ? "The camera is not answering. Check that the phone is charging, on Wi-Fi, and that IP Webcam is running."
      : h.camera_enabled === false ? "The live camera is switched off on this server." : "Waiting for the first picture from the camera…";
    if (h.camera_error) { $("frame").hidden = true; $("frameEmpty").hidden = false; $("frameAge").hidden = true; }
  }
}

function drawHist(ref, cur) {
  const c = $("hist"), g = c.getContext("2d"), W = c.width, H = c.height;
  g.clearRect(0, 0, W, H);
  if (!ref) return;
  const n = ref.length, bw = W / n, max = Math.max(...ref, ...(cur || [0]), 0.01);
  for (let i = 0; i < n; i++) {
    const rh = (ref[i] / max) * (H - 22), ch = cur ? (cur[i] / max) * (H - 22) : 0;
    g.fillStyle = cssVar("--mut"); g.globalAlpha = 0.45;
    g.fillRect(i * bw + 2, H - 18 - rh, bw / 2 - 2, rh);
    g.fillStyle = cssVar("--acc"); g.globalAlpha = 1;
    g.fillRect(i * bw + bw / 2, H - 18 - ch, bw / 2 - 2, ch);
  }
  g.fillStyle = cssVar("--mut"); g.font = "12px system-ui";
  g.fillText("dark", 2, H - 3); g.fillText("bright", W - 38, H - 3);
}

async function refreshDrift() {
  try {
    const d = await api("drift");
    $("ood").textContent = pct(d.ood_share);
    $("psi").textContent = num(d.psi, 2);
    $("thr").textContent = pct(d.threshold);
    $("win").textContent = d.window;
    const m = $("driftMsg");
    if (d.ood_share == null) { m.textContent = "Nothing to compare yet: this starts once a model is loaded and the camera is sending pictures."; m.className = "msg"; }
    else if (d.window < d.min_samples) { m.textContent = `Collecting pictures (${d.window} of ${d.min_samples} needed)…`; m.className = "msg"; }
    else if (d.drifting) { m.textContent = "The room looks different from what the model learned (for example the light changed). Retraining is requested automatically."; m.className = "msg bad"; }
    else if (d.ood_share > 0.2) { m.textContent = "Some pictures are in lighting the model has rarely seen. Still fine."; m.className = "msg warn"; }
    else { m.textContent = "All good: the room looks like what the model learned."; m.className = "msg ok"; }
    $("bright").textContent = `${num(d.brightness_mean, 0)} / ${num(d.reference_brightness_mean, 0)} (0 = black, 255 = white)`;
    $("acc").textContent = d.live_accuracy == null ? "no recording session is running" : `${pct(d.live_accuracy)} right, of ${d.labelled} pictures`;
    drawHist(d.reference_hist, d.current_hist);
    $("confm").textContent = `${pct(d.confidence_mean)} / ${pct(d.reference_confidence_mean)}`;
    const r = await api("retrain");
    $("retrain").textContent = r.requested ? `requested ${ago(r.requested_at)}: ${r.reason}` :
      (r.history && r.history.length ? `last finished ${ago(r.history.at(-1).done_at)}` : "none so far");
  } catch { /* the status pill already says the API is offline */ }
}

async function refreshModel() {
  try {
    const m = await api("model");
    const filtered = m.data_include && Object.keys(m.data_include).length;
    const rows = {
      "Version": "v" + m.version,
      "Room states it knows": m.classes.map(nice).join(", "),
      "Score on unseen pictures": m.metrics && m.metrics.test_macro_f1 != null ? `${num(m.metrics.test_macro_f1, 3)} macro-F1 (1.0 is perfect)` : "–",
      "Learned from": `${int(m.train_frames)} pictures ${filtered ? "(only " + esc(JSON.stringify(m.data_include)) + ")" : "(all lighting conditions)"}`,
      "Trained": m.created_at ? new Date(m.created_at).toLocaleString([], {dateStyle: "medium", timeStyle: "short"}) : "–",
      "In use since": when(m.loaded_at),
    };
    $("model").innerHTML = Object.entries(rows).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("");
  } catch (e) {
    if (e.offline) return;
    $("model").innerHTML = "<dt>Status</dt><dd>No model is loaded yet. It appears here by itself after the first training run.</dd>";
  }
}

// --- photo upload: validate first, explain every failure in plain words ---
const fileInput = $("file"), upError = $("upError");
function showUploadError(text) { upError.textContent = text; upError.hidden = false; $("upOut").hidden = true; }

function pickFile(f) {
  upError.hidden = true;
  $("go").disabled = true;
  if (!f) { $("dropText").innerHTML = "<b>Choose a photo</b> or drop one here (JPEG or PNG, up to 15 MB)"; return; }
  if (!/^image\/(jpeg|png|webp)$/.test(f.type)) return showUploadError("That file is not a photo. Please choose a JPEG or PNG picture.");
  if (f.size > MAX_UPLOAD) return showUploadError("That photo is larger than 15 MB. Please choose a smaller one.");
  $("dropText").innerHTML = `<b>${esc(f.name)}</b> · ready`;
  $("go").disabled = false;
}
fileInput.addEventListener("change", () => pickFile(fileInput.files[0]));
const drop = $("drop");
["dragover", "dragenter"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; pickFile(fileInput.files[0]); }
});

$("upload").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = fileInput.files[0];
  if (!f) return;
  const go = $("go");
  go.disabled = true; go.textContent = "Looking…"; upError.hidden = true;
  try {
    const body = new FormData();
    body.append("file", f);
    const j = await api("predict", {method: "POST", body});
    $("upPreview").src = URL.createObjectURL(f);
    $("upLabel").textContent = nice(j.label);
    $("upDetail").textContent = `${MEANING[j.label] || ""} ${pct(j.confidence)} sure. Answered in ${num(j.latency_ms, 1)} ms by model v${j.model_version}.`;
    $("upOut").hidden = false;
  } catch (err) {
    showUploadError(err.status === 400 ? "RoomSense could not read that file as a picture. Try another photo."
      : err.status === 413 ? "That photo is too large. Please choose one under 15 MB."
      : err.status === 503 && !err.offline ? "No model is loaded yet, so there is nothing to ask. Try again after the first training run."
      : err.message);
  } finally {
    go.disabled = false; go.textContent = "What does it see?";
  }
});

refreshLive(); refreshDrift(); refreshModel();
setInterval(refreshLive, 3000);
setInterval(refreshDrift, 5000);
setInterval(refreshModel, 30000);
document.addEventListener("roomsense:online", () => { refreshDrift(); refreshModel(); });
