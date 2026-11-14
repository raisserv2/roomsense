// Pipeline page: DAG with per-stage status and throughput, controls, ingestion, run console, versions, activity.
// Everything comes from the REST API (/pipeline, /pipeline/runs, /pipeline/registry, /events).

const STATUS_WORD = {ok: "ran", cached: "unchanged", failed: "failed", not_reached: "not reached", running: "running", live: "running"};
let overview = null, runsData = null, registryData = null, showTab = "pipeline", selectedStage = null;

const shortId = (id) => (id ? String(id).slice(0, 8) : "–");
const tile = (label, value, sub = "", cls = "") =>
  `<div class="tile ${cls}"><span>${esc(label)}</span><b>${value}</b><small>${sub}</small></div>`;

function stageStatus(name) {
  if (name === "capture") return overview.ingestion && overview.ingestion.available ? {status: "live"} : {status: "not_reached"};
  if (name === "serve") return {status: registryData && registryData.serving ? "live" : "not_reached"};
  const latest = overview.latest;
  return (latest && latest.stages[name]) || {status: "not_reached"};
}

function stageFigure(name, st) {
  if (name === "capture") {
    const i = overview.ingestion || {};
    return i.available ? `${int(i.frames_total)} pictures` : "no data";
  }
  if (name === "serve") return registryData && registryData.serving ? `model v${esc(registryData.serving)}` : "no model";
  const parts = [];
  if (st.seconds != null) parts.push(duration(st.seconds));
  if (st.throughput_per_s != null) parts.push(`${num(st.throughput_per_s, st.throughput_per_s < 10 ? 1 : 0)}/s`);
  return parts.join(" · ") || "–";
}

function renderOverview() {
  const o = overview, s = o.summary, latest = o.latest, best = o.last_success || latest;
  $("fresh").textContent = "updated " + ago(o.generated_at);
  const notice = $("notice");
  notice.hidden = !o.errors.length;
  notice.textContent = o.errors.join(" ") + (o.errors.length ? " Showing the last information received." : "");

  const head = (best && best.headline) || {};
  const lastTile = !latest
    ? tile("Latest run", "none yet", "run the pipeline once on the training computer")
    : tile("Latest run", latest.ok ? "✓ succeeded" : latest.status === "FAILED" ? "✗ failed" : "… running",
           `${ago(latest.started_at)}${latest.failed_stage ? " · at " + esc(latest.failed_stage) : ""}`, latest.ok ? "ok" : latest.status === "FAILED" ? "bad" : "");
  $("tiles").innerHTML = [
    lastTile,
    tile("Runs recorded", `${s.runs_ok} ✓ &nbsp; ${s.runs_failed} ✗`, s.success_rate == null ? "" : pct(s.success_rate) + " succeeded"),
    tile("Whole pipeline took", best ? duration(best.seconds) : "–", "latest successful run"),
    tile("Data engineering speed", head.ray_resize_fps != null ? int(head.ray_resize_fps) + " pictures/s" : "–", "Ray resize step" + (head.preprocess_fps != null ? ` · ${int(head.preprocess_fps)}/s end to end` : "")),
    tile("Training speed", head.train_images_per_s != null ? int(head.train_images_per_s) + " pictures/s" : "–", "while training"),
    tile("One prediction takes", head.latency_ms_p95 != null ? num(head.latency_ms_p95, 1) + " ms" : "–", "95th percentile, CPU"),
  ].join("");

  const names = ["capture", ...o.dag.stages.map((x) => x.name), "serve"];
  $("dag").innerHTML = names.map((name) => {
    const info = o.stage_info[name] || {title: name, tool: "", machine: ""};
    const st = stageStatus(name);
    return `<li><button type="button" class="node ${esc(st.status)}${selectedStage === name ? " sel" : ""}" data-stage="${esc(name)}" aria-pressed="${selectedStage === name}">
      <span class="where">${esc(info.machine)}</span>
      <b>${esc(info.title)}</b>
      <span class="tool">${esc(info.tool)}</span>
      <span class="st"><i class="dot ${esc(st.status)}"></i>${esc(STATUS_WORD[st.status] || st.status)}</span>
      <span class="fig">${stageFigure(name, st)}</span>
    </button></li>`;
  }).join("");
  const chain = new Set(o.dag.stages.slice(1).map((x, i) => o.dag.stages[i].name + ">" + x.name));
  const extra = o.dag.edges.filter(([a, b]) => !chain.has(a + ">" + b)).map(([a, b]) => `${a} → ${b}`);
  $("dagExtra").textContent = extra.length ? "Steps also reuse earlier results directly: " + extra.join(", ") + "." : "";
  document.querySelectorAll("#dag .node").forEach((b) => b.addEventListener("click", () => {
    selectedStage = selectedStage === b.dataset.stage ? null : b.dataset.stage;
    renderOverview();
  }));
  renderStageDetail();
  renderIngestion();
  renderRetrain();
}

function renderStageDetail() {
  const box = $("stageDetail");
  if (!selectedStage) { box.hidden = true; return; }
  const info = overview.stage_info[selectedStage] || {}, st = stageStatus(selectedStage);
  const spec = overview.dag.stages.find((x) => x.name === selectedStage) || {};
  const rows = [["What it does", esc(info.what || "")], ["Runs on", esc(`${info.machine || ""} · ${info.tool || ""}`)],
                ["Latest run", esc(STATUS_WORD[st.status] || st.status) + (st.seconds != null ? ` in ${duration(st.seconds)}` : "")]];
  if (st.items != null) rows.push(["Processed", `${int(st.items)} items` + (st.throughput_per_s != null ? ` at ${num(st.throughput_per_s, 1)} per second` : "")]);
  if (st.status === "failed" && overview.latest && overview.latest.error) rows.push(["Error", `<code>${esc(overview.latest.error)}</code>`]);
  if (spec.cmd) rows.push(["Command", `<code>${esc(spec.cmd)}</code>`]);
  if (spec.deps && spec.deps.length) rows.push(["Reads", esc(spec.deps.join(", "))]);
  if (spec.outs && spec.outs.length) rows.push(["Writes", esc(spec.outs.join(", "))]);
  box.innerHTML = `<h3>${esc(info.title || selectedStage)}</h3><dl class="kv">${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>`;
  box.hidden = false;
}

function renderIngestion() {
  const i = overview.ingestion || {};
  if (!i.available) {
    $("ingest").innerHTML = `<p class="msg warn">The capture service is not answering, so the totals can't be shown right now.</p>`;
    return;
  }
  const classes = Object.entries(i.per_class || {}).sort((a, b) => b[1].frames - a[1].frames);
  const max = Math.max(1, ...classes.map(([, c]) => c.frames));
  $("ingest").innerHTML = `
    <div class="tiles small">
      ${tile("Pictures", int(i.frames_total), `${int(i.frames_today)} today`)}
      ${tile("Recording sessions", int(i.sessions_total), i.last_frame_at ? "last picture " + esc(new Date(i.last_frame_at).toLocaleString([], {dateStyle: "medium", timeStyle: "short"})) : "")}
      ${tile("Right now", i.recording ? "● recording" : "idle", i.recording ? esc(nice(i.recording.label)) : `a picture every ${num(i.interval_seconds, 0)} s when recording`, i.recording ? "ok" : "")}
    </div>
    <ul class="probs wide">${classes.map(([c, v]) =>
      `<li><span>${esc(nice(c))}</span><div class="bar"><i style="width:${(v.frames / max) * 100}%"></i></div><span>${int(v.frames)} · ${v.sessions} sessions</span></li>`).join("")}</ul>
    <p class="hint">Sessions matter more than pictures: a class needs several separate sessions so that some can be kept aside for testing.</p>`;
}

function renderRetrain() {
  const r = overview.retrain, m = $("retrainState");
  if (r.requested) { m.className = "msg warn"; m.textContent = `Retraining was requested ${ago(r.requested_at)} (${r.reason}). Waiting for the training computer.`; }
  else if (r.history && r.history.length) {
    const h = r.history.at(-1);
    m.className = "msg ok";
    m.textContent = `No retraining is waiting. The last one finished ${ago(h.done_at)}${h.version ? ` with model v${h.version}` : ""}${h.promoted === false ? " (not better, so the old model stayed)" : ""}.`;
  } else { m.className = "msg"; m.textContent = "No retraining is waiting."; }
  $("btnRetrain").disabled = !!r.requested;
}

function renderRuns() {
  if (!runsData) return;
  const box = $("runs");
  if (showTab === "pipeline") {
    const rows = runsData.pipeline_runs;
    if (!rows.length) { box.innerHTML = `<p class="msg">No pipeline run has been recorded yet. On the training computer run <code>python scripts/run_pipeline.py</code>; every run then shows up here, including failed ones.</p>`; return; }
    box.innerHTML = `<table><thead><tr><th>Result</th><th>Started</th><th>Took</th><th>Why it ran</th><th>Stages</th><th>Model</th><th>Code · data</th><th>Run ID</th></tr></thead><tbody>${rows.map((r) => `
      <tr class="${r.ok ? "" : r.status === "FAILED" ? "failed" : ""}">
        <td><span class="tag ${r.ok ? "ok" : r.status === "FAILED" ? "bad" : ""}">${r.ok ? "✓ succeeded" : r.status === "FAILED" ? "✗ failed" : esc(r.status.toLowerCase())}</span></td>
        <td title="${esc(when(r.started_at))}">${esc(ago(r.started_at))}</td>
        <td>${duration(r.seconds)}</td>
        <td>${esc(r.trigger || "manual")}${r.reason ? `<small>${esc(r.reason)}</small>` : ""}</td>
        <td>${Object.entries(r.stages).map(([n, s]) => `<i class="dot ${esc(s.status)}" title="${esc(n)}: ${esc(STATUS_WORD[s.status] || s.status)}${s.seconds != null ? " · " + duration(s.seconds) : ""}"></i>`).join("")}
            ${r.failed_stage ? `<small>failed at <b>${esc(r.failed_stage)}</b>${r.error ? ": " + esc(String(r.error).slice(0, 160)) : ""}</small>` : ""}</td>
        <td>${r.model_version ? "v" + esc(r.model_version) + (r.promoted === "true" ? " ★" : "") : "–"}${r.headline.test_macro_f1 != null ? `<small>F1 ${num(r.headline.test_macro_f1, 3)}</small>` : ""}</td>
        <td><code title="Git commit">${esc(shortId(r.git_commit))}${r.git_dirty ? "*" : ""}</code> <code title="DVC data version">${esc(shortId(r.data_version))}</code></td>
        <td><code title="${esc(r.run_id)}">${esc(shortId(r.run_id))}</code></td>
      </tr>`).join("")}</tbody></table>`;
  } else {
    const rows = runsData.training_runs;
    if (!rows.length) { box.innerHTML = `<p class="msg">No training run has been recorded yet.</p>`; return; }
    box.innerHTML = `<table><thead><tr><th>Result</th><th>Started</th><th>Took</th><th>Test macro-F1</th><th>Validation</th><th>Pictures</th><th>Code · data</th><th>MLflow run ID</th></tr></thead><tbody>${rows.map((r) => `
      <tr class="${r.status === "FAILED" ? "failed" : ""}">
        <td><span class="tag ${r.status === "FINISHED" ? "ok" : r.status === "FAILED" ? "bad" : ""}">${r.status === "FINISHED" ? "✓ finished" : r.status === "FAILED" ? "✗ failed" : esc(r.status.toLowerCase())}</span></td>
        <td title="${esc(when(r.started_at))}">${esc(ago(r.started_at))}</td>
        <td>${duration(r.seconds)}</td>
        <td><b>${num(r.test_macro_f1, 3)}</b></td>
        <td>${num(r.best_val_macro_f1, 3)}</td>
        <td>${esc(r.train_frames ?? "–")}</td>
        <td><code title="Git commit">${esc(shortId(r.git_commit))}</code> <code title="DVC data version">${esc(shortId(r.data_version))}</code></td>
        <td><code title="${esc(r.run_id)}">${esc(shortId(r.run_id))}</code></td>
      </tr>`).join("")}</tbody></table>
      <p class="hint">Any run can be reproduced from its Git commit and its MLflow run ID. Full parameters, curves and files are in MLflow (link at the top).</p>`;
  }
}

function renderRegistry() {
  if (!registryData) return;
  const v = registryData.versions, box = $("versions");
  const btn = $("btnRollback");
  btn.disabled = !registryData.rollback_target;
  $("rollbackHint").textContent = registryData.rollback_target
    ? `Puts model v${registryData.rollback_target} back in use instead of v${registryData.champion}. Nothing is deleted.`
    : "There is no earlier model to go back to yet.";
  if (!v.length) { box.innerHTML = `<p class="msg">No model has been registered yet.</p>`; return; }
  box.innerHTML = `<table><thead><tr><th>Version</th><th>Test macro-F1</th><th>Decision</th><th>Registered</th><th></th></tr></thead><tbody>${v.map((m) => `
    <tr>
      <td><b>v${esc(m.version)}</b> ${m.is_champion ? '<span class="tag ok">in use</span>' : ""}${m.rolled_back_at && !m.is_champion ? '<span class="tag bad">rolled back</span>' : ""}</td>
      <td>${num(m.test_macro_f1, 3)}</td>
      <td>${esc(m.decision || "–")}</td>
      <td title="${esc(when(m.created_at))}">${esc(ago(m.created_at))}</td>
      <td>${m.is_champion ? "" : `<button type="button" class="quiet tiny" data-version="${esc(m.version)}">Use this one</button>`}</td>
    </tr>`).join("")}</tbody></table>`;
  box.querySelectorAll("button[data-version]").forEach((b) => b.addEventListener("click", () => askRollback(b.dataset.version)));
}

async function renderEvents() {
  try {
    const {events} = await api("events?limit=30");
    $("events").innerHTML = events.length ? events.map((e) => `
      <li class="${esc(e.level)}"><i class="dot ${e.level === "error" ? "failed" : e.level === "warning" ? "cached" : e.level === "success" ? "ok" : "idle"}"></i>
        <span>${esc(e.message)}${e.count > 1 ? ` <small>×${e.count}</small>` : ""}</span><time title="${esc(when(e.last_at))}">${esc(ago(e.last_at))}</time></li>`).join("")
      : `<li class="msg">Nothing has happened yet. Alerts, retraining and model changes will be listed here.</li>`;
  } catch { /* offline: the banner says so */ }
}

// --- controls ---------------------------------------------------------------
function say(text, cls = "ok") { const m = $("actionMsg"); m.textContent = text; m.className = "msg " + cls; m.hidden = false; }

async function act(path, button, okText) {
  const headers = $("token").value ? {"X-Token": $("token").value} : {};
  button.disabled = true;
  try {
    const r = await api(path, {method: "POST", headers});
    say(typeof okText === "function" ? okText(r) : okText);
    await refreshAll(true);
  } catch (e) {
    if (e.status === 403) { $("tokenRow").hidden = false; say("This server wants the admin token for that. Type it below and try again.", "warn"); $("token").focus(); }
    else say(e.message, "bad");
  } finally { button.disabled = false; }
}

let pendingVersion = null;
function askRollback(version) {
  pendingVersion = version || registryData.rollback_target;
  $("confirmText").textContent = `Put model v${pendingVersion} back in use instead of v${registryData.champion}? The live page switches within a few seconds. You can switch again at any time.`;
  $("confirm").hidden = false;
  $("confirmYes").focus();
}
$("btnRetrain").addEventListener("click", (e) => act("retrain/request?reason=" + encodeURIComponent("asked from the web page"), e.currentTarget,
  (r) => (r.raised ? "Retraining requested. It starts the next time the training computer checks." : "Retraining was already requested.")));
$("btnReload").addEventListener("click", (e) => act("reload", e.currentTarget, (r) => (r.error ? "Could not check: " + r.error : r.reloaded ? `Now using model v${r.version}.` : `Already on the newest model (v${r.version ?? "none"}).`)));
$("btnRollback").addEventListener("click", () => askRollback());
$("confirmNo").addEventListener("click", () => { $("confirm").hidden = true; });
$("confirmYes").addEventListener("click", async (e) => {
  $("confirm").hidden = true;
  await act("pipeline/rollback?version=" + encodeURIComponent(pendingVersion), e.currentTarget, (r) => `Model v${r.champion} is in use again (was v${r.rolled_back_from}).`);
});
[["tabPipe", "pipeline"], ["tabTrain", "training"]].forEach(([id, tab]) => $(id).addEventListener("click", () => {
  showTab = tab;
  $("tabPipe").setAttribute("aria-selected", tab === "pipeline");
  $("tabTrain").setAttribute("aria-selected", tab === "training");
  renderRuns();
}));

async function refreshAll(force = false) {
  const h = await refreshStatus();
  if (!h) return;
  try {
    [overview, runsData, registryData] = await Promise.all([
      api("pipeline" + (force ? "?refresh=true" : "")), api("pipeline/runs?limit=25"), api("pipeline/registry")]);
    renderOverview(); renderRuns(); renderRegistry();
    await renderEvents();
  } catch (e) {
    if (!e.offline) { const n = $("notice"); n.hidden = false; n.textContent = "The pipeline information could not be loaded: " + e.message; }
  }
}

refreshAll();
setInterval(refreshAll, 15000);
