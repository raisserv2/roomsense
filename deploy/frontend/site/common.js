// Shared by every page: the REST client, the header (navigation, status pill, tool links) and small helpers.
// The backend address comes from config.js (generated from environment variables), never from this file.
const CFG = window.ROOMSENSE_CONFIG || {apiBase: "api/", links: {}};
const API_BASE = CFG.apiBase.endsWith("/") ? CFG.apiBase : CFG.apiBase + "/";

const $ = (id) => document.getElementById(id);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const pct = (v, d = 0) => (v == null ? "–" : (v * 100).toFixed(d) + "%");
const num = (v, d = 1) => (v == null ? "–" : Number(v).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
const int = (v) => (v == null ? "–" : Math.round(v).toLocaleString());
const nice = (label) => String(label ?? "").replaceAll("_", " ");
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function duration(seconds) {
  if (seconds == null) return "–";
  if (seconds < 1) return Math.round(seconds * 1000) + " ms";
  if (seconds < 90) return seconds.toFixed(seconds < 10 ? 1 : 0) + " s";
  if (seconds < 5400) return Math.round(seconds / 60) + " min";
  return (seconds / 3600).toFixed(1) + " h";
}

function ago(epochSeconds) {
  if (!epochSeconds) return "–";
  const s = Date.now() / 1000 - epochSeconds;
  if (s < 0) return "just now";
  if (s < 60) return Math.round(s) + " s ago";
  if (s < 3600) return Math.round(s / 60) + " min ago";
  if (s < 86400 * 2) return Math.round(s / 3600) + " h ago";
  return Math.round(s / 86400) + " days ago";
}

const when = (epochSeconds) => (epochSeconds ? new Date(epochSeconds * 1000).toLocaleString([], {dateStyle: "medium", timeStyle: "short"}) : "–");

// One REST call. Rejects with an Error whose message is fit to show to a person.
async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(API_BASE + path, {cache: "no-store", ...options});
  } catch {
    const e = new Error("Can't reach RoomSense. Check that you are on the Tailscale network.");
    e.offline = true;
    throw e;
  }
  let body = null;
  try { body = await response.json(); } catch { /* not JSON */ }
  if (!response.ok) {
    const e = new Error((body && body.detail) || `The server answered with an error (${response.status}).`);
    e.status = response.status;
    e.offline = response.status >= 502;
    throw e;
  }
  return body;
}

// Header: which page is active, links to the other tools, overall status.
function setupHeader() {
  const here = (location.pathname.split("/").pop() || "index.html").replace(".html", "") || "index";
  document.querySelectorAll("nav.pages a").forEach((a) => {
    if (a.dataset.page === here) a.setAttribute("aria-current", "page");
  });
  const tools = $("tools");
  if (tools) {
    tools.innerHTML = Object.entries(CFG.links || {})
      .map(([name, port]) => `<a href="https://${esc(location.hostname)}:${esc(port)}/" target="_blank" rel="noopener">${esc(name)} ↗</a>`)
      .join("");
  }
}

let wasOffline = false;
async function refreshStatus() {
  const pill = $("health"), banner = $("offline");
  try {
    const h = await api("health");
    const ok = h.model_loaded && !h.camera_error;
    pill.className = "pill " + (ok ? "ok" : "warn");
    pill.textContent = !h.model_loaded ? "No model yet" : h.camera_error ? `Model v${h.model_version} · camera offline` : `Working · model v${h.model_version}`;
    pill.title = h.model_error || h.camera_error || "Everything is running";
    if (banner) banner.hidden = true;
    if (wasOffline) document.dispatchEvent(new Event("roomsense:online"));
    wasOffline = false;
    return h;
  } catch (e) {
    pill.className = "pill bad";
    pill.textContent = "Offline";
    pill.title = e.message;
    if (banner) { banner.hidden = false; banner.querySelector("span").textContent = e.message + " Retrying automatically…"; }
    wasOffline = true;
    return null;
  }
}

setupHeader();
