"use strict";

const POLL_MS = 4000;

let latestConfig = null;
let latestMetrics = null;
let togglePending = false;

const $ = (id) => document.getElementById(id);

function fmtInt(n) {
  if (n === null || n === undefined || isNaN(n)) return "–";
  return Math.round(n).toLocaleString("en-US");
}

function fmtPct(frac) {
  if (frac === null || frac === undefined || isNaN(frac)) return "–";
  return (frac * 100).toFixed(1) + "%";
}

function fmtDollars(n) {
  if (n === null || n === undefined || isNaN(n)) return "–";
  return "$" + n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(url + " -> " + res.status);
  return res.json();
}

function setStatus(state, text) {
  const dot = $("status-dot");
  dot.className = "dot dot-" + state;
  $("status-text").textContent = text;
}

async function refreshHealth() {
  try {
    const health = await getJSON("/health");
    setStatus("ok", "online · v" + (health.version || "?"));
    return true;
  } catch (e) {
    setStatus("down", "daemon unreachable");
    return false;
  }
}

async function refreshConfig() {
  try {
    latestConfig = await getJSON("/config");
    $("strategy").textContent = latestConfig.strategy || "–";
    $("embedding-model").textContent = latestConfig.embedding_model || "–";
    renderModules();
    renderTokens();
  } catch (e) {
    latestConfig = null;
  }
}

async function refreshMetrics() {
  try {
    latestMetrics = await getJSON("/metrics");
    renderTokens();
    renderModules();
    renderStore();
    renderGateway();
  } catch (e) {
    latestMetrics = null;
  }
}

function renderTokens() {
  const m = latestMetrics;
  if (!m) return;
  $("total-tokens").textContent = fmtInt(m.total_tokens_saved);
  $("cache-hit-rate").textContent = fmtPct(m.cache_hit_rate);
  $("compression-ratio").textContent =
    m.compression_ratio ? m.compression_ratio.toFixed(2) + "x" : "–";

  const cost = latestConfig ? latestConfig.cost_per_input_token : null;
  if (cost && m.total_tokens_saved) {
    $("dollars-saved").textContent = fmtDollars(m.total_tokens_saved * cost);
  } else {
    $("dollars-saved").textContent = "–";
  }
}

function moduleMetric(name) {
  if (!latestMetrics || !Array.isArray(latestMetrics.per_module)) return null;
  return latestMetrics.per_module.find((p) => p.module === name) || null;
}

function renderStore() {
  const list = $("store-list");
  const store = latestMetrics && latestMetrics.store;
  if (!store) {
    list.innerHTML = '<li class="muted">no data yet</li>';
    return;
  }
  const rows = [
    ["codebase symbols", store.codebase.symbols],
    ["codebase files", store.codebase.files],
    ["repos indexed", store.codebase.repos],
    ["cache entries", store.cache_entries],
    ["memory: files (tool + native sync)", store.memory.memory_files],
    ["memory: working sessions", store.memory.working_sessions],
    ["memory: episodic", store.memory.episodic],
    ["memory: semantic facts", store.memory.semantic_facts],
    ["retrieval corpus chunks", store.corpus_chunks],
  ];
  list.replaceChildren();
  for (const [label, value] of rows) {
    const li = document.createElement("li");
    li.className = "store-item";
    const name = document.createElement("span");
    name.textContent = label;
    const val = document.createElement("span");
    val.className = "store-value";
    val.textContent = fmtInt(value);
    li.append(name, val);
    list.appendChild(li);
  }
}

function renderGateway() {
  const list = $("gateway-list");
  const gw = latestMetrics && latestMetrics.gateway;
  if (!gw || gw.requests === 0) {
    list.innerHTML = '<li class="muted">no traffic yet — launch with <code>efficient claude</code> ' +
      'to route Claude Code through the gateway</li>';
    return;
  }
  const rows = [
    ["requests proxied", gw.requests],
    ["input tokens", gw.input_tokens],
    ["output tokens", gw.output_tokens],
    ["prompt-cache read tokens", gw.cache_read_tokens],
    ["prompt-cache created tokens", gw.cache_creation_tokens],
    ["duplicate requests (exact-cache potential)", gw.duplicate_requests],
  ];
  const health = [
    ["prompt-cache read ratio", fmtPct(gw.cache_read_ratio)],
    ["cache invalidations detected", fmtInt(gw.invalidations)],
    ["sessions observed", fmtInt(gw.sessions)],
  ];
  list.replaceChildren();
  for (const [label, value] of rows) {
    const li = document.createElement("li");
    li.className = "store-item";
    const name = document.createElement("span");
    name.textContent = label;
    const val = document.createElement("span");
    val.className = "store-value";
    val.textContent = fmtInt(value);
    li.append(name, val);
    list.appendChild(li);
  }
  for (const [label, formatted] of health) {
    const li = document.createElement("li");
    li.className = "store-item";
    const name = document.createElement("span");
    name.textContent = label;
    const val = document.createElement("span");
    val.className = "store-value";
    val.textContent = formatted;
    li.append(name, val);
    list.appendChild(li);
  }
}

function renderModules() {
  const list = $("module-list");
  if (!latestConfig || !latestConfig.modules) {
    list.innerHTML = '<li class="muted">config unavailable</li>';
    return;
  }
  const modules = latestConfig.modules;
  const total = latestMetrics ? latestMetrics.total_tokens_saved : 0;

  list.innerHTML = "";
  Object.keys(modules).forEach((name) => {
    const cfg = modules[name] || {};
    const enabled = !!cfg.enabled;
    const metric = moduleMetric(name);

    const li = document.createElement("li");
    li.className = "module-item";

    const head = document.createElement("div");
    head.className = "module-head";

    const nameSpan = document.createElement("span");
    nameSpan.className = "module-name";
    nameSpan.textContent = name;

    const btn = document.createElement("button");
    btn.className = "toggle " + (enabled ? "toggle-on" : "toggle-off");
    btn.textContent = enabled ? "ON" : "OFF";
    btn.disabled = togglePending;
    btn.addEventListener("click", () => toggleModule(name, !enabled));

    head.appendChild(nameSpan);
    head.appendChild(btn);
    li.appendChild(head);

    const tokens = document.createElement("div");
    tokens.className = "module-tokens";
    if (metric) {
      tokens.textContent =
        fmtInt(metric.tokens_saved) + " tokens · " + fmtInt(metric.events) + " events";
      li.appendChild(tokens);
      const pct = total > 0 ? Math.min(100, (metric.tokens_saved / total) * 100) : 0;
      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill";
      fill.style.width = pct.toFixed(1) + "%";
      track.appendChild(fill);
      li.appendChild(track);
    } else {
      tokens.textContent = "no data yet";
      li.appendChild(tokens);
    }

    list.appendChild(li);
  });
}

async function toggleModule(name, enabled) {
  if (togglePending) return;
  togglePending = true;
  renderModules();
  try {
    const patch = { modules: {} };
    patch.modules[name] = { enabled: enabled };
    const res = await fetch("/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    });
    if (!res.ok) throw new Error("PUT /config -> " + res.status);
    latestConfig = await res.json();
  } catch (e) {
    // leave latestConfig as-is; next poll reconciles
  } finally {
    togglePending = false;
    renderModules();
  }
}

function renderMemory(data) {
  const box = $("memory-results");
  const groups = [
    ["working", data.working],
    ["episodic", data.episodic],
    ["semantic", data.semantic],
  ];
  box.innerHTML = "";
  let any = false;
  groups.forEach(([label, items]) => {
    const arr = Array.isArray(items) ? items : [];
    const group = document.createElement("div");
    group.className = "memory-group";
    const h = document.createElement("h3");
    h.textContent = label + " (" + arr.length + ")";
    group.appendChild(h);
    if (arr.length === 0) {
      const p = document.createElement("p");
      p.className = "muted";
      p.textContent = "no results";
      group.appendChild(p);
    } else {
      any = true;
      const ul = document.createElement("ul");
      arr.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = typeof item === "string" ? item : JSON.stringify(item, null, 2);
        ul.appendChild(li);
      });
      group.appendChild(ul);
    }
    box.appendChild(group);
  });
  if (!any) {
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent = "No memory found for that agent/query.";
    box.prepend(note);
  }
}

async function searchMemory() {
  const btn = $("memory-search");
  const box = $("memory-results");
  const agentId = $("agent-id").value.trim() || "default";
  const query = $("memory-query").value.trim();
  btn.disabled = true;
  box.innerHTML = '<p class="muted">searching…</p>';
  try {
    const res = await fetch("/memory/retrieve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agent_id: agentId, query: query }),
    });
    if (!res.ok) throw new Error("POST /memory/retrieve -> " + res.status);
    renderMemory(await res.json());
  } catch (e) {
    box.innerHTML = "";
    const p = document.createElement("p");
    p.className = "error";
    p.textContent = "Memory retrieval failed: " + e.message;
    box.appendChild(p);
  } finally {
    btn.disabled = false;
  }
}

let activitySince = 0;
const activityLog = [];
const ACTIVITY_MAX_ROWS = 50;

function renderActivity(active) {
  const ops = $("active-ops");
  ops.replaceChildren();
  for (const a of active) {
    const li = document.createElement("li");
    li.className = "active-op";
    li.textContent = a.message + " · " + a.elapsed_s + "s";
    ops.appendChild(li);
  }
  const log = $("activity-log");
  log.replaceChildren();
  if (activityLog.length === 0) {
    const li = document.createElement("li");
    li.className = "muted";
    li.textContent = "no activity yet";
    log.appendChild(li);
    return;
  }
  for (const e of activityLog) {
    const li = document.createElement("li");
    li.className = "activity-event" + (e.level === "error" ? " activity-error" : "");
    const ts = new Date(e.ts).toLocaleTimeString();
    li.textContent = ts + "  " + e.message;
    log.appendChild(li);
  }
}

async function refreshActivity() {
  try {
    const status = await getJSON("/status?since=" + activitySince);
    for (const e of status.events) activityLog.unshift(e);
    activityLog.length = Math.min(activityLog.length, ACTIVITY_MAX_ROWS);
    activitySince = status.last_seq;
    renderActivity(status.active);
  } catch (e) {
    /* daemon down — health dot already reports it */
  }
}

async function poll() {
  const up = await refreshHealth();
  if (up) {
    await Promise.all([refreshConfig(), refreshMetrics(), refreshActivity()]);
    $("last-updated").textContent = "updated " + new Date().toLocaleTimeString();
  }
}

function init() {
  $("memory-search").addEventListener("click", searchMemory);
  $("memory-query").addEventListener("keydown", (e) => {
    if (e.key === "Enter") searchMemory();
  });
  poll();
  setInterval(poll, POLL_MS);
}

document.addEventListener("DOMContentLoaded", init);
