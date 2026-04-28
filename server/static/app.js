// Steno Server SPA — vanilla JS, hash-routed.

const SUPPORTED_EXT = ["wav", "mp3", "m4a", "ogg", "flac", "webm", "mp4"];
const STORAGE_KEY = "steno-server.lang";
const MAX_MB = 500; // matches default; could fetch from /api/health

// State -------------------------------------------------------------------

const state = {
  lang: localStorage.getItem(STORAGE_KEY) || (navigator.language || "").startsWith("en") ? "en" : "es",
  i18n: {},
  selectedFile: null,
  ws: null,
};

// i18n --------------------------------------------------------------------

function tFmt(text, vars) {
  if (!vars) return text;
  return text.replace(/\{(\w+)\}/g, (_, k) => (k in vars ? vars[k] : `{${k}}`));
}

function t(key, vars) {
  const value = state.i18n[key] ?? key;
  return tFmt(value, vars);
}

async function loadI18n(lang) {
  const r = await fetch(`/api/i18n/${lang}`);
  if (!r.ok) return;
  state.i18n = await r.json();
  state.lang = lang;
  document.documentElement.lang = lang;
  applyI18n();
  // Update the supported-formats line with max_mb interpolated.
  const supEl = document.getElementById("upload-supported");
  if (supEl) supEl.textContent = t("upload_supported_formats", { max_mb: MAX_MB });
}

function applyI18n() {
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    const text = state.i18n[key];
    if (text) el.textContent = text;
  });
}

// Routing -----------------------------------------------------------------

function route() {
  const hash = location.hash.slice(1) || "upload";
  const [view, jobId] = hash.split("/");
  document.querySelectorAll(".view").forEach((v) => v.classList.add("hidden"));
  if (view === "upload") {
    document.getElementById("view-upload").classList.remove("hidden");
  } else if (view === "processing" && jobId) {
    document.getElementById("view-processing").classList.remove("hidden");
    enterProcessing(jobId);
  }
}

window.addEventListener("hashchange", route);

// Upload view -------------------------------------------------------------

function setupUpload() {
  const dz = document.getElementById("drop-zone");
  const input = document.getElementById("file-input");
  const submit = document.getElementById("submit-btn");
  const selected = document.getElementById("selected-file");
  const errorEl = document.getElementById("upload-error");

  dz.addEventListener("click", () => input.click());
  dz.addEventListener("dragover", (e) => {
    e.preventDefault();
    dz.classList.add("border-emerald-500");
  });
  dz.addEventListener("dragleave", () => dz.classList.remove("border-emerald-500"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("border-emerald-500");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
  });

  function handleFile(f) {
    errorEl.classList.add("hidden");
    const ext = f.name.split(".").pop().toLowerCase();
    if (!SUPPORTED_EXT.includes(ext)) {
      errorEl.textContent = t("upload_invalid_format");
      errorEl.classList.remove("hidden");
      return;
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      errorEl.textContent = t("upload_too_large", { max_mb: MAX_MB });
      errorEl.classList.remove("hidden");
      return;
    }
    state.selectedFile = f;
    selected.textContent = f.name;
    selected.classList.remove("hidden");
    submit.disabled = false;
  }

  document.getElementById("upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!state.selectedFile) return;
    submit.disabled = true;
    errorEl.classList.add("hidden");

    const fd = new FormData();
    fd.append("file", state.selectedFile);
    fd.append("language", document.querySelector("input[name=language]:checked").value);
    fd.append("enable_denoise", document.querySelector("input[name=enable_denoise]").checked);
    fd.append("enable_diarization", document.querySelector("input[name=enable_diarization]").checked);

    try {
      const r = await fetch("/api/jobs", { method: "POST", body: fd });
      if (!r.ok) {
        const detail = (await r.json().catch(() => ({}))).detail || `Server error: ${r.status}`;
        errorEl.textContent = detail;
        errorEl.classList.remove("hidden");
        submit.disabled = false;
        return;
      }
      const body = await r.json();
      location.hash = `#processing/${body.job_id}`;
    } catch (err) {
      errorEl.textContent = String(err);
      errorEl.classList.remove("hidden");
      submit.disabled = false;
    }
  });
}

// Processing view ---------------------------------------------------------

function setStatus(text) {
  document.getElementById("status-text").textContent = text;
}

function showPhase1() {
  document.getElementById("phase1-block").classList.remove("hidden");
}

function showPhase2() {
  document.getElementById("phase2-block").classList.remove("hidden");
}

function appendChunk(payload) {
  const c = document.getElementById("chunks-container");
  const start = formatTs(payload.start_s);
  const end = formatTs(payload.end_s);
  const div = document.createElement("div");
  div.className = "border-l-2 border-emerald-700 pl-3";
  div.innerHTML = `<span class="text-xs text-slate-500">[${start} → ${end}]</span><div>${escapeHtml(payload.text)}</div>`;
  c.appendChild(div);
  c.scrollTop = c.scrollHeight;
}

function formatTs(seconds) {
  const total = Math.max(0, Math.floor(seconds || 0));
  const h = String(Math.floor(total / 3600)).padStart(2, "0");
  const m = String(Math.floor((total % 3600) / 60)).padStart(2, "0");
  const s = String(total % 60).padStart(2, "0");
  return `${h}:${m}:${s}`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;","<": "&lt;",">": "&gt;","\"": "&quot;","'": "&#39;"}[c]));
}

function showError(msg, isPhase2 = false) {
  if (isPhase2) {
    const block = document.getElementById("phase2-failed-block");
    document.getElementById("phase2-error-msg").textContent = msg;
    block.classList.remove("hidden");
  } else {
    document.getElementById("error-banner").classList.remove("hidden");
    document.getElementById("error-msg").textContent = msg;
  }
}

function setRawDownload(jobId, url) {
  const a = document.getElementById("raw-download");
  a.href = url;
  a.classList.remove("hidden");
}

function setCleanDownload(jobId, url) {
  const a = document.getElementById("clean-download");
  a.href = url;
  a.classList.remove("hidden");
}

async function enterProcessing(jobId) {
  // Reset UI
  document.getElementById("chunks-container").innerHTML = "";
  document.getElementById("phase1-block").classList.add("hidden");
  document.getElementById("phase2-block").classList.add("hidden");
  document.getElementById("phase2-failed-block").classList.add("hidden");
  document.getElementById("error-banner").classList.add("hidden");
  document.getElementById("raw-download").classList.add("hidden");
  document.getElementById("clean-download").classList.add("hidden");

  setStatus("…");

  // Open WS
  if (state.ws) state.ws.close();
  const wsUrl = (location.protocol === "https:" ? "wss" : "ws") + `://${location.host}/ws/jobs/${jobId}`;
  const ws = new WebSocket(wsUrl);
  state.ws = ws;

  ws.addEventListener("message", (e) => {
    const payload = JSON.parse(e.data);
    handleEvent(jobId, payload);
  });

  ws.addEventListener("close", () => {
    state.ws = null;
  });
}

function handleEvent(jobId, payload) {
  switch (payload.type) {
    case "queue_position": {
      const pos = payload.position;
      if (pos === 0) setStatus(t("processing_phase1_running"));
      else if (pos === 1) setStatus(t("processing_queued_first"));
      else setStatus(t("processing_queued", { position: pos }));
      break;
    }
    case "phase1_started":
      setStatus(t("processing_phase1_running"));
      showPhase1();
      break;
    case "phase1_chunk":
      showPhase1();
      appendChunk(payload);
      break;
    case "phase1_completed":
      setStatus(t("processing_phase1_done"));
      showPhase1();
      setRawDownload(jobId, payload.transcript_url);
      break;
    case "phase2_started":
      setStatus(t("processing_phase2_running", { step: t(`processing_phase2_step_${payload.step || "denoise"}`) }));
      showPhase2();
      break;
    case "phase2_progress":
      document.getElementById("phase2-status").textContent =
        t("processing_phase2_running", { step: t(`processing_phase2_step_${payload.step || "transcribe"}`) }) +
        (payload.percent != null ? ` (${payload.percent}%)` : "");
      showPhase2();
      break;
    case "phase2_completed":
      setStatus(t("processing_phase2_done"));
      showPhase2();
      setCleanDownload(jobId, payload.transcript_url);
      break;
    case "error":
      if (payload.phase === "phase2") {
        showError(payload.message, true);
      } else {
        showError(payload.message, false);
      }
      break;
  }
}

// Logs modal --------------------------------------------------------------

function setupLogsModal() {
  const modal = document.getElementById("logs-modal");
  const open = () => {
    modal.classList.remove("hidden");
    fetch("/api/logs/recent").then((r) => r.text()).then((txt) => {
      document.getElementById("logs-content").textContent = txt;
    }).catch((e) => {
      document.getElementById("logs-content").textContent = String(e);
    });
  };
  document.getElementById("open-logs").addEventListener("click", open);
  document.getElementById("footer-logs").addEventListener("click", open);
  document.getElementById("logs-close").addEventListener("click", () => modal.classList.add("hidden"));
  document.getElementById("logs-copy").addEventListener("click", () => {
    const text = document.getElementById("logs-content").textContent;
    navigator.clipboard.writeText(text);
  });
}

// Language toggle ---------------------------------------------------------

function setupLangToggle() {
  document.getElementById("lang-toggle").addEventListener("click", async () => {
    const next = state.lang === "es" ? "en" : "es";
    localStorage.setItem(STORAGE_KEY, next);
    await loadI18n(next);
  });
}

// Boot --------------------------------------------------------------------

(async function init() {
  if (!["es", "en"].includes(state.lang)) state.lang = "es";
  await loadI18n(state.lang);
  setupUpload();
  setupLogsModal();
  setupLangToggle();
  route();
})();
