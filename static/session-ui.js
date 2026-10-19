/* Durable session UI. The backend owns recording and transcription jobs. */
let sessionSnapshot = null;
let sessionGeneration = 0;
let notesDirty = false;
let noteSavePromise = null;
let noteSaveTimer = null;
let jobTimer = null;
let currentJob = null;
let recordingAvailable = true;
let isProcessing = false;
let recordingRequestPending = false;
let acceleratorInfo = null;
let sessionViewRequest = 0;
let desktopCloseApproved = false;
let desktopClosePending = false;
let pendingLightNote = null;
const recordingStopWaiters = new Set();

function uiText(key, fallback) { return t[key] || fallback; }
function uiElement(id) { return document.getElementById(id); }

async function apiRequest(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        const detail = typeof body.detail === "string" ? body.detail : uiText("error_generic_retry", "Something went wrong. Please try again.");
        throw new Error(detail);
    }
    return response;
}

function showError(error) {
    const message = error instanceof Error ? error.message : String(error);
    const banner = uiElement("app-message");
    banner.textContent = message;
    banner.classList.remove("hidden");
    log(message, "error");
}

function clearError() { uiElement("app-message").classList.add("hidden"); }

function renderMarkdown(markdown) {
    const fragment = DOMPurify.sanitize(marked.parse(markdown || ""), {
        USE_PROFILES: { html: true }, RETURN_DOM_FRAGMENT: true,
        FORBID_TAGS: ["style", "form", "input", "button", "iframe", "object", "embed", "audio", "video"],
        FORBID_ATTR: ["style"],
    });
    fragment.querySelectorAll("img").forEach(img => {
        const src = img.getAttribute("src") || "";
        const local = /^\/api\/sessions\/[A-Za-z0-9_-]+\/images\/[A-Za-z0-9_.-]+$/.test(src);
        const embedded = /^data:image\/(png|jpeg|gif|webp);base64,[A-Za-z0-9+/=\s]+$/i.test(src);
        if (!local && !embedded) img.removeAttribute("src");
        img.removeAttribute("srcset");
    });
    fragment.querySelectorAll("a").forEach(anchor => {
        const href = anchor.getAttribute("href") || "";
        if (!/^https?:\/\//i.test(href)) anchor.removeAttribute("href");
        anchor.setAttribute("target", "_blank");
        anchor.setAttribute("rel", "noopener noreferrer");
    });
    return fragment;
}

function draftKey(id) { return `steno_draft_${id}`; }

function markNotesDirty() {
    if (!currentSessionId) return;
    notesDirty = true;
    uiElement("note-save-status").textContent = uiText("notes_unsaved", "Saving notes…");
    try { uiStorage.setItem(draftKey(currentSessionId), uiElement("notes-textarea").value); }
    catch (_) { showError(uiText("draft_storage_error", "Keep Steno open until your notes finish saving.")); }
    clearTimeout(noteSaveTimer);
    noteSaveTimer = setTimeout(() => saveNote(), 500);
}

async function flushNotes() {
    clearTimeout(noteSaveTimer);
    if (noteSavePromise) await noteSavePromise;
    if (!currentSessionId || !notesDirty) return;
    const id = currentSessionId;
    noteSavePromise = (async () => {
        while (currentSessionId === id && notesDirty) {
            const markdown = uiElement("notes-textarea").value;
            await apiRequest(`/api/sessions/${id}/note`, {
                method: "PUT", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ markdown }),
            });
            if (currentSessionId === id && uiElement("notes-textarea").value === markdown) {
                notesDirty = false;
                uiStorage.removeItem(draftKey(id));
                uiElement("note-save-status").textContent = uiText("notes_saved", "All notes saved");
            }
        }
    })();
    try { await noteSavePromise; } finally { noteSavePromise = null; }
}

async function saveNote() {
    try { await flushNotes(); }
    catch (error) {
        uiElement("note-save-status").textContent = uiText("notes_save_failed", "Notes not saved — retry");
        showError(error);
    }
}

function closeConnection() {
    clearTimeout(reconnectTimeout);
    clearTimeout(jobTimer);
    reconnectTimeout = null;
    sessionGeneration++;
    if (ws) {
        const previous = ws;
        ws = null;
        previous.onclose = null;
        previous.onmessage = null;
        previous.onerror = null;
        previous.close();
    }
    isRecording = false;
    recordingRequestPending = false;
    pendingLightNote = null;
    isProcessing = false;
    currentJob = null;
}

function canSwitchSession() {
    if (isRecording || recordingRequestPending) {
        showError(uiText("stop_before_switch", "Stop recording before changing sessions."));
        return false;
    }
    return true;
}

async function loadSessions() {
    try {
        const response = await apiRequest("/api/sessions");
        const sessions = await response.json();
        const list = uiElement("sessions-list");
        list.replaceChildren();
        if (!sessions.length) {
            const empty = document.createElement("div");
            empty.className = "sidebar-empty";
            empty.id = "no-sessions";
            empty.textContent = uiText("no_sessions_yet", "No sessions yet");
            list.appendChild(empty);
        }
        sessions.forEach(session => {
            const item = document.createElement("div");
            item.className = "sidebar-item" + (session.id === currentSessionId ? " active" : "");
            item.style.display = "flex";
            item.style.alignItems = "center";
            const open = document.createElement("button");
            open.className = "toolbar-btn";
            open.style.cssText = "flex:1;text-align:left;min-width:0;display:block;color:inherit";
            const name = document.createElement("div");
            name.className = "session-name";
            name.textContent = session.name;
            const date = document.createElement("div");
            date.className = "session-date";
            date.textContent = new Date(session.created_at).toLocaleDateString(currentLang) + (session.has_audio ? " · ♪" : "");
            open.append(name, date);
            open.onclick = () => viewSession(session.id);
            const remove = document.createElement("button");
            remove.className = "toolbar-btn";
            remove.textContent = "×";
            remove.setAttribute("aria-label", uiText("delete_session", "Delete session"));
            remove.onclick = async () => {
                const confirmed = await showDialog({ message: uiText("confirm_delete", "Delete this session?"), confirmText: uiText("delete", "Delete"), cancelText: uiText("cancel", "Cancel"), danger: true });
                if (confirmed) await deleteSession(session.id);
            };
            item.append(open, remove);
            list.appendChild(item);
        });
    } catch (error) { showError(error); }
}

function entryTimestamp(entry) {
    if (entry.start !== null && entry.start !== undefined) return formatElapsed(entry.start);
    if (entry.elapsed_seconds !== null && entry.elapsed_seconds !== undefined) return formatElapsed(entry.elapsed_seconds);
    return new Date(entry.timestamp).toLocaleTimeString(currentLang);
}

function renderSession(state, preserveNotes = false) {
    sessionSnapshot = state;
    uiElement("session-title").textContent = state.name;
    if (!isRecording) {
        const duration = (state.takes || []).reduce((total, take) => total + (take.duration_seconds || 0), 0);
        uiElement("session-clock").textContent = formatElapsed(duration);
    }
    const area = uiElement("transcript-area");
    area.replaceChildren();
    currentLiveEntry = null;
    if (state.legacy_markdown) {
        const original = document.createElement("article");
        original.className = "legacy-document";
        original.appendChild(renderMarkdown(state.legacy_markdown));
        area.appendChild(original);
    }
    (state.entries || []).forEach(entry => {
        const block = document.createElement("div");
        block.className = "transcript-entry";
        if ((entry.type || entry.kind) === "image") {
            block.appendChild(renderMarkdown(entry.tag || ""));
        } else {
            const stamp = document.createElement("span");
            stamp.className = "timestamp";
            stamp.textContent = `[${entryTimestamp(entry)}] `;
            if (entry.take_id && entry.start !== null && entry.start !== undefined) {
                stamp.style.cursor = "pointer";
                stamp.onclick = () => { showAudioPlayer(currentSessionId, entry.take_id); seekAudio(entry.start); };
            }
            const text = document.createElement("span");
            text.className = "text";
            text.style.whiteSpace = "pre-wrap";
            text.textContent = entry.text;
            block.append(stamp, text);
        }
        area.appendChild(block);
    });
    if (!preserveNotes || !notesDirty) {
        const draft = uiStorage.getItem(draftKey(currentSessionId));
        uiElement("notes-textarea").value = draft === null ? (state.notes || "") : draft;
        notesDirty = draft !== null && draft !== (state.notes || "");
        if (notesDirty) markNotesDirty();
    }
    uiElement("notes-textarea").disabled = false;
    uiElement("export-btn").disabled = false;
    if (state.has_audio) {
        showAudioPlayer(currentSessionId);
        showRetranscribeBar(currentSessionId);
    } else {
        hideAudioPlayer();
        uiElement("retranscribe-bar").style.display = "none";
    }
    if (isPreviewMode) showPreview(true);
    updateSessionControls();
}

async function refreshCurrentSession() {
    const id = currentSessionId;
    if (!id) return;
    const state = await (await apiRequest(`/api/sessions/${id}/state`)).json();
    if (currentSessionId === id) renderSession(state, true);
}

async function viewSession(id) {
    if (!canSwitchSession()) return;
    const request = ++sessionViewRequest;
    try {
        await flushNotes();
        const state = await (await apiRequest(`/api/sessions/${id}/state`)).json();
        if (request !== sessionViewRequest) return;
        closeConnection();
        currentSessionId = id;
        uiStorage.setItem("steno_last_session", id);
        clearError();
        notesDirty = false;
        uiElement("job-panel").classList.add("hidden");
        uiElement("session-modal").classList.add("hidden");
        showPreview(false);
        renderSession(state);
        connectWebSocket();
        await resumeSessionJobs();
        loadSessions();
    } catch (error) { showError(error); }
}

async function showNewSessionModal() {
    if (!canSwitchSession()) return;
    try {
        await flushNotes();
        uiElement("modal-transcription-lang").value = uiStorage.getItem("steno_transcription_lang") || "";
        uiElement("session-name-input").value = "";
        uiElement("session-modal").classList.remove("hidden");
        uiElement("session-name-input").focus();
    } catch (error) { showError(error); }
}

function closeNewSessionModal() { uiElement("session-modal").classList.add("hidden"); }

async function startSession() {
    if (!canSwitchSession()) return;
    const name = uiElement("session-name-input").value.trim();
    if (!name) { uiElement("session-name-input").focus(); return; }
    const button = uiElement("start-session-btn");
    button.disabled = true;
    try {
        await flushNotes();
        const result = await (await apiRequest("/api/sessions/new", {
            method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }),
        })).json();
        uiStorage.setItem("steno_transcription_lang", uiElement("modal-transcription-lang").value);
        uiStorage.setItem("steno_mic_index", uiElement("modal-mic-select").value);
        uiElement("mic-select").value = uiElement("modal-mic-select").value;
        await viewSession(result.session_id);
        uiElement("session-clock").textContent = "00:00:00";
    } catch (error) { showError(error); }
    finally { button.disabled = false; }
}

async function deleteSession(id) {
    if (id === currentSessionId && !canSwitchSession()) return;
    try {
        if (id === currentSessionId) await flushNotes();
        await apiRequest(`/api/sessions/${id}`, { method: "DELETE" });
        uiStorage.removeItem(draftKey(id));
        if (currentSessionId === id) {
            closeConnection();
            currentSessionId = null;
            sessionSnapshot = null;
            uiStorage.removeItem("steno_last_session");
            uiElement("notes-textarea").value = "";
            uiElement("notes-textarea").disabled = true;
            uiElement("transcript-area").replaceChildren();
            uiElement("session-title").textContent = "—";
            uiElement("export-btn").disabled = true;
            uiElement("job-panel").classList.add("hidden");
            hideAudioPlayer();
            uiElement("retranscribe-bar").style.display = "none";
            updateSessionControls();
        }
        await loadSessions();
    } catch (error) { showError(error); }
}

function connectWebSocket() {
    if (!currentSessionId) return;
    if (ws && [WebSocket.OPEN, WebSocket.CONNECTING].includes(ws.readyState)) return;
    const id = currentSessionId;
    const connection = new WebSocket(`${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws/${id}`);
    ws = connection;
    connection.onopen = () => {
        if (ws !== connection) return;
        reconnectDelay = 1000;
        setConnectionStatus(true);
        updateSessionControls();
        refreshCurrentSession().catch(showError);
    };
    connection.onmessage = event => {
        if (ws !== connection || currentSessionId !== id) return;
        try { handleWSMessage(JSON.parse(event.data)); } catch (error) { showError(error); }
    };
    connection.onclose = () => {
        if (ws !== connection) return;
        ws = null;
        isRecording = false;
        recordingRequestPending = false;
        pendingLightNote = null;
        setConnectionStatus(false);
        updateSessionControls();
        showError(uiText("connection_lost", "Connection lost. Reconnecting…"));
        scheduleReconnect();
    };
    connection.onerror = () => connection.close();
}

function scheduleReconnect() {
    clearTimeout(reconnectTimeout);
    if (!currentSessionId) return;
    reconnectTimeout = setTimeout(connectWebSocket, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, 30000);
}

function updateSessionControls() {
    updateRecordButton();
    updateStatusUI();
    uiElement("record-btn").disabled = !currentSessionId || !recordingAvailable || recordingRequestPending || (!isRecording && isProcessing) || !ws || ws.readyState !== WebSocket.OPEN;
    uiElement("model-select").disabled = isRecording || isProcessing;
    uiElement("mic-select").disabled = isRecording || recordingRequestPending;
    uiElement("audio-file-input").disabled = !currentSessionId || isRecording || isProcessing;
    uiElement("retranscribe-btn").disabled = isRecording || isProcessing;
    if (isProcessing && !isRecording) uiElement("status-text").textContent = uiText("job_running", "Processing saved audio…");
    uiElement("light-mode-notes").style.display = isRecording && sessionMode === "light" ? "flex" : "none";
}

function handleWSMessage(message) {
    if (message.type === "transcript") addTranscriptEntry(message.text, message.timestamp);
    else if (message.type === "status") {
        isRecording = Boolean(message.recording);
        isProcessing = Boolean(message.processing);
        recordingRequestPending = false;
        if (!isRecording) recordingStopWaiters.forEach(resolve => resolve());
        updateSessionControls();
        if (message.job_id && currentJob?.id !== message.job_id) watchJob(message.job_id);
        if (message.has_audio && !isRecording) refreshCurrentSession().catch(showError);
    } else if (message.type === "elapsed") {
        elapsedSeconds = message.seconds;
        uiElement("session-clock").textContent = formatElapsed(elapsedSeconds);
    } else if (message.type === "error") {
        recordingRequestPending = false;
        pendingLightNote = null;
        showError(message.message);
        updateSessionControls();
        if (message.job_id) watchJob(message.job_id);
    } else if (message.type === "note_saved") {
        const input = uiElement("light-note-input");
        if (pendingLightNote && input.value === pendingLightNote) input.value = "";
        pendingLightNote = null;
        refreshCurrentSession().catch(showError);
    }
    else if (message.type === "model_loading") setModelStatus("loading");
    else if (message.type === "model_ready") setModelStatus("ready");
}

function toggleRecording() {
    if (!ws || ws.readyState !== WebSocket.OPEN) { showError(uiText("connection_lost", "Connection lost. Reconnecting…")); return; }
    clearError();
    recordingRequestPending = true;
    if (isRecording) ws.send(JSON.stringify({ type: "stop" }));
    else {
        const index = parseInt(uiElement("mic-select").value, 10);
        ws.send(JSON.stringify({ type: "start", device_index: Number.isNaN(index) ? null : index,
            language: uiStorage.getItem("steno_transcription_lang") || null, mode: sessionMode }));
    }
    updateSessionControls();
}

function showPreview(show) {
    isPreviewMode = show;
    const preview = uiElement("preview-container");
    if (show) preview.replaceChildren(renderMarkdown(uiElement("notes-textarea").value));
    uiElement("editor-container").style.display = show ? "none" : "flex";
    preview.style.display = show ? "block" : "none";
    uiElement("btn-preview").querySelector("span").textContent = show ? uiText("edit", "Edit") : uiText("preview", "Preview");
}

async function uploadImageFile(file) {
    if (!currentSessionId) return;
    const id = currentSessionId;
    try {
        const form = new FormData();
        form.append("file", file);
        form.append("caption", file.name);
        await apiRequest(`/api/sessions/${id}/image`, { method: "POST", body: form });
        if (currentSessionId === id) await refreshCurrentSession();
    } catch (error) { showError(error); }
}

async function exportSession() {
    if (!currentSessionId) return;
    try {
        await flushNotes();
        const format = uiElement("export-format").value;
        const response = await apiRequest(`/api/sessions/${currentSessionId}/export?format=${format}`, { method: "POST" });
        const url = URL.createObjectURL(await response.blob());
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = `${currentSessionId}.${format}`;
        anchor.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { showError(error); }
}

async function handleAudioUpload(event) {
    const file = event.target.files[0];
    event.target.value = "";
    if (!file || !currentSessionId) return;
    const id = currentSessionId;
    try {
        await flushNotes();
        clearError();
        uiElement("upload-progress").classList.add("active");
        uiElement("upload-bar").style.width = "0%";
        uiElement("upload-status").textContent = uiText("uploading_audio", "Saving audio…");
        const form = new FormData();
        form.append("file", file);
        form.append("language", uiStorage.getItem("steno_transcription_lang") || "");
        const result = await (await apiRequest(`/api/sessions/${id}/transcribe-file`, { method: "POST", body: form })).json();
        if (currentSessionId === id) watchJob(result.job_id);
    } catch (error) { showError(error); }
    finally { uiElement("upload-progress").classList.remove("active"); }
}

function renderJob(job) {
    currentJob = job;
    const panel = uiElement("job-panel");
    panel.classList.remove("hidden");
    const labels = { queued: ["job_queued", "Queued"], running: ["job_running", "Processing saved audio…"], cancelling: ["job_cancelling", "Stopping after the current segment…"], completed: ["job_completed", "Transcription complete"], failed: ["job_failed", "Transcription failed"], cancelled: ["job_cancelled", "Transcription cancelled"] };
    const label = labels[job.status] || labels.running;
    uiElement("job-label").textContent = uiText(...label) + (job.error ? ` — ${job.error}` : "");
    const pending = ["queued", "running", "cancelling"].includes(job.status);
    isProcessing = pending && !isRecording;
    uiElement("job-cancel").hidden = !pending;
    uiElement("job-cancel").disabled = job.status === "cancelling";
    uiElement("job-retry").hidden = !["failed", "cancelled"].includes(job.status);
    uiElement("job-cpu").hidden = !["failed", "cancelled"].includes(job.status) || acceleratorInfo?.backend === "mlx";
    updateSessionControls();
}

async function watchJob(id) {
    clearTimeout(jobTimer);
    const sessionId = currentSessionId;
    const generation = sessionGeneration;
    try {
        const job = await (await apiRequest(`/api/jobs/${id}`)).json();
        if (sessionId !== currentSessionId || generation !== sessionGeneration) return;
        renderJob(job);
        if (["completed", "failed", "cancelled"].includes(job.status)) {
            await refreshCurrentSession();
            loadSessions();
        } else jobTimer = setTimeout(() => watchJob(id), 1000);
    } catch (error) {
        if (sessionId === currentSessionId && generation === sessionGeneration) {
            showError(error);
            jobTimer = setTimeout(() => watchJob(id), 3000);
        }
    }
}

async function resumeSessionJobs() {
    const id = currentSessionId;
    const data = await (await apiRequest(`/api/jobs?session_id=${id}`)).json();
    if (currentSessionId !== id) return;
    const jobs = Array.isArray(data) ? data : (data.jobs || []);
    const pending = jobs.find(job => ["queued", "running", "cancelling"].includes(job.status));
    const latest = jobs[jobs.length - 1];
    if (pending) watchJob(pending.id);
    else if (latest && ["failed", "cancelled"].includes(latest.status)) renderJob(latest);
}

async function cancelCurrentJob() {
    if (!currentJob) return;
    try { await apiRequest(`/api/jobs/${currentJob.id}/cancel`, { method: "POST" }); watchJob(currentJob.id); }
    catch (error) { showError(error); }
}

async function retryCurrentJob(device = null) {
    if (!currentJob) return;
    try {
        const result = await (await apiRequest(`/api/jobs/${currentJob.id}/retry`, {
            method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(device ? { device } : {}),
        })).json();
        clearError();
        watchJob(result.job_id || result.id || currentJob.id);
    } catch (error) { showError(error); }
}

function showAudioPlayer(id, takeId = null) {
    if (id !== currentSessionId) return;
    const select = uiElement("audio-take-select");
    const previous = takeId || select.value;
    select.replaceChildren();
    (sessionSnapshot?.takes || []).forEach((take, index) => {
        const option = document.createElement("option");
        option.value = take.id;
        option.textContent = `${uiText("recording_take", "Recording")} ${index + 1}`;
        select.appendChild(option);
    });
    if (Array.from(select.options).some(option => option.value === previous)) select.value = previous;
    else if (select.options.length) select.selectedIndex = select.options.length - 1;
    const source = `/api/sessions/${id}/audio${select.value ? `?take_id=${encodeURIComponent(select.value)}` : ""}`;
    const player = uiElement("session-audio-player");
    if (player.getAttribute("src") !== source) player.setAttribute("src", source);
    select.hidden = select.options.length < 2;
    uiElement("audio-player-container").style.display = "flex";
}

async function showRetranscribeBar(id) {
    try {
        const data = await (await apiRequest("/api/models")).json();
        if (id !== currentSessionId) return;
        const select = uiElement("retranscribe-model-select");
        select.replaceChildren();
        (data.models || []).filter(model => model.downloaded).forEach(model => {
            const option = document.createElement("option");
            option.value = model.key;
            option.textContent = model.key;
            option.selected = model.active;
            select.appendChild(option);
        });
        uiElement("retranscribe-bar").style.display = "flex";
        updateSessionControls();
    } catch (error) { showError(error); }
}

async function retranscribeSession() {
    try {
        await flushNotes();
        const id = currentSessionId;
        const result = await (await apiRequest(`/api/sessions/${id}/retranscribe`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_key: uiElement("retranscribe-model-select").value,
                take_id: uiElement("audio-take-select").value || null,
                language: uiStorage.getItem("steno_transcription_lang") || null }),
        })).json();
        if (currentSessionId === id) watchJob(result.job_id);
    } catch (error) { showError(error); }
}

function addLightNote() {
    const input = uiElement("light-note-input");
    const text = input.value.trim();
    if (!text || pendingLightNote || !isRecording || !ws || ws.readyState !== WebSocket.OPEN) return;
    pendingLightNote = input.value;
    try { ws.send(JSON.stringify({ type: "note", text, elapsed_seconds: elapsedSeconds })); }
    catch (error) { pendingLightNote = null; showError(error); }
}

async function testMicrophone() {
    const button = uiElement("mic-test-btn");
    button.disabled = true;
    const status = uiElement("mic-test-label");
    uiElement("mic-test-area").style.display = "block";
    try {
        const index = uiElement("setup-mic-select").value;
        status.textContent = uiText("setup_mic_listening", "Listening… speak to test");
        const result = await (await apiRequest(`/api/mic-test?device_index=${encodeURIComponent(index)}`)).json();
        if (result.status !== "ok") throw new Error(result.message || uiText("error_start_recording", "Could not access microphone."));
        uiElement("mic-level-fill").style.width = `${Math.min(result.rms * 500, 100)}%`;
        status.textContent = uiText(result.has_audio ? "mic_test_ok" : "mic_test_silent", result.has_audio ? "Microphone works" : "No sound detected. Check the selected microphone.");
    } catch (error) { status.textContent = error.message; showError(error); }
    finally { button.disabled = false; }
}

async function finishSetup() {
    uiStorage.setItem("steno_transcription_lang", uiElement("setup-transcription-lang").value);
    uiStorage.setItem("steno_ui_lang", currentLang);
    const index = uiElement("setup-mic-select").value;
    uiStorage.setItem("steno_mic_index", index);
    uiElement("mic-select").value = index;
    uiElement("modal-mic-select").value = index;
    uiElement("setup-overlay").classList.add("hidden");
    await loadModelSelector();
    await showNewSessionModal();
}

async function toggleLanguage() {
    await loadI18n(currentLang === "en" ? "es" : "en");
    uiStorage.setItem("steno_ui_lang", currentLang);
    applySessionI18n();
    loadSessions();
}

function applySessionI18n() {
    const labels = { "job-cancel": ["cancel", "Cancel"], "job-retry": ["setup_retry", "Retry"], "job-cpu": ["retry_cpu", "Retry on CPU"], "upload-label-text": ["upload_audio", "Upload audio file to transcribe"] };
    Object.entries(labels).forEach(([id, values]) => { uiElement(id).textContent = uiText(...values); });
    if (currentJob) renderJob(currentJob);
    uiElement("open-logs-btn").hidden = typeof window.electronAPI?.openLogs !== "function";
    renderCapabilities();
}

function renderCapabilities() {
    if (!acceleratorInfo) return;
    const device = acceleratorInfo.device || acceleratorInfo.backend || "";
    uiElement("accelerator-banner").textContent = [
        device ? `${uiText("processing_device", "Processing")}: ${device === "metal" ? "Metal" : device.toUpperCase()}` : "",
        acceleratorInfo.fallback_reason ? fallbackDescription(acceleratorInfo.fallback_reason) : "",
        acceleratorInfo.available === false ? uiText("engine_unavailable", "Transcription engine unavailable. Reinstall Steno.") : "",
    ].filter(Boolean).join(" · ");
}

async function checkSetup() {
    try {
        const status = await (await apiRequest("/api/status")).json();
        recordingAvailable = status.portaudio_available !== false;
        acceleratorInfo = status.capabilities || {};
        renderCapabilities();
        if (!recordingAvailable) uiElement("portaudio-banner").classList.remove("hidden");
        applySessionI18n();
        updateSessionControls();
        if (!status.setup_complete) await showSetup();
        else {
            const last = uiStorage.getItem("steno_last_session");
            if (last) {
                const sessions = await (await apiRequest("/api/sessions")).json();
                if (sessions.some(session => session.id === last)) { await viewSession(last); return; }
                uiStorage.removeItem("steno_last_session");
            }
            // Keep existing sessions accessible; a new session can always be created.
            const sessions = await (await apiRequest("/api/sessions")).json();
            if (!sessions.length) await showNewSessionModal();
        }
    } catch (error) { showError(error); }
}

async function resetApp() {
    if (!canSwitchSession()) return;
    const confirmed = await showDialog({ message: uiText("settings_reset_confirm_1", "Delete sessions and reset settings?"), confirmText: uiText("delete", "Delete"), cancelText: uiText("cancel", "Cancel"), danger: true });
    if (!confirmed) return;
    try {
        await flushNotes();
        await apiRequest("/api/reset", { method: "POST" });
        closeConnection();
        for (const key of uiStorage.keys()) if (key.startsWith("steno_")) uiStorage.removeItem(key);
        location.reload();
    } catch (error) { showError(error); }
}

uiElement("notes-textarea").disabled = true;
uiElement("notes-textarea").addEventListener("input", markNotesDirty);
uiElement("audio-take-select").addEventListener("change", () => showAudioPlayer(currentSessionId, uiElement("audio-take-select").value));
window.addEventListener("steno-open-settings", showSettingsModal);
window.addEventListener("beforeunload", event => {
    if (!desktopCloseApproved && (notesDirty || isRecording || recordingRequestPending)) { event.preventDefault(); event.returnValue = ""; }
});

async function cancelModelDownload() {
    try { await apiRequest("/api/models/download/cancel", { method: "POST" }); }
    catch (error) { showError(error); }
}

function fallbackDescription(reason) {
    const reasons = {
        cuda_unavailable: ["fallback_cuda_unavailable", "No compatible NVIDIA GPU or CUDA driver was detected. Using CPU."],
        cuda_initialization_failed: ["fallback_cuda_initialization", "The NVIDIA GPU could not initialize CUDA. Check its driver and CUDA libraries; using CPU."],
        cuda_probe_failed: ["fallback_cuda_probe", "CUDA detection failed. Check the NVIDIA driver; using CPU."],
        cpu_int8_unavailable: ["fallback_cpu_int8", "CPU transcription is unavailable: this computer does not support the required INT8 mode."],
        runtime_unavailable: ["fallback_runtime", "The transcription runtime is unavailable. Reinstall Steno."],
    };
    return uiText(...(reasons[reason] || ["cpu_fallback", "CUDA is unavailable. Processing on CPU."]));
}

async function refreshDebugInfo() {
    const content = uiElement("debug-info-content");
    try {
        const info = await (await apiRequest("/api/diagnostics")).json();
        uiElement("support-data-path").textContent = info.logs_path
            ? `${uiText("diagnostics_logs", "Logs folder")}: ${info.logs_path}`
            : info.data_dir ? `${uiText("diagnostics_data", "Data folder")}: ${info.data_dir}` : "";
        const engine = info.transcriber || {};
        const rows = [
            [uiText("diagnostics_version", "Version"), info.app_version],
            [uiText("diagnostics_platform", "Operating system"), info.platform],
            [uiText("diagnostics_backend", "Engine"), engine.backend],
            [uiText("processing_device", "Processing"), engine.device],
            [uiText("diagnostics_validated", "Inference validated"), uiText(engine.validated ? "yes" : "no", engine.validated ? "Yes" : "No")],
            [uiText("diagnostics_model", "Model"), engine.model_name],
            [uiText("diagnostics_data", "Data folder"), info.data_dir],
        ];
        if (engine.fallback_reason) rows.push([uiText("diagnostics_fallback", "CPU fallback"), fallbackDescription(engine.fallback_reason)]);
        if (engine.diagnostic) rows.push([uiText("diagnostics_detail", "Diagnostic detail"), engine.diagnostic]);
        content.replaceChildren();
        rows.forEach(([label, value]) => {
            const row = document.createElement("div");
            row.className = "debug-row";
            const title = document.createElement("span");
            title.className = "debug-label";
            title.textContent = label;
            const detail = document.createElement("span");
            detail.className = "debug-value";
            detail.textContent = value == null ? "—" : String(value);
            row.append(title, detail);
            content.appendChild(row);
        });
    } catch (error) { content.textContent = error.message; showError(error); }
}

function showSupportError(error) {
    const message = uiElement("support-message");
    message.textContent = error.message || String(error);
    message.hidden = false;
    showError(error);
}

async function openLogsFolder() {
    if (typeof window.electronAPI?.openLogs !== "function") return;
    const button = uiElement("open-logs-btn");
    button.disabled = true;
    uiElement("support-message").hidden = true;
    try {
        const result = await window.electronAPI.openLogs();
        if (!result?.ok) throw new Error(result?.error || uiText("open_logs_failed", "Could not open the logs folder."));
    } catch (error) { showSupportError(error); }
    finally { button.disabled = false; }
}

async function exportDiagnostics() {
    const button = uiElement("export-diagnostics-btn");
    button.disabled = true;
    uiElement("support-message").hidden = true;
    try {
        const response = await apiRequest("/api/diagnostics/export");
        const url = URL.createObjectURL(await response.blob());
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = "steno-diagnostics.zip";
        anchor.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { showSupportError(error); }
    finally { button.disabled = false; }
}

function stopBeforeClose() {
    if (!isRecording && !recordingRequestPending) return Promise.resolve();
    return new Promise((resolve, reject) => {
        const finish = () => { clearTimeout(timer); recordingStopWaiters.delete(finish); resolve(); };
        const timer = setTimeout(() => {
            recordingStopWaiters.delete(finish);
            reject(new Error(uiText("close_recording_timeout", "Recording has not stopped yet. Stop it before closing Steno.")));
        }, 10000);
        recordingStopWaiters.add(finish);
        if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "stop" }));
        else { clearTimeout(timer); recordingStopWaiters.delete(finish); reject(new Error(uiText("connection_lost", "Connection lost. Reconnecting…"))); }
    });
}

async function prepareDesktopClose() {
    if (desktopClosePending || !window.electronAPI?.closeReady) return;
    desktopClosePending = true;
    try {
        if (notesDirty && currentSessionId) {
            uiStorage.setItem(draftKey(currentSessionId), uiElement("notes-textarea").value);
        }
        await stopBeforeClose();
        let saveTimeout;
        try {
            await Promise.race([
                flushNotes(),
                new Promise((_, reject) => { saveTimeout = setTimeout(() => reject(new Error(uiText("close_save_timeout", "Notes are still saving. Try closing Steno again."))), 3000); }),
            ]);
        } finally { clearTimeout(saveTimeout); }
        desktopCloseApproved = true;
        window.electronAPI.closeReady(true);
    } catch (error) {
        // Compare the latest editor value: input may change while recording stops.
        let durableDraft = !notesDirty;
        try { if (notesDirty) durableDraft = !!currentSessionId && uiStorage.getItem(draftKey(currentSessionId)) === uiElement("notes-textarea").value; }
        catch (_) { durableDraft = false; }
        desktopCloseApproved = durableDraft && !isRecording && !recordingRequestPending;
        if (!desktopCloseApproved) showError(error);
        window.electronAPI.closeReady(desktopCloseApproved);
    } finally { desktopClosePending = false; }
}
window.addEventListener("steno-before-close", prepareDesktopClose);
