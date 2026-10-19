const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const source = readFileSync(join(__dirname, '../static/session-ui.js'), 'utf8');
const storageSource = readFileSync(join(__dirname, '../static/ui-storage.js'), 'utf8');

class Element {
    constructor(tag = 'div') {
        this.tagName = tag;
        this.children = [];
        this.style = {};
        this.attributes = {};
        this.value = '';
        this.textContent = '';
        this.listeners = {};
        this.classes = new Set();
        this.classList = { add: c => this.classes.add(c), remove: c => this.classes.delete(c) };
    }
    append(...children) { this.children.push(...children); }
    appendChild(child) { this.append(child); return child; }
    replaceChildren(...children) { this.children = children; }
    get options() { return this.children.filter(child => child.tagName === 'option'); }
    set selectedIndex(index) { this.value = this.options[index]?.value || ''; }
    querySelector() { return new Element('span'); }
    querySelectorAll() { return []; }
    setAttribute(key, value) { this.attributes[key] = value; }
    getAttribute(key) { return this.attributes[key] ?? null; }
    removeAttribute(key) { delete this.attributes[key]; }
    addEventListener(name, fn) { this.listeners[name] = fn; }
    focus() {}
    click() { this.clicked = true; }
}

function harness(fetch, electronAPI = null) {
    const elements = new Map();
    const values = new Map();
    const purifyOptions = [];
    const context = {
        document: { getElementById: id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); },
            createElement: tag => new Element(tag) },
        localStorage: { getItem: key => values.has(key) ? values.get(key) : null,
            setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) },
        window: { electronAPI, addEventListener() {} }, location: { protocol: 'http:', host: '127.0.0.1:8080' },
        fetch, t: {}, currentLang: 'en', currentSessionId: null, isRecording: false,
        isPreviewMode: false, ws: null, reconnectTimeout: null, reconnectDelay: 1000,
        sessionMode: 'full', elapsedSeconds: 0, sessionStartTime: null, currentLiveEntry: null,
        setTimeout: () => 1, clearTimeout() {},
        DOMPurify: { sanitize(markdown, options) { purifyOptions.push(options); const element = new Element(); element.rendered = markdown; return element; } },
        marked: { parse: value => value },
        log() {}, updateRecordButton() {}, updateStatusUI() {}, hideAudioPlayer() {},
        formatElapsed: value => String(value), setConnectionStatus() {}, showSettingsModal() {},
        startClock() {}, showDialog: async () => true,
        URL: { createObjectURL: () => 'blob:test', revokeObjectURL() {} },
        FormData: class { append() {} },
        WebSocket: class { static OPEN = 1; static CONNECTING = 0; constructor(url) { this.url = url; this.readyState = 0; } close() { this.closed = true; } },
    };
    vm.createContext(context);
    vm.runInContext(storageSource, context);
    vm.runInContext(source, context);
    return { context, elements, values, purifyOptions, run: code => vm.runInContext(code, context), el: id => context.document.getElementById(id) };
}

const response = body => ({ ok: true, json: async () => body, blob: async () => 'blob' });

test('history renders persisted content and switches the note editor to that session', async () => {
    const saved = { id: 'saved', name: 'Saved session', notes: 'SAVED NOTES', entries: [{ type: 'transcript', text: 'SAVED TRANSCRIPT', timestamp: '2026-01-01T10:00:00' }], takes: [] };
    const h = harness(async url => response(url.endsWith('/state') ? saved : []));
    h.el('notes-textarea').value = 'WRONG SESSION NOTES';
    await h.context.viewSession('saved');
    assert.equal(h.context.currentSessionId, 'saved');
    assert.equal(h.el('notes-textarea').value, 'SAVED NOTES');
    assert.equal(h.el('transcript-area').children[0].children[1].textContent, 'SAVED TRANSCRIPT');
    assert.equal(h.el('session-title').textContent, 'Saved session');
});

test('notes save by replacement, including an empty editor', async () => {
    const calls = [];
    const h = harness(async (url, options) => { calls.push({ url, ...options }); return response({}); });
    h.context.currentSessionId = 'notes';
    h.el('notes-textarea').value = '';
    h.context.markNotesDirty();
    await h.context.flushNotes();
    assert.equal(calls[0].method, 'PUT');
    assert.equal(JSON.parse(calls[0].body).markdown, '');
    assert.equal(h.values.has('steno_draft_notes'), false);
    await h.context.flushNotes();
    assert.equal(calls.length, 1);
});

test('typing during an in-flight save is flushed before a switch or export', async () => {
    const saves = [];
    let release;
    const h = harness(async (url, options) => {
        saves.push(JSON.parse(options.body).markdown);
        if (saves.length === 1) await new Promise(resolve => { release = resolve; });
        return response({});
    });
    h.context.currentSessionId = 'notes';
    h.el('notes-textarea').value = 'first'; h.context.markNotesDirty();
    const pending = h.context.flushNotes();
    h.el('notes-textarea').value = 'second'; h.context.markNotesDirty();
    release(); await pending;
    assert.deepEqual(saves, ['first', 'second']);
    assert.equal(h.run('notesDirty'), false);
});

test('HTTP errors preserve the local draft and prevent exporting stale notes', async () => {
    const calls = [];
    const h = harness(async url => { calls.push(url); return { ok: false, json: async () => ({ detail: 'disk full' }) }; });
    h.context.currentSessionId = 'notes';
    h.el('notes-textarea').value = 'important'; h.context.markNotesDirty();
    await h.context.exportSession();
    assert.deepEqual(calls, ['/api/sessions/notes/note']);
    assert.equal(h.values.get('steno_draft_notes'), 'important');
    assert.equal(h.el('app-message').textContent, 'disk full');
});

test('export waits for save and honors ZIP format', async () => {
    const calls = [];
    const h = harness(async url => { calls.push(url); return response({}); });
    h.context.currentSessionId = 'notes';
    h.el('export-format').value = 'zip';
    h.el('notes-textarea').value = 'latest'; h.context.markNotesDirty();
    await h.context.exportSession();
    assert.deepEqual(calls, ['/api/sessions/notes/note', '/api/sessions/notes/export?format=zip']);
});

test('recording prevents a session switch and keeps the owning socket', async () => {
    const h = harness(async () => { throw new Error('should not fetch'); });
    h.context.currentSessionId = 'recording'; h.context.isRecording = true;
    const socket = { readyState: 1 }; h.context.ws = socket;
    await h.context.viewSession('other');
    assert.equal(h.context.currentSessionId, 'recording');
    assert.equal(h.context.ws, socket);
    assert.match(h.el('app-message').textContent, /Stop recording/);
});

test('intentional connection close disables callbacks before closing', () => {
    const h = harness(async () => response({}));
    let callbacks;
    const socket = { onclose() {}, onmessage() {}, onerror() {}, close() { callbacks = [this.onclose, this.onmessage, this.onerror]; } };
    h.context.ws = socket;
    h.context.closeConnection();
    assert.deepEqual(callbacks, [null, null, null]);
    assert.equal(h.context.ws, null);
});

test('a stale response cannot replace another sessions history', async () => {
    let release;
    const h = harness(async () => { await new Promise(resolve => { release = resolve; }); return response({ name: 'old', entries: [] }); });
    h.context.currentSessionId = 'old';
    const pending = h.context.refreshCurrentSession();
    h.context.currentSessionId = 'new';
    h.el('session-title').textContent = 'new';
    release(); await pending;
    assert.equal(h.el('session-title').textContent, 'new');
});

test('rapid history clicks keep the last selection when responses arrive out of order', async () => {
    let releaseFirst;
    const h = harness(async url => {
        if (url === '/api/sessions/first/state') {
            await new Promise(resolve => { releaseFirst = resolve; });
            return response({ name: 'First', notes: '', entries: [], takes: [] });
        }
        return response(url.endsWith('/state') ? { name: 'Last', notes: '', entries: [], takes: [] } : []);
    });
    const first = h.context.viewSession('first');
    await Promise.resolve();
    await h.context.viewSession('last');
    releaseFirst();
    await first;
    assert.equal(h.context.currentSessionId, 'last');
    assert.equal(h.el('session-title').textContent, 'Last');
});

test('failed file upload displays the HTTP error instead of success with zero segments', async () => {
    const h = harness(async () => ({ ok: false, json: async () => ({ detail: 'Unsupported format' }) }));
    h.context.currentSessionId = 'session';
    await h.context.handleAudioUpload({ target: { files: [{ name: 'broken.mp3' }], value: 'x' } });
    assert.equal(h.el('app-message').textContent, 'Unsupported format');
    assert.notEqual(h.el('job-label').textContent, '0 segments');
});

test('job UI uses actual queued/running/failed statuses and exposes retry', () => {
    const h = harness(async () => response({}));
    h.context.renderJob({ id: 'job', status: 'failed', error: 'GPU unavailable' });
    assert.equal(h.el('job-retry').hidden, false);
    assert.equal(h.el('job-cancel').hidden, true);
    assert.match(h.el('job-label').textContent, /GPU unavailable/);
    h.context.renderJob({ id: 'job', status: 'queued' });
    assert.equal(h.el('job-cancel').hidden, false);
    assert.equal(h.el('job-retry').hidden, true);
    assert.equal(h.el('job-label').textContent, 'Queued');
});

test('Markdown is sanitized and external media URLs are removed', () => {
    const h = harness(async () => response({}));
    const external = new Element('img'); external.setAttribute('src', 'https://tracking.example/pixel'); external.setAttribute('srcset', '//evil/image 1x');
    const local = new Element('img'); local.setAttribute('src', '/api/sessions/abc/images/pic.png');
    const svg = new Element('img'); svg.setAttribute('src', 'data:image/svg+xml;base64,AAA=');
    const link = new Element('a'); link.setAttribute('href', 'javascript:alert(1)');
    h.context.DOMPurify.sanitize = (input, options) => {
        assert.equal(options.USE_PROFILES.html, true);
        assert.equal(options.RETURN_DOM_FRAGMENT, true);
        return { querySelectorAll: selector => selector === 'img' ? [external, local, svg] : [link] };
    };
    h.context.renderMarkdown('<img onerror=alert(1)>');
    assert.equal(external.getAttribute('src'), null);
    assert.equal(external.getAttribute('srcset'), null);
    assert.equal(svg.getAttribute('src'), null);
    assert.equal(local.getAttribute('src'), '/api/sessions/abc/images/pic.png');
    assert.equal(link.getAttribute('href'), null);
});

test('desktop drafts and preferences restore through a stable bridge across origins', () => {
    const saved = new Map();
    const storage = { getItem: key => saved.has(key) ? saved.get(key) : null,
        setItem: (key, value) => saved.set(key, value), removeItem: key => saved.delete(key), keys: () => [...saved.keys()] };
    const first = harness(async () => response({}), { storage });
    first.context.currentSessionId = 'session';
    first.el('notes-textarea').value = 'durable draft'; first.context.markNotesDirty();
    first.run('uiStorage.setItem("steno_ui_lang", "es")');
    const second = harness(async () => response({}), { storage });
    second.context.location.host = '127.0.0.1:50299';
    second.context.currentSessionId = 'session';
    second.context.renderSession({ name: 'session', notes: 'old server value', entries: [], takes: [] });
    assert.equal(second.el('notes-textarea').value, 'durable draft');
    assert.equal(second.run('uiStorage.getItem("steno_ui_lang")'), 'es');
    assert.equal(second.values.size, 0);
});

test('desktop close acknowledges only after notes are flushed', async () => {
    const actions = [];
    const h = harness(async () => { actions.push('saved'); return response({}); }, { closeReady: success => actions.push(success) });
    h.context.currentSessionId = 'session';
    h.el('notes-textarea').value = 'final words'; h.context.markNotesDirty();
    await h.context.prepareDesktopClose();
    assert.deepEqual(actions, ['saved', true]);
    assert.equal(h.run('desktopCloseApproved'), true);
});

test('failed close save allows recovery only when the latest draft is durable', async () => {
    const approvals = [];
    const h = harness(async () => ({ ok: false, json: async () => ({ detail: 'offline' }) }), { closeReady: success => approvals.push(success) });
    h.context.currentSessionId = 'session';
    h.el('notes-textarea').value = 'recover me'; h.context.markNotesDirty();
    await h.context.prepareDesktopClose();
    assert.deepEqual(approvals, [true]);
    assert.equal(h.values.get('steno_draft_session'), 'recover me');
    h.context.localStorage.setItem = () => { throw new Error('disk full'); };
    h.el('notes-textarea').value = 'newer draft'; h.context.markNotesDirty();
    await h.context.prepareDesktopClose();
    assert.deepEqual(approvals, [true, false]);
});

test('desktop close stops recording and awaits backend status before flushing', async () => {
    const actions = [];
    const h = harness(async () => { actions.push('save'); return response({}); }, { closeReady: success => actions.push(success) });
    h.context.currentSessionId = 'session'; h.context.isRecording = true;
    h.el('notes-textarea').value = 'notes'; h.context.markNotesDirty();
    h.context.ws = { readyState: 1, send: payload => actions.push(JSON.parse(payload).type) };
    const pending = h.context.prepareDesktopClose();
    assert.deepEqual(actions, ['stop']);
    h.context.handleWSMessage({ type: 'status', recording: false, processing: true });
    await pending;
    assert.deepEqual(actions, ['stop', 'save', true]);
});

test('diagnostics uses the production endpoint and renders diagnostic details as text', async () => {
    const calls = [];
    const h = harness(async url => { calls.push(url); return response({ app_version: '1.0.0', platform: 'Windows', data_dir: '/safe', transcriber: { backend: 'faster-whisper', device: 'cpu', validated: false, fallback_reason: 'cuda_initialization_failed', diagnostic: '<script>unsafe</script>' } }); });
    await h.context.refreshDebugInfo();
    assert.deepEqual(calls, ['/api/diagnostics']);
    assert.equal(h.el('debug-info-content').children.at(-1).children[1].textContent, '<script>unsafe</script>');
    assert.match(h.el('debug-info-content').children.at(-2).children[1].textContent, /could not initialize CUDA/);
});

test('timestamped notes remain visible until acknowledged and preserve edits made in flight', () => {
    const sent = [];
    const h = harness(async () => response({}));
    h.context.isRecording = true;
    h.context.ws = { readyState: 1, send: payload => sent.push(JSON.parse(payload)) };
    h.el('light-note-input').value = 'first';
    h.context.addLightNote(); h.context.addLightNote();
    assert.equal(h.el('light-note-input').value, 'first');
    assert.equal(sent.length, 1);
    h.el('light-note-input').value = 'new edit';
    h.context.handleWSMessage({ type: 'note_saved' });
    assert.equal(h.el('light-note-input').value, 'new edit');
    h.context.addLightNote();
    h.context.handleWSMessage({ type: 'note_saved' });
    assert.equal(h.el('light-note-input').value, '');
    h.el('light-note-input').value = 'keep after error';
    h.context.addLightNote();
    h.context.handleWSMessage({ type: 'error', message: 'disk full' });
    assert.equal(h.el('light-note-input').value, 'keep after error');
});
