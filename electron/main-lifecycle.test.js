"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const { EventEmitter } = require("node:events");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const vm = require("node:vm");

function desktopHarness(t, { portable = false, blocked = false } = {}) {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "steno-startup-test-"));
  t.after(() => fs.rmSync(temporary, { recursive: true, force: true }));
  const dataDir = portable ? path.join(temporary, "Steno Data") : path.join(temporary, "fresh", "Application Data", "Steno");
  if (portable) fs.writeFileSync(path.join(temporary, "portable.json"), JSON.stringify({ format: "steno-portable", version: 1 }));
  if (blocked) fs.writeFileSync(path.join(temporary, "fresh"), "A file prevents this directory from being created");
  assert.equal(fs.existsSync(dataDir), false);
  const configuredPaths = {};
  const app = new EventEmitter();
  const ipcMain = new EventEmitter();
  const invokeHandlers = new Map();
  ipcMain.handle = (name, handler) => invokeHandlers.set(name, handler);
  const openedPaths = [];
  const messages = [];
  const errors = [];
  let stops = 0;
  let exits = 0;
  let timeout;
  Object.assign(app, {
    setPath: (name, directory) => {
      assert.equal(fs.statSync(directory).isDirectory(), true, "Electron setPath requires an existing directory");
      configuredPaths[name] = directory;
    }, requestSingleInstanceLock: () => true,
    isPackaged: portable, getPath: () => path.join(temporary, "Steno.exe"),
    whenReady: () => new Promise(() => {}), getLocale: () => "en",
    exit: () => { exits++; },
  });
  const contents = { send: (name) => messages.push(name) };
  const context = vm.createContext({
    require: (name) => name === "electron" ? { app, ipcMain,
      dialog: { showErrorBox: (...args) => errors.push(args) },
      shell: { openPath: async (directory) => { openedPaths.push(directory); return ""; } },
    } : require(name),
    __dirname, process: { platform: portable ? "win32" : process.platform,
      env: { ...process.env, STENO_DATA_DIR: portable ? "" : dataDir } }, URL,
    setTimeout: (callback) => { timeout = callback; return 1; }, clearTimeout: () => { timeout = null; },
    fixtureWindow: { webContents: contents, isDestroyed: () => false },
    fixtureBackend: { stop: async () => { stops++; } },
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "main.js"), "utf8"), context);
  vm.runInContext('mainWindow = fixtureWindow; backend = fixtureBackend; backendUrl = "http://127.0.0.1:23456"; rendererReady = true;', context);
  return {
    request: () => {
      let prevented = false;
      app.emit("before-quit", { preventDefault: () => { prevented = true; } });
      assert.equal(prevented, true);
    },
    acknowledge: (success, trusted = true) => ipcMain.emit("close-ready", {
      sender: contents, senderFrame: { url: trusted ? "http://127.0.0.1:23456/" : "https://example.com/" },
    }, success),
    openLogs: (trusted = true, extraPath) => invokeHandlers.get("open-logs")({
      sender: contents, senderFrame: { url: trusted ? "http://127.0.0.1:23456/" : "https://example.com/" },
    }, extraPath),
    expire: () => timeout(), messages, errors, dataDir, configuredPaths, openedPaths,
    counts: () => ({ stops, exits }),
  };
}

test("clean desktop startup creates absent data and Chromium session directories before setPath", (t) => {
  const desktop = desktopHarness(t);
  assert.deepEqual(desktop.configuredPaths, { userData: desktop.dataDir, sessionData: desktop.dataDir });
  assert.equal(fs.statSync(desktop.dataDir).isDirectory(), true);
});

test("portable Windows launcher creates data and logs beside its executable", (t) => {
  const desktop = desktopHarness(t, { portable: true });
  assert.deepEqual(desktop.configuredPaths, { userData: desktop.dataDir, sessionData: desktop.dataDir });
  assert.equal(fs.statSync(path.join(desktop.dataDir, "logs")).isDirectory(), true);
  assert.equal(path.basename(desktop.dataDir), "Steno Data");
});

test("unwritable startup reports a useful error and exits before configuring Electron paths", (t) => {
  const desktop = desktopHarness(t, { blocked: true });
  assert.deepEqual(desktop.configuredPaths, {});
  assert.deepEqual(desktop.counts(), { stops: 0, exits: 1 });
  assert.equal(desktop.errors.length, 1);
  assert.match(desktop.errors[0][1], /extract the entire ZIP into a writable folder/);
});

test("open logs accepts only the app frame and ignores caller-supplied paths", async (t) => {
  const desktop = desktopHarness(t);
  assert.equal((await desktop.openLogs(false, "/arbitrary")).ok, false);
  assert.deepEqual(desktop.openedPaths, []);
  assert.equal((await desktop.openLogs(true, "/arbitrary")).ok, true);
  assert.deepEqual(desktop.openedPaths, [path.join(desktop.dataDir, "logs")]);
});

test("desktop quit waits for trusted renderer save acknowledgment before stopping its backend", async (t) => {
  const desktop = desktopHarness(t);
  desktop.request();
  assert.deepEqual(desktop.messages, ["before-close"]);
  assert.deepEqual(desktop.counts(), { stops: 0, exits: 0 });
  desktop.acknowledge(true, false);
  assert.deepEqual(desktop.counts(), { stops: 0, exits: 0 });
  desktop.acknowledge(true);
  await new Promise(setImmediate);
  assert.deepEqual(desktop.counts(), { stops: 1, exits: 1 });
});

test("failed or timed out save cancels desktop quit without killing the backend", (t) => {
  const desktop = desktopHarness(t);
  desktop.request();
  desktop.acknowledge(false);
  assert.deepEqual(desktop.counts(), { stops: 0, exits: 0 });
  desktop.request();
  desktop.expire();
  assert.equal(desktop.errors.length, 1);
  desktop.acknowledge(true); // A late acknowledgment cannot silently quit later.
  assert.deepEqual(desktop.counts(), { stops: 0, exits: 0 });
  desktop.request();
  assert.equal(desktop.messages.length, 3);
});
