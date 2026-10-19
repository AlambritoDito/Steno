"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const { EventEmitter } = require("node:events");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function desktopHarness() {
  const app = new EventEmitter();
  const ipcMain = new EventEmitter();
  const messages = [];
  const errors = [];
  let stops = 0;
  let exits = 0;
  let timeout;
  Object.assign(app, {
    setPath: () => {}, requestSingleInstanceLock: () => true,
    whenReady: () => new Promise(() => {}), getLocale: () => "en",
    exit: () => { exits++; },
  });
  const contents = { send: (name) => messages.push(name) };
  const context = vm.createContext({
    require: (name) => name === "electron" ? { app, ipcMain, dialog: { showErrorBox: (...args) => errors.push(args) } } : require(name),
    __dirname, process, URL,
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
    expire: () => timeout(), messages, errors,
    counts: () => ({ stops, exits }),
  };
}

test("desktop quit waits for trusted renderer save acknowledgment before stopping its backend", async () => {
  const desktop = desktopHarness();
  desktop.request();
  assert.deepEqual(desktop.messages, ["before-close"]);
  assert.deepEqual(desktop.counts(), { stops: 0, exits: 0 });
  desktop.acknowledge(true, false);
  assert.deepEqual(desktop.counts(), { stops: 0, exits: 0 });
  desktop.acknowledge(true);
  await new Promise(setImmediate);
  assert.deepEqual(desktop.counts(), { stops: 1, exits: 1 });
});

test("failed or timed out save cancels desktop quit without killing the backend", () => {
  const desktop = desktopHarness();
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
