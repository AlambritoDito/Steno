"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { UiStorage } = require("./ui-storage");

test("preferences and unsubmitted notes survive application launches and origin changes", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "steno-ui-test-"));
  try {
    const file = path.join(directory, "ui-state.json");
    const firstLaunch = new UiStorage(file);
    firstLaunch.invoke("setItem", "steno_ui_lang", "es");
    firstLaunch.invoke("setItem", "steno_draft_session-123", "Unsaved notes\nwith accents: á");
    const secondLaunch = new UiStorage(file);
    assert.equal(secondLaunch.invoke("getItem", "steno_ui_lang"), "es");
    assert.equal(secondLaunch.invoke("getItem", "steno_draft_session-123"), "Unsaved notes\nwith accents: á");
    secondLaunch.invoke("removeItem", "steno_draft_session-123");
    assert.equal(new UiStorage(file).invoke("getItem", "steno_draft_session-123"), null);
    assert.deepEqual(secondLaunch.invoke("keys"), ["steno_ui_lang"]);
  } finally { fs.rmSync(directory, { recursive: true, force: true }); }
});

test("invalid or oversized renderer writes cannot replace existing state", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "steno-ui-test-"));
  try {
    const store = new UiStorage(path.join(directory, "ui-state.json"));
    store.invoke("setItem", "steno_ui_lang", "en");
    assert.throws(() => store.invoke("setItem", "../../arbitrary", "data"), /Unsupported/);
    assert.throws(() => store.invoke("setItem", "__proto__", "data"), /Unsupported/);
    assert.throws(() => store.invoke("setItem", "steno_draft_123", "x".repeat(11 * 1024 * 1024)), /10 MB/);
    assert.deepEqual(new UiStorage(store.file).invoke("keys"), ["steno_ui_lang"]);
  } finally { fs.rmSync(directory, { recursive: true, force: true }); }
});

test("malformed preferences are preserved for recovery without preventing startup", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "steno-ui-test-"));
  try {
    const file = path.join(directory, "ui-state.json");
    fs.writeFileSync(file, '{"steno_draft_123":"partial');
    const warnings = [];
    const store = new UiStorage(file, (warning) => warnings.push(warning));
    assert.deepEqual(store.invoke("keys"), []);
    const backup = fs.readdirSync(directory).find((name) => name.includes(".corrupt-"));
    assert.equal(fs.readFileSync(path.join(directory, backup), "utf8"), '{"steno_draft_123":"partial');
    assert.equal(warnings.length, 1);
    store.invoke("setItem", "steno_ui_lang", "es");
    assert.equal(new UiStorage(file).invoke("getItem", "steno_ui_lang"), "es");
  } finally { fs.rmSync(directory, { recursive: true, force: true }); }
});
