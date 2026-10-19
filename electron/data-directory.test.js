"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { resolveDataDirectory } = require("./data-directory");

test("installed Windows keeps native data storage and only the packaged portable marker opts into adjacent data", (t) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "steno-mode-test-"));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const options = { platform: "win32", packaged: true, executable: path.join(directory, "Steno.exe"),
    home: path.join(directory, "home"), localAppData: path.join(directory, "LocalAppData") };
  assert.equal(resolveDataDirectory(options), path.join(options.localAppData, "Steno"));
  fs.writeFileSync(path.join(directory, "portable.json"), JSON.stringify({ format: "steno-portable", version: 1 }));
  assert.equal(resolveDataDirectory(options), path.join(directory, "Steno Data"));
  assert.equal(resolveDataDirectory({ ...options, packaged: false }), path.join(options.localAppData, "Steno"));
  assert.equal(resolveDataDirectory({ ...options, override: path.join(directory, "isolated") }), path.join(directory, "isolated"));
});

test("invalid portable marker fails explicitly instead of silently switching users data", (t) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "steno-mode-test-"));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  fs.writeFileSync(path.join(directory, "portable.json"), JSON.stringify({ dataDirectory: "/arbitrary" }));
  assert.throws(() => resolveDataDirectory({ platform: "win32", packaged: true,
    executable: path.join(directory, "Steno.exe"), home: directory }), /Invalid portable/);
});
