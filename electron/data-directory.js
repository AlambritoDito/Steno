"use strict";
const fs = require("node:fs");
const path = require("node:path");
const { randomUUID } = require("node:crypto");

function resolveDataDirectory({ platform, packaged, executable, override, localAppData, home }) {
  if (override) return path.resolve(override);
  if (platform === "win32" && packaged) {
    const folder = path.dirname(executable);
    const marker = path.join(folder, "portable.json");
    if (fs.existsSync(marker)) {
      const settings = JSON.parse(fs.readFileSync(marker, "utf8"));
      if (settings.format !== "steno-portable" || settings.version !== 1) throw new Error("Invalid portable edition marker");
      return path.join(folder, "Steno Data");
    }
  }
  return platform === "win32"
    ? path.join(localAppData || path.join(home, "AppData", "Local"), "Steno")
    : path.join(home, "Library", "Application Support", "Steno");
}

function prepareDataDirectory(directory) {
  // A real write catches Windows ACL restrictions that access(W_OK) can miss.
  fs.mkdirSync(directory, { recursive: true, mode: 0o700 });
  const probe = path.join(directory, `.write-test-${randomUUID()}`);
  const descriptor = fs.openSync(probe, "wx", 0o600);
  try { fs.writeSync(descriptor, "Steno"); }
  finally { fs.closeSync(descriptor); fs.unlinkSync(probe); }
  fs.mkdirSync(path.join(directory, "logs"), { recursive: true, mode: 0o700 });
}

module.exports = { resolveDataDirectory, prepareDataDirectory };
