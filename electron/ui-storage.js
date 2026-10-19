"use strict";

const fs = require("node:fs");
const path = require("node:path");

const preferences = new Set([
  "steno_ui_lang", "steno_transcription_lang", "steno_mic_index",
  "steno_last_session", "steno_update_dismissed",
]);

function allowedKey(key) {
  return typeof key === "string" && (preferences.has(key) || /^steno_draft_[a-zA-Z0-9_-]{1,180}$/.test(key));
}

class UiStorage {
  constructor(file, onWarning = () => {}) {
    this.file = file;
    this.values = Object.create(null);
    if (fs.existsSync(file)) {
      const contents = fs.readFileSync(file, "utf8");
      let saved;
      try {
        saved = JSON.parse(contents);
        if (!saved || typeof saved !== "object" || Array.isArray(saved)) throw new SyntaxError("Invalid application preferences");
      } catch (error) {
        if (!(error instanceof SyntaxError)) throw error;
        const backup = `${file}.corrupt-${Date.now()}.json`;
        fs.renameSync(file, backup);
        onWarning(`Unreadable UI preferences preserved at ${backup}`);
        return;
      }
      for (const [key, value] of Object.entries(saved)) {
        if (allowedKey(key) && typeof value === "string") this.values[key] = value;
      }
    }
  }

  invoke(operation, key, value) {
    if (operation === "keys") return Object.keys(this.values);
    if (!allowedKey(key)) throw new Error("Unsupported application preference");
    if (operation === "getItem") return this.values[key] ?? null;
    const next = { ...this.values };
    if (operation === "setItem" && typeof value === "string") next[key] = value;
    else if (operation === "removeItem") delete next[key];
    else throw new Error("Invalid preference operation");
    const serialized = JSON.stringify(next);
    if (Buffer.byteLength(serialized) > 10 * 1024 * 1024) throw new Error("Application preferences and drafts exceed 10 MB");
    fs.mkdirSync(path.dirname(this.file), { recursive: true });
    const temporary = `${this.file}.tmp`;
    const descriptor = fs.openSync(temporary, "w", 0o600);
    try {
      fs.writeFileSync(descriptor, serialized, "utf8");
      fs.fsyncSync(descriptor);
    } finally {
      fs.closeSync(descriptor);
    }
    fs.renameSync(temporary, this.file);
    this.values = next;
    return null;
  }
}

module.exports = { UiStorage, allowedKey };
