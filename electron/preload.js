/**
 * Steno — Electron preload script.
 *
 * Exposes a minimal API so the renderer can detect it's running
 * inside Electron (e.g. to hide "open in browser" links).
 */

const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  isElectron: true,
});
