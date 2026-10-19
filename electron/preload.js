"use strict";
const { contextBridge, ipcRenderer } = require("electron");
function storageCall(operation, key, value) {
  const response = ipcRenderer.sendSync("ui-storage", operation, key, value);
  if (!response || response.error) throw new Error(response?.error || "Application storage unavailable");
  return response.value;
}
contextBridge.exposeInMainWorld("electronAPI", {
  isElectron: true,
  storage: {
    getItem: (key) => storageCall("getItem", key),
    setItem: (key, value) => storageCall("setItem", key, String(value)),
    removeItem: (key) => storageCall("removeItem", key),
    keys: () => storageCall("keys"),
  },
  closeReady: (success) => ipcRenderer.send("close-ready", success === true),
});
ipcRenderer.on("before-close", () => {
  window.dispatchEvent(new Event("steno-before-close"));
});
ipcRenderer.on("open-settings", () => {
  window.dispatchEvent(new Event("steno-open-settings"));
});
