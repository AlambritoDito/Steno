/* Electron preferences/drafts survive randomized backend ports. Browser mode stays local. */
const uiStorage = window.electronAPI?.storage || {
    getItem: key => localStorage.getItem(key),
    setItem: (key, value) => localStorage.setItem(key, value),
    removeItem: key => localStorage.removeItem(key),
    keys: () => Object.keys(localStorage),
};
