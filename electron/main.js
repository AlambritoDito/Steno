"use strict";

const { app, BrowserWindow, Menu, dialog, shell, session, systemPreferences, ipcMain } = require("electron");
const path = require("node:path");
const os = require("node:os");
const fs = require("node:fs");
const { Backend, backendCommand } = require("./backend");
const { UiStorage } = require("./ui-storage");
const { resolveDataDirectory, prepareDataDirectory } = require("./data-directory");

// The portable ZIP carries a marker beside its executable; installers never do.
let dataDir;
let startupError;
try {
  dataDir = resolveDataDirectory({
    platform: process.platform, packaged: app.isPackaged,
    executable: process.platform === "win32" && app.isPackaged ? app.getPath("exe") : null,
    override: process.env.STENO_DATA_DIR, localAppData: process.env.LOCALAPPDATA, home: os.homedir(),
  });
  prepareDataDirectory(dataDir);
  // Electron requires overridden paths to exist, including on a clean install.
  app.setPath("userData", dataDir);
  app.setPath("sessionData", dataDir);
} catch (error) { startupError = error; }
let mainWindow;
let backend;
let quitting = false;
let backendUrl;
let uiStorage;
let rendererReady = false;
let closePending = false;
let closeTimer;
const spanish = () => {
  try { return (app.getLocale() || Intl.DateTimeFormat().resolvedOptions().locale).startsWith("es"); }
  catch (_) { return Intl.DateTimeFormat().resolvedOptions().locale.startsWith("es"); }
};
const text = (en, es) => spanish() ? es : en;

function writeLog(message) {
  try {
    const directory = path.join(dataDir, "logs");
    fs.mkdirSync(directory, { recursive: true });
    const file = path.join(directory, "electron.log");
    if (fs.existsSync(file) && fs.statSync(file).size > 5 * 1024 * 1024) {
      fs.rmSync(file + ".old", { force: true });
      fs.renameSync(file, file + ".old");
    }
    fs.appendFileSync(file, `[${new Date().toISOString()}] ${message}\n`);
  } catch (_) { /* Logging must not prevent startup or shutdown. */ }
}

function isTrustedUrl(url) {
  try { return new URL(url).origin === backendUrl; } catch (_) { return false; }
}

function trustedSender(event) {
  return mainWindow && event.sender === mainWindow.webContents && isTrustedUrl(event.senderFrame?.url);
}

ipcMain.on("ui-storage", (event, operation, key, value) => {
  if (!trustedSender(event)) { event.returnValue = { error: "Untrusted application frame" }; return; }
  try { event.returnValue = { value: uiStorage.invoke(operation, key, value) }; }
  catch (error) { event.returnValue = { error: error.message }; }
});

ipcMain.handle("open-logs", async (event) => {
  if (!trustedSender(event)) return { ok: false, error: "Untrusted application frame" };
  try {
    const directory = path.join(dataDir, "logs");
    fs.mkdirSync(directory, { recursive: true });
    const error = await shell.openPath(directory);
    if (error) throw new Error(error);
    return { ok: true };
  } catch (error) { return { ok: false, error: error.message }; }
});

function finishQuit() {
  quitting = true;
  clearTimeout(closeTimer);
  Promise.resolve(backend?.stop()).finally(() => app.exit());
}

function requestQuit() {
  if (quitting || closePending) return;
  if (!rendererReady || !mainWindow || mainWindow.isDestroyed()) { finishQuit(); return; }
  closePending = true;
  mainWindow.webContents.send("before-close");
  closeTimer = setTimeout(() => {
    closePending = false;
    dialog.showErrorBox("Steno", text(
      "Steno is still saving. Wait for your notes or recording to finish, then close the app again.",
      "Steno sigue guardando. Espera a que terminen tus notas o grabación y vuelve a cerrar la aplicación.",
    ));
  }, 15000);
}

ipcMain.on("close-ready", (event, success) => {
  if (!trustedSender(event) || !closePending) return;
  clearTimeout(closeTimer);
  closePending = false;
  if (success === true) finishQuit();
});

function openExternal(url) {
  try {
    if (["https:", "http:"].includes(new URL(url).protocol)) void shell.openExternal(url);
  } catch (_) { /* Reject malformed URLs and non-web protocols. */ }
}

async function createWindow() {
  const appSession = session.fromPartition("persist:steno-desktop");
  await appSession.clearStorageData({ storages: ["cookies"] });
  await appSession.cookies.set({
    url: backendUrl, name: "steno_auth", value: backend.token,
    httpOnly: true, sameSite: "strict", path: "/", secure: false,
  });
  appSession.setPermissionRequestHandler((contents, permission, callback, details) => {
    const audioOnly = !details.mediaTypes || details.mediaTypes.every((type) => type === "audio");
    callback(isTrustedUrl(contents.getURL()) && permission === "media" && audioOnly);
  });
  appSession.setPermissionCheckHandler((contents, permission, origin) => {
    return isTrustedUrl(origin) && permission === "media";
  });
  mainWindow = new BrowserWindow({
    width: 1280, height: 820, minWidth: 900, minHeight: 600, title: "Steno",
    ...(process.platform === "darwin" ? { titleBarStyle: "hiddenInset", trafficLightPosition: { x: 16, y: 16 } } : {}),
    webPreferences: {
      session: appSession, preload: path.join(__dirname, "preload.js"),
      contextIsolation: true, nodeIntegration: false, sandbox: true,
    },
  });
  mainWindow.webContents.on("will-navigate", (event, url) => {
    if (!isTrustedUrl(url)) { event.preventDefault(); openExternal(url); }
  });
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    openExternal(url);
    return { action: "deny" };
  });
  mainWindow.on("closed", () => { mainWindow = null; });
  mainWindow.on("close", (event) => {
    if (!quitting) { event.preventDefault(); requestQuit(); }
  });
  await mainWindow.loadURL(backendUrl);
  rendererReady = true;
}

function buildMenu() {
  const settings = {
    label: text("Settings…", "Configuración…"), accelerator: "CmdOrCtrl+,",
    click: () => mainWindow?.webContents.send("open-settings"),
  };
  const menus = [
    ...(process.platform === "darwin" ? [{ role: "appMenu", submenu: [
      { role: "about" }, settings, { type: "separator" }, { role: "hide" }, { role: "quit" },
    ] }] : [{ label: text("File", "Archivo"), submenu: [settings, { role: "quit" }] }]),
    { role: "editMenu" },
    { label: text("View", "Ver"), submenu: [
      { role: "resetZoom" }, { role: "zoomIn" }, { role: "zoomOut" },
      ...(app.isPackaged ? [] : [{ role: "toggleDevTools" }]),
    ] },
    { role: "windowMenu" },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(menus));
}

if (startupError) {
  dialog.showErrorBox("Steno", text(
    "Steno cannot write to its data folder. For the portable edition, extract the entire ZIP into a writable folder such as Documents, then open Steno.exe again. For an installed copy, check that your application data folder is writable.",
    "Steno no puede escribir en su carpeta de datos. Para la edición portátil, extrae todo el ZIP en una carpeta donde puedas guardar archivos, como Documentos, y abre Steno.exe de nuevo. Si lo instalaste, comprueba que puedes escribir en la carpeta de datos de la aplicación.",
  ));
  app.exit(1);
} else if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (mainWindow?.isMinimized()) mainWindow.restore();
    mainWindow?.focus();
  });
  app.whenReady().then(async () => {
    try {
      fs.mkdirSync(dataDir, { recursive: true });
      uiStorage = new UiStorage(path.join(dataDir, "ui-state.json"), writeLog);
      buildMenu();
      if (process.platform === "darwin" && systemPreferences.getMediaAccessStatus("microphone") !== "granted") {
        await systemPreferences.askForMediaAccess("microphone");
      }
      const projectRoot = path.join(__dirname, "..");
      const [command, args] = backendCommand(app.isPackaged, process.resourcesPath, projectRoot);
      backend = new Backend({
        command, args, cwd: app.isPackaged ? dataDir : projectRoot,
        env: { STENO_DATA_DIR: dataDir }, log: writeLog,
        onExit: () => {
          if (quitting) return;
          rendererReady = false;
          dialog.showErrorBox("Steno", text(
            "The transcription engine stopped. Reopen Steno to continue. Your saved sessions are preserved.",
            "El motor de transcripción se detuvo. Abre Steno de nuevo para continuar. Tus sesiones guardadas se conservan.",
          ));
          app.quit();
        },
      });
      backendUrl = await backend.start();
      await createWindow();
    } catch (error) {
      writeLog(error.message);
      dialog.showErrorBox("Steno", text(
        "Could not start Steno. See the logs in the application data folder. For source development run uv sync first.",
        "No se pudo iniciar Steno. Consulta los registros en la carpeta de datos. Para desarrollo ejecuta uv sync primero.",
      ));
      app.quit();
    }
  });
  app.on("window-all-closed", () => app.quit());
  app.on("before-quit", (event) => {
    if (quitting) return;
    event.preventDefault();
    requestQuit();
  });
}
