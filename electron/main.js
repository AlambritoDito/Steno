/**
 * Steno — Electron main process.
 *
 * Spawns the Python/FastAPI backend as a child process,
 * waits for it to be ready, then opens a BrowserWindow.
 */

const { app, BrowserWindow, Menu, dialog, shell, systemPreferences } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const net = require("net");

const PORT = 8080;
const HOST = "127.0.0.1";
const SERVER_URL = `http://${HOST}:${PORT}`;
const STARTUP_TIMEOUT_MS = 60_000; // 60 s (model download can be slow)

let pythonProcess = null;
let mainWindow = null;
let electronLogStream = null;

// ---------------------------------------------------------------------------
// Electron-side log file (packaged app only)
// ---------------------------------------------------------------------------

function initElectronLog() {
  if (!app.isPackaged) return;
  const os = require("os");
  const logDir = path.join(os.homedir(), "Documents", "Steno", "logs");
  fs.mkdirSync(logDir, { recursive: true });
  const logFile = path.join(logDir, "electron.log");

  // Rotate: if existing log > 5 MB, move to .old
  try {
    const stats = fs.statSync(logFile);
    if (stats.size > 5 * 1024 * 1024) {
      fs.renameSync(logFile, logFile + ".old");
    }
  } catch (_) {
    // File doesn't exist yet — fine
  }

  electronLogStream = fs.createWriteStream(logFile, { flags: "a" });
  electronLog("=== Steno Electron started ===");
}

function electronLog(msg) {
  const line = `[${new Date().toISOString()}] ${msg}\n`;
  if (electronLogStream) electronLogStream.write(line);
}

// ---------------------------------------------------------------------------
// Python server management
// ---------------------------------------------------------------------------

async function startPythonServer() {
  // Check for a stale server occupying the port
  const occupied = await isPortOccupied(PORT, HOST);
  if (occupied) {
    electronLog(`Port ${PORT} already occupied — attempting to kill stale process`);
    try {
      const { execSync } = require("child_process");
      const lsofOut = execSync(`lsof -ti tcp:${PORT}`, { encoding: "utf-8" }).trim();
      if (lsofOut) {
        for (const pidStr of lsofOut.split("\n")) {
          const stalePid = parseInt(pidStr, 10);
          if (stalePid > 0) {
            electronLog(`Killing stale process PID ${stalePid} on port ${PORT}`);
            try { process.kill(stalePid, "SIGKILL"); } catch (_) {}
          }
        }
        // Brief pause to let the OS release the port
        await new Promise((r) => setTimeout(r, 1000));
      }
    } catch (_) {
      // lsof may fail if no process found — ignore
    }
  }

  return new Promise((resolve, reject) => {
    const env = { ...process.env, STENO_ELECTRON: "1" };

    if (app.isPackaged) {
      // Production — use the PyInstaller binary bundled as an extra resource
      const bin = path.join(
        process.resourcesPath,
        "steno-server",
        "steno-server"
      );

      // Ensure the binary exists before trying to spawn it
      if (!fs.existsSync(bin)) {
        reject(new Error(`Backend binary not found at: ${bin}`));
        return;
      }

      // Ensure HOME is set (needed for ~/.cache/huggingface model downloads)
      env.HOME = env.HOME || require("os").homedir();

      pythonProcess = spawn(bin, [], { env, detached: true, stdio: ["ignore", "pipe", "pipe"] });
    } else {
      // Development — use uv from the project root
      const cwd = path.join(__dirname, "..");
      pythonProcess = spawn("uv", ["run", "main.py"], { cwd, env, detached: true, stdio: ["ignore", "pipe", "pipe"] });
    }

    let stderrBuffer = "";

    pythonProcess.stdout.on("data", (d) => {
      process.stdout.write(`[py] ${d}`);
      electronLog(`[py:stdout] ${d.toString().trimEnd()}`);
    });
    pythonProcess.stderr.on("data", (d) => {
      const text = d.toString();
      stderrBuffer += text;
      process.stderr.write(`[py] ${text}`);
      electronLog(`[py:stderr] ${text.trimEnd()}`);
    });

    pythonProcess.on("error", (err) => {
      reject(new Error(`Failed to start Python backend: ${err.message}`));
    });

    pythonProcess.on("exit", (code) => {
      if (code !== null && code !== 0) {
        const hint = stderrBuffer.slice(-500);
        reject(new Error(
          `Python backend exited with code ${code}\n\nLast output:\n${hint}`
        ));
      }
    });

    waitForPort(PORT, HOST, STARTUP_TIMEOUT_MS).then(resolve).catch(reject);
  });
}

function stopPythonServer() {
  if (!pythonProcess) return;

  const pid = pythonProcess.pid;
  electronLog(`Stopping Python server (PID ${pid})`);

  // Kill the entire process group so child processes (uvicorn workers) also die
  try {
    process.kill(-pid, "SIGTERM");
  } catch (_) {
    // Process group kill failed — fall back to direct kill
    try { pythonProcess.kill("SIGTERM"); } catch (_) { /* already dead */ }
  }

  // Force-kill after 3 seconds if still alive
  const forceKillTimer = setTimeout(() => {
    try {
      process.kill(-pid, "SIGKILL");
    } catch (_) {
      try { process.kill(pid, "SIGKILL"); } catch (_) { /* already dead */ }
    }
    electronLog(`Force-killed Python server (PID ${pid})`);
  }, 3000);

  // Don't let the timer prevent Node from exiting
  if (forceKillTimer.unref) forceKillTimer.unref();

  pythonProcess = null;
}

/**
 * Check if a port is already in use.  Returns true if occupied.
 */
function isPortOccupied(port, host) {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    socket.setTimeout(1000);
    socket.on("connect", () => { socket.destroy(); resolve(true); });
    socket.on("error", () => { socket.destroy(); resolve(false); });
    socket.on("timeout", () => { socket.destroy(); resolve(false); });
    socket.connect(port, host);
  });
}

// ---------------------------------------------------------------------------
// Port readiness helper
// ---------------------------------------------------------------------------

function waitForPort(port, host, timeout) {
  return new Promise((resolve, reject) => {
    const start = Date.now();

    function tryConnect() {
      if (Date.now() - start > timeout) {
        reject(new Error("Server startup timed out"));
        return;
      }

      const socket = new net.Socket();
      socket.setTimeout(1000);

      socket.on("connect", () => {
        socket.destroy();
        resolve();
      });

      socket.on("error", () => {
        socket.destroy();
        setTimeout(tryConnect, 500);
      });

      socket.on("timeout", () => {
        socket.destroy();
        setTimeout(tryConnect, 500);
      });

      socket.connect(port, host);
    }

    tryConnect();
  });
}

// ---------------------------------------------------------------------------
// Application Menu (macOS standard with Cmd+, for Settings)
// ---------------------------------------------------------------------------

function buildAppMenu() {
  const template = [
    {
      label: app.name,
      submenu: [
        { role: "about" },
        { type: "separator" },
        {
          label: "Settings…",
          accelerator: "CmdOrCtrl+,",
          click: () => {
            if (mainWindow) {
              mainWindow.webContents.send("open-settings");
              // Also call via JS in case preload bridge isn't set up
              mainWindow.webContents.executeJavaScript(
                "if(typeof showSettingsModal==='function')showSettingsModal();"
              ).catch(() => {});
            }
          },
        },
        { type: "separator" },
        { role: "services" },
        { type: "separator" },
        { role: "hide" },
        { role: "hideOthers" },
        { role: "unhide" },
        { type: "separator" },
        { role: "quit" },
      ],
    },
    {
      label: "Edit",
      submenu: [
        { role: "undo" },
        { role: "redo" },
        { type: "separator" },
        { role: "cut" },
        { role: "copy" },
        { role: "paste" },
        { role: "selectAll" },
      ],
    },
    {
      label: "View",
      submenu: [
        { role: "reload" },
        { role: "forceReload" },
        ...(app.isPackaged ? [] : [{ role: "toggleDevTools" }]),
        { type: "separator" },
        { role: "resetZoom" },
        { role: "zoomIn" },
        { role: "zoomOut" },
        { type: "separator" },
        { role: "togglefullscreen" },
      ],
    },
    {
      label: "Window",
      submenu: [
        { role: "minimize" },
        { role: "zoom" },
        { type: "separator" },
        { role: "front" },
      ],
    },
  ];

  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

// ---------------------------------------------------------------------------
// Window
// ---------------------------------------------------------------------------

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 900,
    minHeight: 600,
    titleBarStyle: "hiddenInset",
    trafficLightPosition: { x: 16, y: 16 },
    title: "Steno",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadURL(SERVER_URL);

  // Open external links in the default browser
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

// ---------------------------------------------------------------------------
// App lifecycle
// ---------------------------------------------------------------------------

app.whenReady().then(async () => {
  initElectronLog();
  try {
    // Request microphone access on macOS before starting
    if (process.platform === "darwin") {
      const micStatus = systemPreferences.getMediaAccessStatus("microphone");
      if (micStatus !== "granted") {
        await systemPreferences.askForMediaAccess("microphone");
      }
    }

    buildAppMenu();
    await startPythonServer();
    createWindow();
  } catch (err) {
    const message = err.message || String(err);
    let detail = `Could not start the transcription engine.\n\n${message}`;
    if (app.isPackaged) {
      detail += `\n\nResources path: ${process.resourcesPath}`;
    }
    dialog.showErrorBox("Steno", detail);
    app.quit();
  }
});

app.on("window-all-closed", () => {
  stopPythonServer();
  app.quit();
});

app.on("before-quit", stopPythonServer);

app.on("activate", () => {
  // macOS: re-create window when clicking the dock icon
  if (mainWindow === null) {
    createWindow();
  }
});
