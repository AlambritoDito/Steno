/**
 * Steno — Electron main process.
 *
 * Spawns the Python/FastAPI backend as a child process,
 * waits for it to be ready, then opens a BrowserWindow.
 */

const { app, BrowserWindow, dialog, shell } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const net = require("net");

const PORT = 8080;
const HOST = "127.0.0.1";
const SERVER_URL = `http://${HOST}:${PORT}`;
const STARTUP_TIMEOUT_MS = 60_000; // 60 s (model download can be slow)

let pythonProcess = null;
let mainWindow = null;

// ---------------------------------------------------------------------------
// Python server management
// ---------------------------------------------------------------------------

function startPythonServer() {
  return new Promise((resolve, reject) => {
    const env = { ...process.env, STENO_ELECTRON: "1" };

    if (app.isPackaged) {
      // Production — use the PyInstaller binary bundled as an extra resource
      const bin = path.join(
        process.resourcesPath,
        "steno-server",
        "steno-server"
      );
      pythonProcess = spawn(bin, [], { env });
    } else {
      // Development — use uv from the project root
      const cwd = path.join(__dirname, "..");
      pythonProcess = spawn("uv", ["run", "main.py"], { cwd, env });
    }

    pythonProcess.stdout.on("data", (d) => process.stdout.write(`[py] ${d}`));
    pythonProcess.stderr.on("data", (d) => process.stderr.write(`[py] ${d}`));

    pythonProcess.on("error", (err) => {
      reject(new Error(`Failed to start Python backend: ${err.message}`));
    });

    pythonProcess.on("exit", (code) => {
      if (code !== null && code !== 0) {
        reject(new Error(`Python backend exited with code ${code}`));
      }
    });

    waitForPort(PORT, HOST, STARTUP_TIMEOUT_MS).then(resolve).catch(reject);
  });
}

function stopPythonServer() {
  if (pythonProcess) {
    pythonProcess.kill();
    pythonProcess = null;
  }
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
  try {
    await startPythonServer();
    createWindow();
  } catch (err) {
    dialog.showErrorBox(
      "Steno",
      `Could not start the transcription engine.\n\n${err.message}`
    );
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
