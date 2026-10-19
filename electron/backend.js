"use strict";

const { spawn } = require("node:child_process");
const http = require("node:http");
const path = require("node:path");
const crypto = require("node:crypto");

function backendCommand(packaged, resourcesPath, projectRoot, platform = process.platform) {
  const executable = platform === "win32" ? "steno-server.exe" : "steno-server";
  if (packaged) return [path.join(resourcesPath, "steno-server", executable), []];
  const python = platform === "win32" ? ["Scripts", "python.exe"] : ["bin", "python"];
  return [path.join(projectRoot, ".venv", ...python), [path.join(projectRoot, "main.py")]];
}

function checkHealth(port, token) {
  return new Promise((resolve) => {
    const request = http.get({
      hostname: "127.0.0.1", port, path: "/api/health", timeout: 1000,
      headers: { Authorization: `Bearer ${token}` },
    }, (response) => {
      let body = "";
      response.on("data", (chunk) => { body += chunk; if (body.length > 4096) request.destroy(new Error("Invalid health response")); });
      response.on("end", () => {
        try { resolve(response.statusCode === 200 && JSON.parse(body).status === "ok"); }
        catch (_) { resolve(false); }
      });
    });
    request.on("error", () => resolve(false));
    request.on("timeout", () => { request.destroy(); resolve(false); });
  });
}

class Backend {
  constructor({ command, args = [], env = {}, cwd, log = () => {}, onExit = () => {} }) {
    this.options = { command, args, env, cwd, log, onExit };
    this.token = crypto.randomBytes(32).toString("hex");
    this.child = null;
    this.stopping = false;
  }

  async start(timeout = 120000) {
    const { command, args, env, cwd, log, onExit } = this.options;
    return new Promise((resolve, reject) => {
      let settled = false;
      let pending = "";
      let port = null;
      const child = spawn(command, args, {
        cwd, windowsHide: true, stdio: ["pipe", "pipe", "pipe"],
        env: { ...process.env, ...env, STENO_ELECTRON: "1", STENO_PORT: "0", STENO_AUTH_TOKEN: this.token },
      });
      this.child = child;
      const timer = setTimeout(() => fail(new Error("Backend startup timed out")), timeout);
      const fail = (error) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        reject(error);
        void this.stop();
      };
      child.stdin.on("error", () => {});
      child.on("error", (error) => fail(new Error(`Backend could not start: ${error.message}`)));
      child.on("exit", (code, signal) => {
        this.child = null;
        if (!settled) fail(new Error(`Backend exited during startup (${code ?? signal})`));
        else if (!this.stopping) onExit(code, signal);
      });
      child.stderr.on("data", (data) => log(data.toString().replaceAll(this.token, "[redacted]")));
      child.stdout.on("data", (data) => {
        pending += data.toString();
        if (pending.length > 65536) return fail(new Error("Invalid backend startup output"));
        const lines = pending.split("\n");
        pending = lines.pop();
        for (const line of lines) {
          if (!line.startsWith("STENO_READY ")) {
            log(line.replaceAll(this.token, "[redacted]"));
            continue;
          }
          if (port !== null) continue;
          try {
            port = JSON.parse(line.slice(12)).port;
            if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error();
          } catch (_) { fail(new Error("Invalid backend readiness message")); continue; }
          const poll = async () => {
            if (settled) return;
            if (await checkHealth(port, this.token)) {
              if (settled) return;
              settled = true;
              clearTimeout(timer);
              this.url = `http://127.0.0.1:${port}`;
              resolve(this.url);
            } else if (!settled) setTimeout(poll, 100);
          };
          void poll();
        }
      });
    });
  }

  async stop() {
    this.stopping = true;
    const child = this.child;
    if (!child || child.exitCode !== null || child.signalCode !== null) return;
    await new Promise((resolve) => {
      const timer = setTimeout(() => {
        if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
      }, 5000);
      child.once("exit", () => { clearTimeout(timer); resolve(); });
      child.once("error", () => { clearTimeout(timer); resolve(); });
      // EOF asks our Python process to shut down on every platform.
      child.stdin.end();
      if (!child.pid) { clearTimeout(timer); resolve(); }
    });
  }
}

module.exports = { Backend, backendCommand, checkHealth };
