"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const http = require("node:http");
const { Backend, backendCommand, checkHealth } = require("./backend");

const fixture = `
const http = require('node:http');
const server = http.createServer((req, res) => {
  const valid = req.headers.authorization === 'Bearer ' + process.env.STENO_AUTH_TOKEN;
  res.writeHead(valid ? 200 : 401, {'Content-Type':'application/json'});
  res.end(JSON.stringify({status: valid ? 'ok' : 'unauthorized'}));
});
server.listen(0, '127.0.0.1', () => {
  console.log('STENO_READY ' + JSON.stringify({port:server.address().port}));
});
process.stdin.resume();
process.stdin.on('end', () => server.close());
`;

test("backend authenticates readiness on an OS assigned port and closes only its child", async () => {
  const unrelated = http.createServer((_req, response) => response.end("other application"));
  await new Promise((resolve) => unrelated.listen(0, "127.0.0.1", resolve));
  const backend = new Backend({ command: process.execPath, args: ["-e", fixture] });
  try {
    const url = await backend.start(5000);
    assert.match(url, /^http:\/\/127\.0\.0\.1:\d+$/);
    assert.notEqual(Number(new URL(url).port), unrelated.address().port);
    assert.equal(await checkHealth(new URL(url).port, "wrong-token"), false);
    assert.equal(await checkHealth(new URL(url).port, backend.token), true);
    await backend.stop();
    assert.equal(backend.child, null);
    assert.equal(unrelated.listening, true);
  } finally {
    await backend.stop();
    await new Promise((resolve) => unrelated.close(resolve));
  }
});

test("startup rejects an exited child instead of trusting another HTTP service", async () => {
  const backend = new Backend({ command: process.execPath, args: ["-e", "process.exit(3)"] });
  await assert.rejects(backend.start(3000), /exited during startup/);
});

test("invalid handshake is rejected and the launched process is stopped", async () => {
  const backend = new Backend({ command: process.execPath, args: ["-e", "console.log('STENO_READY {}'); process.stdin.resume(); process.stdin.on('end', () => process.exit())"] });
  await assert.rejects(backend.start(3000), /Invalid backend readiness/);
  await backend.stop();
});

test("health response must be authenticated and identify ready state", async () => {
  const server = http.createServer((_req, response) => response.end('{"status":"other"}'));
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try { assert.equal(await checkHealth(server.address().port, "token"), false); }
  finally { await new Promise((resolve) => server.close(resolve)); }
});

test("launch chooses native executable and direct venv Python without a shell", () => {
  assert.match(backendCommand(true, "/resources", "/project", "win32")[0], /steno-server\.exe$/);
  assert.match(backendCommand(false, "/resources", "/project", "win32")[0], /Scripts[\\/]python\.exe$/);
  assert.deepEqual(backendCommand(true, "/resources", "/project", "darwin"), ["/resources/steno-server/steno-server", []]);
});
