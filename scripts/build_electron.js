"use strict";
const { spawnSync } = require("node:child_process");
const path = require("node:path");
if (!["darwin", "win32"].includes(process.platform)) throw new Error("Build on macOS arm64 or Windows x64");
const cli = require.resolve("electron-builder/out/cli/cli.js");
const platform = process.platform === "darwin" ? ["--mac", "--arm64"] : ["--win", "--x64"];
const result = spawnSync(process.execPath, [cli, ...platform, "--publish", "never"], {
  cwd: path.join(__dirname, ".."), stdio: "inherit",
  env: { ...process.env, CSC_IDENTITY_AUTO_DISCOVERY: "false" },
});
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status ?? 1);
if (process.platform === "win32") {
  const portable = spawnSync(path.join(__dirname, "..", ".venv", "Scripts", "python.exe"), [path.join(__dirname, "build_portable.py")], {
    cwd: path.join(__dirname, ".."), stdio: "inherit",
  });
  if (portable.error) throw portable.error;
  process.exit(portable.status ?? 1);
}
