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
process.exit(result.status ?? 1);
