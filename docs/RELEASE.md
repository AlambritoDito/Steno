# Steno 1.0 desktop release verification

Status: **beta implementation; stable publication is blocked on the remaining hardware and clean-install gates below.** A successful build or mocked GPU test is not evidence of an installed application working on a user's computer.

Scope: Apple Silicon on macOS 27 (initial validation target 27.0.1), Windows 11 x64 with NVIDIA CUDA and Windows CPU. No public signing certificate or notarization is required for this release; Mac bundles still require a valid ad hoc signature. Steno Server has an independent release cycle.

## Implementation and issue map

This branch incorporates the desktop changes from PR #22, then replaces the unsafe persistence, recording and startup paths. Issues must only be closed after their acceptance checks pass on the applicable platforms.

| Tracking | Desktop disposition |
|---|---|
| #8, #9, #10 | Shared backend contract; MLX / faster-whisper; actual initialization probes; Windows CPU fallback; native NSIS build and app-local CUDA runtime. Windows release gate remains mandatory. |
| #4, #5, #6, #7, #13, #14 | Offline editor assets, per-take archived audio, retranscription, model/download state, diagnostics and repaired desktop lifecycle. Covered by focused regressions and real Mac inference; installer acceptance remains open. |
| #12 | Collect runtime dependencies and licenses; measure actual bundle size only after frozen inference passes. Do not remove native libraries merely to shrink the installer. |
| #11, #19, #20, #21 | README, architecture/API, diagnostics, release checklist and CI workflows added. Remaining manual/platform evidence belongs in this checklist. |
| #15–#18 | Deferred: system audio, speakers, search, sync/backup. |
| #23–#34 | Independent server backlog. |

## Additional defects recorded during implementation

| Finding | Repair / regression evidence |
|---|---|
| Legacy Markdown parser discards arbitrary note lines | Import the complete document as an immutable historical block; preserve originals and copy external media. Storage migration tests cover multiline text and colliding legacy names. |
| Repeated editor saves duplicate the complete note | Idempotent SQLite update; UI flushes pending notes before switch/export. Workflow tests save the same note three times and export once. |
| Resuming overwrites the only recording | Unique take ID and exclusive creation per Start; prior audio remains separately playable. |
| Stop loses recordings shorter than one chunk | Archive capture first, then process the final partial range; regression stops at two seconds. |
| Overlap is added twice | Single archive-range overlap; no second overlapping memory queue. |
| Slow inference grows audio memory indefinitely | Bounded callback queue and archived ranges; file decoding uses 30-second windows. A two-hour real-device soak is still required. |
| Stop/cancel races leave jobs permanently cancelling | Reserve cancellation before yielding; idempotent cancel; preserve the last committed cursor; tested with blocked native inference. |
| Device shutdown exception bypasses cleanup | Always close the stream and join the writer; preserve visible capture errors and saved samples. |
| WAV header can lag stored samples after interruption | Repair the owned PCM header on recovery before resuming jobs; test verifies appended samples survive. |
| Locked Markdown export resurrects a deleted session | Transactional deletion tombstone and retried filesystem cleanup, including Windows-style sharing failures. |
| Warm-up failure changes the active model prematurely | Warm a candidate, then atomically persist/swap selection; retain the prior active selection on failure. |
| Public model download inherits an invalid local HF credential | Explicit anonymous public downloads, independent of developer login. Confirmed by actual packaged tiny-model download. |
| Fixed-port cleanup can terminate an unrelated process | Bind an available loopback socket; terminate only the shell's child process. Test dynamic binding and owned shutdown. |
| Dynamic port changes browser storage origin | Atomic application storage survives port changes and preserves per-input drafts. The close acknowledgment protocol waits for saved notes; failed saves cancel exit. Node regressions cover both paths. |

## Evidence gathered locally

Environment: Apple Silicon M4 Pro, macOS 27.0.1, Python 3.12. Tests use isolated temporary data and do not inspect or modify personal recordings.

- Complete Python suite: 182 tests passed after the final runtime changes; final CI links are reported in the pull request.
- JavaScript tests cover Electron process ownership and frontend session/note/job behavior.
- Real Chrome checks exercised setup, English/Spanish UI, saved history, note editing, session isolation, audio controls and failed-job actions using an explicit fixture. The fixture bypassed model setup only; it is not a microphone or GPU test.
- A native PyInstaller backend downloaded `tiny`, warmed MLX/Metal, transcribed synthetic spoken audio, reported `validated=true`, rejected an incorrect auth token and shut down when its parent's pipe closed.
- The Mac DMG was built locally and the complete `.app` passed `codesign --verify --deep --strict`. This does not prove Gatekeeper acceptance of an internet-downloaded DMG on a clean machine.

## Required acceptance matrix

Record installer filename/SHA-256, commit, OS build, CPU/GPU/RAM, driver version, model, effective device, result and diagnostic log for each run. Do not include private recordings in reports.

| Gate | macOS 27.0.1 | Windows 11 NVIDIA | Windows 11 CPU |
|---|---|---|---|
| Native package build and authenticated backend startup | Local pass | Windows CI required | Windows CI required |
| Actual packaged speech inference with expected device | Metal local pass | **Pending real hardware** | Hosted CPU check and clean machine required |
| Install without Python, uv, Node or build tools | **Pending clean machine** | **Pending** | **Pending** |
| Launch internet-downloaded unsigned installer using documented system prompts | **Pending** | **Pending** | **Pending** |
| First model download, cancel, retry, repair incomplete files | Unit + local happy path; interruption manual pending | **Pending** | **Pending** |
| Restart and infer with network disconnected | **Pending actual packaged run** | **Pending** | **Pending** |
| Short capture, stop/drain, resume without overwriting, disconnect mic | Automated regressions; real microphone pending | **Pending** | **Pending** |
| Two-hour capture while inference is slower than real time | **Pending** | **Pending** | **Pending** |
| Close during processing; reopen and resume without duplicate text | Automated regressions; packaged manual pending | **Pending** | **Pending** |
| Notes/images/transcripts survive save, reopen, edit and portable export | Automated + browser checks; installed relaunch pending | **Pending** | **Pending** |
| Occupied port, double launch, manual upgrade and uninstall preserve data | Automated lifecycle checks; installed manual pending | **Pending** | **Pending** |
| Missing/incompatible driver and mid-job GPU failure preserve source / CPU retry | Not applicable | **Pending real hardware** | CPU path required |

## Reproducible commands

```sh
uv sync --frozen --extra dev --extra build
uv run --no-sync pytest tests/ -v
npm ci
npm run test:electron
npm run test:ui
npm run build
uv run --no-sync python scripts/smoke_backend.py
uv run --no-sync python scripts/smoke_backend.py --audio /path/to/spoken-fixture.wav --expect-device metal
uv run --no-sync python scripts/checksums.py
```

On Windows, replace the expected device with `cpu` or `cuda`. CUDA acceptance **must** assert `device=cuda` and `validated=true`; a silent CPU fallback cannot satisfy it. The GPU workflow requires an explicitly configured Windows NVIDIA runner and a local, non-private speech fixture. A hosted Windows runner can prove CPU operation, not NVIDIA operation.

`packaging/cuda-runtime.json` pins NVIDIA downloads and their SHA-256 hashes. The installer bundles CUDA 12.4.1, cuBLAS, cuDNN 9.1.1.17 and redistribution notices. It neither installs drivers nor modifies global PATH. Model weights remain separate downloads. The recommended Windows NVIDIA driver baseline (551.78+) is provisional until exercised by the CUDA gate.

Cancellation is cooperative between native inference calls and download progress callbacks. Closing the shell may interrupt an expensive in-flight call after its shutdown grace period; audio, committed segments and durable jobs must recover on the next launch.

## Publication order

1. Build and review `1.0.0-beta.1`. Attach native DMG/NSIS files, checksums, requirements, unsigned-install instructions, tested hardware and explicit remaining limitations. Do not publish a source-only release as an installer release.
2. Complete all applicable acceptance rows above and run a release candidate with the same frozen dependencies and packaging scripts. Repeat failed cases after repairs.
3. Publish `1.0.0` only when data preservation and clean installation are demonstrated for Mac, Windows CUDA and Windows CPU. Close the matching desktop blockers with evidence. Server and deferred features remain separate.

The workflows produce CI artifacts, not an automatic stable GitHub Release. Manual upgrades and uninstall preserve user data by default. Before testing a downgrade, back up the closed application's entire data folder; newer database schema versions deliberately reject older readers.
