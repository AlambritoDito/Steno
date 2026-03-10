# Changelog

All notable changes to Steno are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/) with pre-release labels (`-alpha.N`, `-beta.N`).

---

## [0.2.0] — 2026-03-09

First stable Mac release targeting non-technical users.

### Added
- **Auto-update notifications** — checks GitHub Releases on startup and every 4 hours, shows a dismissable banner when a new version is available (#new)
- **Real download progress** — model downloads show actual MB downloaded/total with polling instead of fake progress bar (#13)
- **Download progress API** — new `GET /api/download-progress` endpoint for real-time download tracking
- **Update check API** — new `GET /api/update-check` endpoint using GitHub Releases API
- **Styled confirmation dialogs** — all native `alert()`/`confirm()` calls replaced with app-styled modals
- **ARIA accessibility** — added `role`, `aria-label`, `aria-live`, and `aria-modal` attributes to key interactive elements and modals
- New locale keys for update notifications, PortAudio errors, and debug hints (EN + ES)

### Fixed
- **[Critical]** marked.js loaded from CDN broke offline usage — now vendored locally in `static/vendor/` (#4)
- **[Critical]** Missing PortAudio caused cryptic crash — now detected gracefully with user-friendly banner and disabled recording (#5)
- **[Critical]** Memory leak in long recording sessions — append-only file save during recording, proper async task lifecycle management, session eviction on disconnect, and audio buffer cleanup (#6, #14)
- **[Critical]** Model not unloaded from RAM when switching — explicit `unload_model()` with `gc.collect()` and `mlx.core.metal.clear_cache()` (#7)
- Raw JavaScript error messages no longer exposed to users in download failures
- Generic "Error loading models" replaced with actionable user-friendly message

### Changed
- Debug panel hidden behind Cmd+Shift+D keyboard shortcut (was visible by default in Settings)
- DevTools removed from View menu in production builds
- Debug API endpoint `/api/debug/info` guarded in production (returns 403 unless `STENO_DEBUG` env is set)
- App reset accessible via `/api/reset` (in addition to `/api/debug/reset`)
- Bundle size reduced via PyInstaller `--strip` flag and `--exclude-module` for unused packages (tkinter, matplotlib, scipy, etc.) (#12)
- Build script includes post-build cleanup of `__pycache__` and test directories

### Removed
- CDN dependency on marked.js (replaced with vendored copy)
- Native browser `alert()`/`confirm()` dialogs (replaced with styled modals)

---

## [0.1.0-alpha.1] — 2026-03-06

First tagged release. The app is functional as a proof of concept but is still rough around the edges — expect breaking changes between alpha builds.

### Added
- **Real-time transcription** via [mlx-whisper](https://github.com/ml-explore/mlx-examples) running fully on-device (Apple Silicon, no cloud)
- **Multi-model support** — download and switch between Whisper tiny / base / small / medium / large-v3 from the UI
- **Multi-language transcription** — auto-detect or choose from EN, ES, FR, DE, IT, PT, JA, KO, ZH
- **Session management** — create named sessions, save transcripts and notes per session, view/delete past sessions
- **Notes editor** with Markdown preview, formatting toolbar, and image drag & drop
- **Electron desktop wrapper** — standalone macOS app (arm64 DMG) with native menu, Cmd+, Settings shortcut
- **Settings / Debug panel** — live event log, system info, app version display, one-click app reset
- **Audio file transcription** — drag & drop or upload audio files for offline transcription
- **Branding** — app icon, dark/light mode logo variants, favicon
- **i18n** — English and Spanish UI strings
- **User preference persistence** — mic device and transcription language saved across sessions

### Fixed
- Recording pipeline thread-safety: `asyncio.Queue` now fed via `loop.call_soon_threadsafe` from the sounddevice callback
- Logo dark/light mode swap was inverted in CSS defaults
- Model selector silently failed when no models were cached; now falls back to `/api/status` and surfaces errors in the UI
- Model detection self-heals: HuggingFace cache is scanned on startup so models present after an app reset are recognised without re-downloading

### Changed
- Notes editor replaced CodeMirror 6 (CDN-loaded) with a native `<textarea>` for reliability and offline use
- `snapshot_download` calls now use `resume_download=True` to recover interrupted downloads

### Known limitations (alpha)
- macOS / Apple Silicon only (mlx-whisper requirement)
- No speaker diarisation
- No export formats beyond Markdown
- No auto-update mechanism
- Tests cover server/API layer only; no end-to-end or UI tests yet

---

## Legend

| Label | Meaning |
|---|---|
| `alpha` | Proof of concept — functional but rough, breaking changes expected |
| `beta` | Feature-complete enough for broader testing, API stabilising |
| `1.0.0` | First stable release |
