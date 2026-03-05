<p align="center">
  <img src="https://img.shields.io/badge/platform-macOS-000000?style=flat-square&logo=apple&logoColor=white" alt="macOS">
  <img src="https://img.shields.io/badge/Apple%20Silicon-native-000000?style=flat-square&logo=apple&logoColor=white" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License">
  <img src="https://img.shields.io/badge/privacy-100%25%20local-blue?style=flat-square" alt="100% Local">
</p>

<h1 align="center">Steno</h1>

<p align="center">
  <strong>Real-time local transcription for classes and meetings.</strong><br>
  Runs 100% on your Mac — no cloud, no data leaves your machine.<br>
  Powered by Whisper via MLX for fast, private, Apple Silicon-native transcription.
</p>

<p align="center">
  <em>Transcripción local en tiempo real para clases y juntas.</em><br>
  <em>Corre 100% en tu Mac — sin nube, ningún dato sale de tu máquina.</em>
</p>

---

## Features

- **Real-time transcription** — Live audio-to-text powered by [Whisper Large V3 Turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo) running natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx)
- **Markdown notes** — Take rich notes with a built-in CodeMirror 6 editor alongside the live transcript
- **Image support** — Drag & drop whiteboard photos, screenshots, or diagrams directly into your notes
- **Session management** — Save, browse, and export sessions as clean `.md` files
- **Multilingual UI** — English and Spanish with auto-detection from your browser, switchable at runtime
- **100% offline** — Everything runs locally. No accounts, no telemetry, no data ever leaves your machine
- **Single command** — Install and run with `uv` in seconds

## Requirements

| Requirement | Details |
|---|---|
| **Hardware** | Mac with Apple Silicon (M1 / M2 / M3 / M4) |
| **OS** | macOS 13 Ventura or later |
| **Python** | 3.11 or higher |
| **Package manager** | [uv](https://docs.astral.sh/uv/) |

## Quick Start

```bash
git clone https://github.com/AlambritoDito/Steno.git
cd steno
uv sync
uv run main.py
```

Steno opens automatically at **http://localhost:8080**.

## First Launch

On the first run, Steno downloads the Whisper Large V3 Turbo model (~1.5 GB). This happens **only once** — the model is cached locally for all future sessions.

## Usage

### Classes & Lectures

1. Launch Steno with `uv run main.py`
2. Enter a session name (e.g. *Physics — Thermodynamics*)
3. Select your microphone and click **Start Session**
4. Click **Start Recording** — the transcript streams in real time
5. Take notes in the right panel using Markdown
6. Drag & drop whiteboard photos or screenshots into the editor
7. Click **Export session (.md)** when done

### Meetings

1. Create a new session with the meeting name
2. Start recording when the meeting begins
3. Add your own notes alongside the live transcript
4. Stop recording and export the full session

## Tech Stack

| Layer | Technology |
|---|---|
| Transcription | [mlx-whisper](https://github.com/ml-explore/mlx-examples) (Whisper Large V3 Turbo) |
| Audio capture | [sounddevice](https://python-sounddevice.readthedocs.io/) + NumPy |
| Backend | [FastAPI](https://fastapi.tiangolo.com/) + WebSockets + [uvicorn](https://www.uvicorn.org/) |
| Frontend | Vanilla HTML/CSS/JS — single file, no build step |
| Editor | [CodeMirror 6](https://codemirror.net/) (CDN) |
| Markdown rendering | [marked.js](https://marked.js.org/) (CDN) |
| Package manager | [uv](https://docs.astral.sh/uv/) |

## Project Structure

```
steno/
├── main.py                 # Entry point: uv run main.py
├── pyproject.toml           # Project config & dependencies
├── steno/
│   ├── server.py            # FastAPI app, routes, WebSockets
│   ├── transcriber.py       # MLX-Whisper wrapper
│   ├── audio.py             # Microphone capture
│   ├── session.py           # Session management (save/load/export)
│   ├── config.py            # Global configuration
│   └── i18n.py              # Internationalization helpers
├── static/
│   └── index.html           # Complete UI (inline CSS + JS)
├── locales/
│   ├── en.json              # English strings
│   └── es.json              # Spanish strings
├── sessions/                # Saved session files (.md)
└── tests/                   # Test suite (pytest)
```

## Languages

The UI is available in **English** and **Spanish**. Language is auto-detected from your browser (`navigator.language`) and can be toggled at any time without reloading the page.

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Start the dev server
uv run main.py
```

## Privacy

Steno is **100% local and fully offline**.

- All audio processing happens on-device via Apple Silicon GPU
- Transcription runs through MLX — no API calls, no cloud services
- Session data is stored as plain Markdown files on your filesystem
- No accounts, no analytics, no telemetry
- No network requests except loading CodeMirror and marked.js from CDN on first page load

## Roadmap

- [ ] **Desktop app (Electron)** — One-click install for non-technical users. No terminal, no Python setup. Just download, open, and start transcribing. *(Coming soon)*
- [ ] **Virtual meeting capture** — Record and transcribe Zoom, Google Meet, and Microsoft Teams sessions by capturing system audio. Auto-capture presentation slides and screen shares as timestamped screenshots embedded in your notes
- [ ] **Speaker diarization** — Identify and label different speakers in the transcript
- [ ] **Search across sessions** — Full-text search over all past transcripts and notes
- [ ] **Additional languages** — More UI translations and transcription language support
- [ ] **PDF export** — Export sessions as formatted PDF documents

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
