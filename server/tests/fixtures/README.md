# Test fixtures

Three audio files used by `pytest` to exercise the audio I/O, VAD, transcriber, and pipeline modules.

| File | Purpose | Spec |
|---|---|---|
| `sample_silence.wav` | Anti-hallucination contract: silence in → no text out | 5 s, 16 kHz mono, all-zero samples |
| `sample_clean_es.wav` | Happy-path Spanish transcription | ~10 s, 16 kHz mono, clear speech |
| `sample_noisy_es.wav` | VAD + Whisper robustness | Same content as clean, with additive Gaussian noise |

## Regenerating from the script (default path)

The fixtures are NOT checked in. Generate them with:

```bash
uv run --directory server python tests/generate_fixtures.py
```

This uses macOS `say -v Paulina` for the Spanish samples and validates each one by round-tripping through `mlx-whisper` and checking that expected keywords appear in the transcription. If validation fails, follow the manual fallback below.

Optionally skip the validation if you're offline (or you trust the TTS):

```bash
uv run --directory server python tests/generate_fixtures.py --skip-validate
```

## Manual recording fallback

If `say -v Paulina` doesn't produce intelligible audio for `whisper-tiny` (validation fails), record the samples yourself:

### `sample_clean_es.wav`

Record yourself reading this script clearly in a quiet room. Aim for ~10 s.

> Reunión de planeación del proyecto. El viernes vamos a presentar los resultados al equipo. Tenemos tres puntos importantes que discutir.

Convert to 16 kHz mono PCM 16-bit WAV. With ffmpeg:

```bash
ffmpeg -i input.m4a -ac 1 -ar 16000 -sample_fmt s16 server/tests/fixtures/sample_clean_es.wav
```

Or if you have the file already in a high quality format, in QuickTime, export → Audio Only, then convert.

### `sample_noisy_es.wav`

Easiest: record the same script in a noisier environment (open window, fan running). Same conversion as above. Alternatively, regenerate it from the clean recording by re-running `generate_fixtures.py` after you've placed `sample_clean_es.wav` — the script will skip generating that one if it's already there… *(currently it always regenerates; track as a small follow-up if this becomes a recurring need)*.

### `sample_silence.wav`

Always regenerate: `generate_fixtures.py` produces it deterministically with `numpy.zeros`, so it's identical every time. No manual fallback needed.

## v1.1 follow-up

Better-quality offline TTS (piper-tts, coqui) is tracked as enhancement #34 — see `gh issue view 34` for the rationale.
