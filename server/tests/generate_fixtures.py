"""Generate the audio fixtures used by the steno-server test suite.

Run from anywhere with: ``uv run --directory server python tests/generate_fixtures.py``

Strategy:

1. ``sample_silence.wav`` — pure silence, 5 s, 16 kHz mono PCM16. Generated
   with numpy + soundfile.

2. ``sample_clean_es.wav`` — short Spanish script read by macOS ``say -v Paulina``
   directly to 16 kHz mono WAV. After generation, the script runs
   ``mlx_whisper.transcribe`` and asserts the output contains a known set
   of keywords. If the assertion fails, the script aborts and points at the
   manual-recording fallback (``server/tests/fixtures/README.md``).

3. ``sample_noisy_es.wav`` — sample_clean_es + additive Gaussian noise, with
   the same keyword validation (looser threshold, since noise hurts WER).

The ``mlx_whisper`` validation step uses ``whisper-tiny`` by default to keep
the script light; override with ``STENO_SERVER_TEST_MODEL`` if needed.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_RATE = 16_000

CLEAN_ES_SCRIPT = (
    "Reunión de planeación del proyecto. "
    "El viernes vamos a presentar los resultados al equipo. "
    "Tenemos tres puntos importantes que discutir."
)

# Loose: any of these substrings is enough to consider the round-trip valid.
# Tight nouns/verbs that survive Whisper transcription noise.
CLEAN_ES_KEYWORDS = ("reunion", "proyecto", "viernes", "presentar", "equipo")
NOISY_ES_KEYWORDS = ("reunion", "proyecto", "viernes")  # looser; noise eats words


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _ffmpeg_available() -> bool:
    """mlx-whisper shells out to ffmpeg for all audio decoding."""
    return subprocess.run(
        ["which", "ffmpeg"], capture_output=True, text=True, check=False
    ).returncode == 0


def _validate_fixture(fixture_path: Path, keywords: tuple[str, ...]) -> None:
    """Round-trip the fixture through mlx_whisper and assert keywords appear.

    Raises ``SystemExit`` with a clear error if validation fails. Skips with
    a printed warning if ffmpeg isn't installed (mlx-whisper requires it).
    """
    if not _ffmpeg_available():
        print(
            f"  ⚠ skipping validation of {fixture_path.name}: ffmpeg not installed.\n"
            f"    Install with `brew install ffmpeg`, or rerun with --skip-validate."
        )
        return

    import mlx_whisper

    model = os.environ.get("STENO_SERVER_TEST_MODEL", "mlx-community/whisper-tiny")
    result = mlx_whisper.transcribe(
        str(fixture_path),
        path_or_hf_repo=model,
        language="es",
        task="transcribe",
        temperature=0.0,
        condition_on_previous_text=False,
        fp16=True,
    )
    actual = (result.get("text") or "").strip()
    normalized_actual = _normalize(actual)
    matches = [kw for kw in keywords if kw in normalized_actual]
    if not matches:
        sys.stderr.write(
            "TTS quality insufficient for {fixture}.\n"
            "  Whisper transcribed: {actual!r}\n"
            "  Expected keywords  : {keywords}\n"
            "  Fall back to manual recording — see "
            "server/tests/fixtures/README.md\n".format(
                fixture=fixture_path.name,
                actual=actual,
                keywords=keywords,
            )
        )
        raise SystemExit(2)
    print(f"  validated {fixture_path.name}: matched {matches}")


def _make_silence() -> Path:
    out_path = FIXTURES_DIR / "sample_silence.wav"
    samples = np.zeros(SAMPLE_RATE * 5, dtype=np.float32)
    sf.write(str(out_path), samples, SAMPLE_RATE, subtype="PCM_16")
    print(f"  wrote {out_path.name} ({out_path.stat().st_size} bytes)")
    return out_path


def _make_clean_es(skip_validate: bool) -> Path:
    out_path = FIXTURES_DIR / "sample_clean_es.wav"
    cmd = [
        "say",
        "-v",
        "Paulina",
        "--file-format=WAVE",
        "--data-format=LEI16@16000",
        "-o",
        str(out_path),
        CLEAN_ES_SCRIPT,
    ]
    subprocess.run(cmd, check=True)
    print(f"  wrote {out_path.name} ({out_path.stat().st_size} bytes)")
    if not skip_validate:
        _validate_fixture(out_path, CLEAN_ES_KEYWORDS)
    return out_path


def _make_noisy_es(clean_path: Path, skip_validate: bool) -> Path:
    out_path = FIXTURES_DIR / "sample_noisy_es.wav"
    samples, sr = sf.read(str(clean_path), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    rng = np.random.default_rng(seed=42)
    # ~ -25 dB noise (rms ~0.056); enough to challenge VAD without obliterating the words.
    noise = rng.standard_normal(samples.shape).astype(np.float32) * 0.056
    noisy = np.clip(samples + noise, -1.0, 1.0)
    sf.write(str(out_path), noisy, sr, subtype="PCM_16")
    print(f"  wrote {out_path.name} ({out_path.stat().st_size} bytes)")
    if not skip_validate:
        _validate_fixture(out_path, NOISY_ES_KEYWORDS)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip the mlx-whisper round-trip check (useful in offline CI).",
    )
    args = parser.parse_args()

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating fixtures into {FIXTURES_DIR}")
    _make_silence()
    clean_path = _make_clean_es(args.skip_validate)
    _make_noisy_es(clean_path, args.skip_validate)
    print("All fixtures generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
