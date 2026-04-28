"""Audio file ingestion: validation, normalization, chunking.

Whisper expects 16 kHz mono float32 PCM. This module turns whatever the user
uploaded (mp3, m4a, flac, ogg, webm, wav) into that representation, validates
size/duration up front, and exposes a chunk iterator the pipeline uses to
stream segments to the model.

ffmpeg is required at runtime for non-WAV/FLAC/OGG inputs because librosa
delegates to it via the soundfile backend's audioread fallback. The deploy
README covers ``brew install ffmpeg``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .config import settings

SAMPLE_RATE = 16_000
MIN_DURATION_S = 0.5  # shorter than this is almost certainly a misclick
MAX_DURATION_S = 4 * 60 * 60  # 4 hours, generous for long meetings

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}


@dataclass
class AudioInfo:
    """Description of a normalized audio file."""

    path: Path
    duration_s: float
    sample_rate: int
    num_samples: int


@dataclass
class AudioChunk:
    """A contiguous slice of normalized audio, with its position in the source."""

    samples: np.ndarray  # float32, mono
    start_s: float
    end_s: float
    sample_rate: int


class AudioValidationError(Exception):
    """Raised when an upload fails the pre-pipeline validation step."""


@dataclass
class ValidationResult:
    """Outcome of validate_upload."""

    ok: bool
    duration_s: float | None = None
    size_mb: float | None = None
    extension: str | None = None
    error: str | None = None


def validate_upload(file_path: Path) -> ValidationResult:
    """Validate an uploaded audio file.

    Checks (in order): existence, extension, file size, decodability, duration.
    Returns a structured result rather than raising so callers can surface a
    clean error message in the API response.
    """
    if not file_path.exists():
        return ValidationResult(ok=False, error=f"File not found: {file_path}")

    extension = file_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        return ValidationResult(
            ok=False,
            extension=extension,
            error=(
                f"Unsupported audio format: {extension}. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )

    size_bytes = file_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        return ValidationResult(
            ok=False,
            extension=extension,
            size_mb=round(size_mb, 2),
            error=(
                f"File too large: {size_mb:.1f} MB exceeds limit "
                f"{settings.max_upload_size_mb} MB"
            ),
        )

    try:
        duration = get_duration(file_path)
    except Exception as exc:  # noqa: BLE001 — librosa raises a wide variety
        return ValidationResult(
            ok=False,
            extension=extension,
            size_mb=round(size_mb, 2),
            error=f"Could not decode audio: {exc}",
        )

    if duration < MIN_DURATION_S:
        return ValidationResult(
            ok=False,
            extension=extension,
            size_mb=round(size_mb, 2),
            duration_s=duration,
            error=f"Audio too short: {duration:.2f}s (minimum {MIN_DURATION_S}s)",
        )
    if duration > MAX_DURATION_S:
        return ValidationResult(
            ok=False,
            extension=extension,
            size_mb=round(size_mb, 2),
            duration_s=duration,
            error=f"Audio too long: {duration:.0f}s (maximum {MAX_DURATION_S}s)",
        )

    return ValidationResult(
        ok=True,
        duration_s=duration,
        size_mb=round(size_mb, 2),
        extension=extension,
    )


def get_duration(file_path: Path) -> float:
    """Return audio duration in seconds without reading the full file into memory."""
    try:
        # soundfile handles WAV / FLAC / OGG natively, no ffmpeg required.
        info = sf.info(str(file_path))
        return info.frames / info.samplerate
    except Exception:
        # Fall back to librosa (uses audioread / ffmpeg for compressed formats)
        return float(librosa.get_duration(path=str(file_path)))


def normalize_audio(input_path: Path, output_path: Path) -> AudioInfo:
    """Decode any supported format to 16 kHz mono float32 WAV.

    Raises ``AudioValidationError`` if decoding produces an empty array
    (corrupt file that didn't trip the duration check).
    """
    samples, _ = librosa.load(str(input_path), sr=SAMPLE_RATE, mono=True)
    if samples.size == 0:
        raise AudioValidationError(f"Decoded zero samples from {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), samples, SAMPLE_RATE, subtype="PCM_16")

    return AudioInfo(
        path=output_path,
        duration_s=len(samples) / SAMPLE_RATE,
        sample_rate=SAMPLE_RATE,
        num_samples=len(samples),
    )


def load_normalized(file_path: Path) -> tuple[np.ndarray, int]:
    """Load a normalized WAV into a float32 mono numpy array."""
    samples, sample_rate = sf.read(str(file_path), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, int(sample_rate)


def chunk_audio(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    chunk_seconds: int = 30,
    overlap_seconds: float = 0.0,
) -> Iterator[AudioChunk]:
    """Yield fixed-size chunks of audio with optional overlap.

    Whisper handles 30 s windows internally; we keep that as the default
    chunk size for streaming-style processing. The optional overlap helps
    avoid clipping words at chunk boundaries.
    """
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be in [0, chunk_seconds)")

    chunk_size = chunk_seconds * sample_rate
    step = int((chunk_seconds - overlap_seconds) * sample_rate)
    if step <= 0:
        step = chunk_size

    pos = 0
    total = len(samples)
    while pos < total:
        end = min(pos + chunk_size, total)
        yield AudioChunk(
            samples=samples[pos:end],
            start_s=pos / sample_rate,
            end_s=end / sample_rate,
            sample_rate=sample_rate,
        )
        if end >= total:
            break
        pos += step
