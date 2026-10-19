"""Shared errors, model validation, and native-free audio decoding interfaces."""

import json
from pathlib import Path

import numpy as np


class BackendUnavailableError(RuntimeError):
    """The selected inference runtime or device cannot be initialized."""


class BackendInferenceError(RuntimeError):
    """A job failed without changing backend or losing its source audio."""

    def __init__(self, message: str, *, retry_cpu: bool = False):
        super().__init__(message)
        self.retry_cpu = retry_cpu


class ModelNotDownloadedError(RuntimeError):
    """A complete local model is required before starting inference."""


class BackendBusyError(RuntimeError):
    """A model change was requested while inference owns the runtime."""


def validate_model_files(path: Path, backend: str) -> None:
    """Reject missing, partial, invalid-JSON and Git LFS pointer model files.

    Native model loading plus a warm-up inference validates the weight format.
    This lightweight check is safe to use while listing available models.
    """
    required = ["config.json"]
    if backend == "mlx":
        weights = next((name for name in ("weights.safetensors", "weights.npz")
                        if (path / name).is_file()), None)
        if weights is None:
            raise ValueError("MLX model weights are missing")
        required.append(weights)
    elif backend == "faster-whisper":
        required.extend(["model.bin", "tokenizer.json", "preprocessor_config.json"])
    else:
        raise ValueError("Unsupported model format")
    for name in required:
        file = path / name
        if not file.is_file() or file.stat().st_size == 0:
            raise ValueError(f"Incomplete model: {name}")
        with file.open("rb") as stream:
            if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
                raise ValueError(f"Unresolved model download: {name}")
        if name.endswith(".json"):
            value = json.loads(file.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"Invalid model configuration: {name}")


def iter_audio_chunks(path: str, sample_rate: int = 16000, chunk_seconds: int = 30):
    """Yield (sample offset, mono PCM) with a bounded decoded-audio buffer."""
    import av
    if sample_rate <= 0 or chunk_seconds <= 0:
        raise ValueError("Sample rate and chunk duration must be positive")
    chunk_size = sample_rate * chunk_seconds
    buffer = np.empty(chunk_size, dtype=np.float32)
    filled = offset = 0
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
    with av.open(path) as container:
        if not container.streams.audio:
            raise ValueError("The file contains no audio stream")
        def frames():
            for frame in container.decode(audio=0):
                yield from resampler.resample(frame)
            yield from resampler.resample(None)
        for converted in frames():
            samples = converted.to_ndarray().reshape(-1)
            position = 0
            while position < len(samples):
                count = min(chunk_size - filled, len(samples) - position)
                buffer[filled:filled + count] = samples[position:position + count]
                position += count
                filled += count
                if filled == chunk_size:
                    yield offset, buffer
                    offset += filled
                    filled = 0
                    buffer = np.empty(chunk_size, dtype=np.float32)
        if filled:
            yield offset, buffer[:filled].copy()


def decode_audio(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Convenience decoder for small files; jobs consume iter_audio_chunks instead."""
    parts = [samples for _, samples in iter_audio_chunks(path, sample_rate)]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
