"""MLX-Whisper transcription wrapper for Steno."""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import numpy as np

from steno.config import Config


class Transcriber:
    """Wraps mlx-whisper for streaming transcription."""

    def __init__(
        self,
        model_name: str = Config.MODEL_NAME,
        language: str | None = None,
        status_callback: Callable[..., Coroutine[Any, Any, None]] | None = None,
    ):
        self._model_name = model_name
        self._language = language
        self._model = None
        self._loaded = False
        self._status_callback = status_callback

    def set_model(self, model_name: str) -> None:
        """Change the model (resets loaded state)."""
        if model_name != self._model_name:
            self._model_name = model_name
            self._loaded = False
            self._model = None

    async def transcribe(self, audio_chunk: np.ndarray) -> str:
        """Transcribe an audio chunk, returning the text string.

        Returns '' for silence. Uses asyncio.to_thread() to avoid
        blocking the event loop.
        """
        # Detect silence via RMS
        rms = float(np.sqrt(np.mean(audio_chunk**2)))
        if rms < Config.SILENCE_THRESHOLD:
            return ""

        # Lazy-load model on first call
        if not self._loaded:
            if self._status_callback:
                await self._status_callback("model_loading")
            await asyncio.to_thread(self._load_model)
            self._loaded = True
            if self._status_callback:
                await self._status_callback("model_ready")

        result = await asyncio.to_thread(self._run_transcription, audio_chunk)
        return result

    def _load_model(self) -> None:
        """Load the mlx-whisper model (runs in a thread)."""
        import mlx_whisper
        # Trigger model download/load by running a tiny silent transcription
        mlx_whisper.transcribe(
            np.zeros(Config.SAMPLE_RATE, dtype=np.float32),
            path_or_hf_repo=self._model_name,
            language=self._language,
        )

    def _run_transcription(self, audio_chunk: np.ndarray) -> str:
        """Run transcription in a thread."""
        import mlx_whisper
        result = mlx_whisper.transcribe(
            audio_chunk,
            path_or_hf_repo=self._model_name,
            language=self._language,
        )
        text = result.get("text", "").strip()
        return text

    def download_model(self) -> None:
        """Download/cache the model without transcribing (runs in a thread)."""
        self._load_model()
        self._loaded = True

    @staticmethod
    def download_model_by_repo(repo: str) -> None:
        """Download/cache any model by repo name (runs in thread)."""
        import mlx_whisper
        mlx_whisper.transcribe(
            np.zeros(Config.SAMPLE_RATE, dtype=np.float32),
            path_or_hf_repo=repo,
        )

    def transcribe_file(self, file_path: str) -> list[dict]:
        """Transcribe an entire audio file. Returns list of segments.

        Each segment: {"start": float, "end": float, "text": str}
        Runs in a thread — call via asyncio.to_thread().
        """
        import mlx_whisper

        if not self._loaded:
            self._load_model()
            self._loaded = True

        result = mlx_whisper.transcribe(
            file_path,
            path_or_hf_repo=self._model_name,
            language=self._language,
        )

        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", "").strip(),
            })

        # Fallback: if no segments but there's top-level text
        if not segments and result.get("text", "").strip():
            segments.append({
                "start": 0,
                "end": 0,
                "text": result["text"].strip(),
            })

        return segments

    def is_loaded(self) -> bool:
        """Return whether the model has been loaded."""
        return self._loaded

    def get_model_info(self) -> dict:
        """Return model information."""
        return {
            "model_name": self._model_name,
            "loaded": self._loaded,
        }
