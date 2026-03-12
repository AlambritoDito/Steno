"""MLX-Whisper transcription wrapper for Steno."""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

import numpy as np

from steno.config import Config

logger = logging.getLogger("steno.transcriber")


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

    def unload_model(self) -> None:
        """Explicitly release the current model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False

        import gc
        gc.collect()

        try:
            import mlx.core
            mlx.core.metal.clear_cache()
        except (ImportError, AttributeError):
            pass

        logger.info("Model unloaded and Metal cache cleared")

    def set_model(self, model_name: str) -> None:
        """Change the model (unloads previous model first)."""
        if model_name != self._model_name:
            self.unload_model()
            self._model_name = model_name

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
        logger.info("Loading model: %s (language=%s)", self._model_name, self._language)
        # Trigger model download/load by running a tiny silent transcription
        mlx_whisper.transcribe(
            np.zeros(Config.SAMPLE_RATE, dtype=np.float32),
            path_or_hf_repo=self._model_name,
            language=self._language,
        )
        logger.info("Model loaded successfully: %s", self._model_name)

    def _run_transcription(self, audio_chunk: np.ndarray) -> str:
        """Run transcription in a thread."""
        import mlx_whisper
        logger.debug("Transcribing chunk of %d samples...", len(audio_chunk))
        result = mlx_whisper.transcribe(
            audio_chunk,
            path_or_hf_repo=self._model_name,
            language=self._language,
        )
        text = result.get("text", "").strip()
        if text:
            logger.info("Transcription result: %s", text[:100])
        return text

    def download_model(self, progress_state: dict | None = None) -> None:
        """Download/cache the model without transcribing (runs in a thread).

        Uses huggingface_hub directly to avoid importing mlx native
        extensions, which fail inside PyInstaller bundles.

        If progress_state dict is provided, it will be updated with
        download progress: {bytes_downloaded, bytes_total, status}.
        """
        logger.info("Downloading model: %s", self._model_name)
        try:
            _download_with_progress(self._model_name, progress_state)
            logger.info("Model download complete: %s", self._model_name)
        except Exception as e:
            logger.error("Model download failed for %s: %s", self._model_name, e)
            raise

    @staticmethod
    def download_model_by_repo(repo: str, progress_state: dict | None = None) -> None:
        """Download/cache any model by repo name (runs in thread)."""
        logger.info("Downloading model by repo: %s", repo)
        try:
            _download_with_progress(repo, progress_state)
            logger.info("Model download complete: %s", repo)
        except Exception as e:
            logger.error("Model download failed for %s: %s", repo, e)
            raise

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


def _download_with_progress(repo: str, progress_state: dict | None = None) -> None:
    """Download a model from HuggingFace Hub with optional progress tracking."""
    from huggingface_hub import snapshot_download

    if progress_state is not None:
        progress_state.update({"bytes_downloaded": 0, "bytes_total": 0, "status": "downloading"})

        # Use tqdm_class to intercept progress
        class _ProgressTracker:
            """Adapts tqdm-style updates to a shared progress dict."""

            def __init__(self, *args, **kwargs):
                self.total = kwargs.get("total", 0)
                self.n = 0
                if self.total:
                    progress_state["bytes_total"] += self.total

            def update(self, n=1):
                self.n += n
                progress_state["bytes_downloaded"] += n

            def close(self):
                pass

            def set_postfix_str(self, *args, **kwargs):
                pass

            def refresh(self, *args, **kwargs):
                pass

            def clear(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        snapshot_download(repo, tqdm_class=_ProgressTracker)
        progress_state["status"] = "complete"
    else:
        snapshot_download(repo)
