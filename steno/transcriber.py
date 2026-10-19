"""Serialized, offline inference facade for MLX and CTranslate2."""

import asyncio
import logging
import threading
import weakref
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import numpy as np

from steno.backends import create_backend
from steno.backends.base import BackendBusyError, ModelNotDownloadedError, iter_audio_chunks, validate_model_files
from steno.config import Config, default_backend, resolve_model_repo

logger = logging.getLogger("steno.transcriber")
_INFERENCE_LOCK = threading.RLock()
_DOWNLOAD_LOCK = threading.Lock()
_active_transcriber: weakref.ReferenceType | None = None


class DownloadCancelled(RuntimeError):
    """A model download was cancelled; partial cache files may be resumed later."""


class Transcriber:
    """Keep a single native model resident and serialize every inference call."""

    def __init__(
        self,
        model_name: str = Config.MODEL_NAME,
        language: str | None = None,
        status_callback: Callable[..., Coroutine[Any, Any, None]] | None = None,
        *,
        device: str = "auto",
        backend: str | None = None,
    ):
        self._backend_name = backend or default_backend()
        self._model_name = resolve_model_repo(model_name, self._backend_name)
        self._language = language
        self._model = None  # Retained for compatibility with diagnostic consumers.
        self._loaded = False
        self._status_callback = status_callback
        self._backend = create_backend(self._backend_name, device=device)

    def _unload_locked(self) -> None:
        self._backend.unload()
        self._model = None
        self._loaded = False

    def unload_model(self) -> None:
        with _INFERENCE_LOCK:
            self._unload_locked()

    def unload(self) -> None:
        self.unload_model()

    def set_model(self, model_name: str) -> None:
        model_name = resolve_model_repo(model_name, self._backend_name)
        if not _INFERENCE_LOCK.acquire(blocking=False):
            raise BackendBusyError("Cannot change model while transcription is running")
        try:
            if model_name != self._model_name:
                self._unload_locked()
                self._model_name = model_name
        finally:
            _INFERENCE_LOCK.release()

    def set_language(self, language: str | None) -> None:
        if language is not None and (not isinstance(language, str) or not language.isalpha() or len(language) > 3):
            raise ValueError("Invalid transcription language")
        if not _INFERENCE_LOCK.acquire(blocking=False):
            raise BackendBusyError("Cannot change language while transcription is running")
        try:
            self._language = language
        finally:
            _INFERENCE_LOCK.release()

    def _load_model(self) -> None:
        global _active_transcriber
        with _INFERENCE_LOCK:
            active = _active_transcriber() if _active_transcriber else None
            if active is not None and active is not self:
                active._unload_locked()
            if not self._loaded:
                path = Config.model_snapshot_path(self._model_name)
                if path is None:
                    raise ModelNotDownloadedError(f"Download or repair model {self._model_name} before transcribing")
                self._backend.load(path)
                self._loaded = True
            _active_transcriber = weakref.ref(self)

    def warmup(self) -> None:
        """Load and run the native backend probe, even when the sample is silent."""
        self._load_model()

    @staticmethod
    def _validate_audio(audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Audio must be finite mono samples at 16000 Hz")
        return np.ascontiguousarray(audio)

    async def transcribe(self, audio_chunk: np.ndarray) -> str:
        audio = self._validate_audio(audio_chunk)
        if not audio.size or float(np.sqrt(np.mean(audio ** 2))) < Config.SILENCE_THRESHOLD:
            return ""
        loading = not self._loaded
        if loading and self._status_callback:
            await self._status_callback("model_loading")
        result = await asyncio.to_thread(self._run_transcription, audio)
        if loading and self._status_callback:
            await self._status_callback("model_ready")
        return result

    def _run_transcription(self, audio_chunk: np.ndarray) -> str:
        return " ".join(s["text"] for s in self.transcribe_segments(audio_chunk))

    def transcribe_segments(self, audio: np.ndarray | str | Path, *, language: str | None = None) -> list[dict]:
        """Synchronous job API; returns offsets in seconds relative to this input.

        Run in a worker thread. Files are decoded using bundled PyAV. Calls from
        multiple jobs are serialized, including model load and generator iteration.
        """
        if isinstance(audio, (str, Path)):
            return self.transcribe_file(str(audio))
        samples = self._validate_audio(audio)
        if not samples.size or float(np.sqrt(np.mean(samples ** 2))) < Config.SILENCE_THRESHOLD:
            return []
        with _INFERENCE_LOCK:
            self._load_model()
            return self._backend.transcribe(samples, language if language is not None else self._language)

    def transcribe_file(self, file_path: str) -> list[dict]:
        return [segment for _, batch in self.iter_transcribe_file(file_path) for segment in batch]

    def iter_transcribe_file(self, file_path: str, *, start_frame: int = 0):
        """Yield (processed seconds, segments) so jobs can cancel between windows.

        Each decoding/inference window is bounded to 30 seconds. Offset adjustment
        keeps timestamps monotonic for files longer than one hour, including silence.
        The caller must close the iterator on cancellation to release the file.
        A durable cursor skips inference for windows already committed by this job.
        """
        if not isinstance(start_frame, int) or start_frame < 0:
            raise ValueError("Invalid resume frame")
        chunks = iter_audio_chunks(file_path, Config.SAMPLE_RATE)
        try:
            for offset, samples in chunks:
                if offset + len(samples) <= start_frame:
                    continue
                if offset < start_frame:
                    samples = samples[start_frame - offset:]
                    offset = start_frame
                batch = self.transcribe_segments(samples)
                seconds = offset / Config.SAMPLE_RATE
                for segment in batch:
                    segment["start"] += seconds
                    segment["end"] += seconds
                yield (offset + len(samples)) / Config.SAMPLE_RATE, batch
        finally:
            chunks.close()

    def download_model(self, progress_state: dict | None = None) -> None:
        _download_with_progress(self._model_name, progress_state)

    @staticmethod
    def download_model_by_repo(repo: str, progress_state: dict | None = None) -> None:
        _download_with_progress(resolve_model_repo(repo), progress_state)

    def is_loaded(self) -> bool:
        return self._loaded

    def get_capabilities(self) -> dict:
        return self._backend.capabilities()

    def get_model_info(self) -> dict:
        return {"model_name": self._model_name, "loaded": self._loaded, **self.get_capabilities()}


def _download_with_progress(repo: str, progress_state: dict | None = None) -> None:
    """Download with the full tqdm protocol, byte progress and cancellation."""
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm

    Config._cache_folder(repo)  # Reject paths before making a network request.
    state = progress_state if progress_state is not None else {}
    if not _DOWNLOAD_LOCK.acquire(blocking=False):
        raise BackendBusyError("Another model download is already running")

    def check_cancel():
        if state.get("cancel_requested"):
            raise DownloadCancelled("Model download cancelled")

    class ProgressTracker(tqdm):
        def __init__(self, *args, **kwargs):
            self._tracks_bytes = kwargs.get("unit") == "B"
            self._completed_bytes = kwargs.get("initial", 0)
            # HuggingFace passes a progress-bar grouping name; vanilla tqdm does not.
            kwargs.pop("name", None)
            super().__init__(*args, **kwargs)
            self._publish()

        def _publish(self):
            if self._tracks_bytes:
                state["bytes_downloaded"] = self._completed_bytes
                state["bytes_total"] = int(getattr(self, "total", 0) or 0)

        def update(self, n=1):
            check_cancel()
            self._completed_bytes += n
            result = super().update(n)
            self._publish()
            return result

        def refresh(self, *args, **kwargs):
            check_cancel()
            self._publish()
            return super().refresh(*args, **kwargs)

    try:
        state.update({"bytes_downloaded": 0, "bytes_total": 0, "status": "downloading"})
        logger.info("event=model_download_started model=%s", repo)
        check_cancel()
        backend = "mlx" if repo.startswith("mlx-community/") else "faster-whisper"
        patterns = (
            ["config.json", "*.safetensors", "*.npz"] if backend == "mlx" else
            ["config.json", "model.bin", "tokenizer.json", "preprocessor_config.json", "vocabulary.*"]
        )
        path = snapshot_download(
            repo, cache_dir=str(Config.models_dir()), tqdm_class=ProgressTracker,
            allow_patterns=patterns, token=False,
        )
        check_cancel()
        validate_model_files(Path(path), backend)
        state["status"] = "complete"
        logger.info("event=model_download_completed model=%s bytes=%d", repo, state.get("bytes_downloaded", 0))
    except DownloadCancelled:
        state["status"] = "cancelled"
        logger.info("event=model_download_cancelled model=%s", repo)
        raise
    except Exception as exc:
        state.update({"status": "error", "error": str(exc)})
        logger.error("event=model_download_failed model=%s error_type=%s", repo, type(exc).__name__)
        raise
    finally:
        _DOWNLOAD_LOCK.release()
