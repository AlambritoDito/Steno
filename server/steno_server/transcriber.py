"""mlx-whisper wrapper with the anti-hallucination parameter set.

The desktop app's ``app/steno/transcriber.py`` is the design reference: same
explicit ``unload_model()`` + ``mx.metal.clear_cache()`` pattern when the
language or model changes. We re-implement instead of importing because the
two products are independent workspace members.

Two transcription paths:

1. ``transcribe_segment(audio, sample_rate, prompt=None)`` — one-shot inference
   on an in-memory array. Used by the streaming path where the pipeline has
   already chunked audio.

2. ``transcribe_streaming(audio_path, language, on_chunk)`` — convenience wrapper
   that runs Whisper over a whole file and invokes ``on_chunk`` once per Whisper
   segment so the API can emit WS events as text becomes available.

mlx-whisper does NOT expose ``no_repeat_ngram_size`` in its public Python API
as of 0.4.x; tracked as v1.1 GH issue. Looping is mitigated by the
``postprocess.delooping`` pass.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from .config import settings
from .logging_setup import get_logger
from .postprocess import TranscriptSegment

logger = get_logger(__name__)


@dataclass(frozen=True)
class TranscriptChunk:
    """A single segment as emitted by Whisper (passed to ``on_chunk``)."""

    text: str
    start_s: float
    end_s: float
    is_partial: bool = False


def _resolve_model_name() -> str:
    """Pick the model to load.

    ``STENO_SERVER_TEST_MODEL`` overrides the configured default — used by the
    test suite to swap in ``whisper-tiny`` (~75 MB) instead of pulling
    ``whisper-large-v3-turbo`` (~3 GB) on every CI run.
    """
    test_model = os.environ.get("STENO_SERVER_TEST_MODEL")
    if test_model:
        return test_model
    return settings.model_name


# Whisper params shared by every call. Centralized so tests can assert what
# the production transcriber actually passes through.
def whisper_call_kwargs(language: str, *, word_timestamps: bool = True) -> dict[str, Any]:
    """Anti-hallucination parameter dict for ``mlx_whisper.transcribe``.

    Pure function so tests can assert each value without instantiating the
    Transcriber (which loads the model on first use).
    """
    return {
        "language": language,
        "task": "transcribe",
        "temperature": settings.temperature,
        "condition_on_previous_text": settings.condition_on_previous_text,
        "compression_ratio_threshold": settings.compression_ratio_threshold,
        "logprob_threshold": settings.logprob_threshold,
        "no_speech_threshold": settings.no_speech_threshold,
        "hallucination_silence_threshold": settings.hallucination_silence_threshold,
        "word_timestamps": word_timestamps,
        "fp16": True,
    }


class Transcriber:
    """Stateful wrapper that lazily loads the mlx-whisper model.

    A single instance is intended to be shared across the FastAPI worker
    process (mlx-whisper is not thread-safe — the queue worker holds the
    only reference and dispatches one call at a time).
    """

    def __init__(self, model_name: str | None = None, language: str | None = None) -> None:
        self.model_name = model_name or _resolve_model_name()
        self.language = language or settings.default_language
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _load(self) -> None:
        # The model is "loaded" the first time mlx_whisper.transcribe is called
        # in our process; mlx-whisper has no separate explicit-load API. The
        # flag exists so we can record the transition for /api/health.
        if self._loaded:
            return
        # Importing here keeps test environments without MLX-capable hardware
        # from importing this module accidentally.
        import mlx_whisper  # noqa: F401

        self._loaded = True
        logger.info("transcriber_ready", model=self.model_name, language=self.language)

    def unload(self) -> None:
        """Release the model from VRAM (call before switching models)."""
        if not self._loaded:
            return
        try:
            import mlx.core as mx

            mx.metal.clear_cache()
        except Exception as exc:  # noqa: BLE001
            logger.warning("metal_clear_cache_failed", error=str(exc))
        self._loaded = False

    def set_language(self, language: str) -> None:
        if language not in settings.supported_languages:
            raise ValueError(
                f"Unsupported language: {language!r}. "
                f"Supported: {settings.supported_languages}"
            )
        self.language = language

    async def transcribe_segment(
        self,
        audio,  # np.ndarray, deferred typing to keep numpy out of import-time hot path
        sample_rate: int,
        prompt: str | None = None,
    ) -> list[TranscriptSegment]:
        """Run mlx-whisper on a single in-memory audio array.

        Runs in the default executor so the asyncio loop isn't blocked.
        """
        self._load()
        kwargs = whisper_call_kwargs(self.language)
        if prompt:
            kwargs["initial_prompt"] = prompt

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _do_transcribe(audio, kwargs, model_path=self.model_name),
        )
        return _segments_from_result(result)

    async def transcribe_streaming(
        self,
        audio_path: Path,
        language: str | None,
        on_chunk: Callable[[TranscriptChunk], Awaitable[None]],
    ) -> list[TranscriptSegment]:
        """Run mlx-whisper on a whole file, emitting each segment via on_chunk."""
        if language:
            self.set_language(language)
        self._load()
        kwargs = whisper_call_kwargs(self.language)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _do_transcribe(str(audio_path), kwargs, model_path=self.model_name),
        )
        segments = _segments_from_result(result)
        for seg in segments:
            await on_chunk(
                TranscriptChunk(text=seg.text, start_s=seg.start_s, end_s=seg.end_s)
            )
        return segments


# ---------------------------------------------------------------------------
# mlx-whisper bridge
# ---------------------------------------------------------------------------


def _do_transcribe(audio_or_path, kwargs: dict[str, Any], *, model_path: str) -> dict[str, Any]:
    """Synchronous shim around mlx_whisper.transcribe.

    Kept module-level so tests can monkey-patch it without instantiating the
    Transcriber class.
    """
    import mlx_whisper

    return mlx_whisper.transcribe(audio_or_path, path_or_hf_repo=model_path, **kwargs)


def _segments_from_result(result: dict[str, Any]) -> list[TranscriptSegment]:
    """Turn mlx-whisper's raw output into our typed segments."""
    raw_segments = result.get("segments") or []
    out: list[TranscriptSegment] = []
    for raw in raw_segments:
        text = (raw.get("text") or "").strip()
        if not text:
            continue
        out.append(
            TranscriptSegment(
                text=text,
                start_s=float(raw.get("start", 0.0)),
                end_s=float(raw.get("end", 0.0)),
            )
        )
    return out
