"""Transcription pipeline orchestrator.

Phase 1 is implemented here; Phase 2 lands later. The orchestrator pulls
together audio_io, vad, transcriber, and postprocess into a single async
flow, emitting structured progress events through the caller-supplied
callback so the WebSocket layer can broadcast them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from . import __version__
from .audio_io import (
    AudioInfo,
    AudioValidationError,
    load_normalized,
    normalize_audio,
)
from .config import settings
from .jobs import JobStatus, update_status
from .logging_setup import get_logger
from .markdown_export import RawExportInputs, render_raw_md
from .postprocess import (
    TranscriptSegment,
    bag_of_hallucinations_filter,
    delooping,
    remap_timestamps,
)
from .storage import (
    get_normalized_path,
    get_transcript_path,
)
from .transcriber import Transcriber, TranscriptChunk
from .vad import VADProcessor

logger = get_logger(__name__)

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class RawTranscript:
    """Output of run_phase_1 — segments + the path the caller can serve."""

    segments: list[TranscriptSegment]
    transcript_path: Path
    audio_duration_seconds: float
    phase1_duration_seconds: float


async def _emit(callback: ProgressCallback | None, payload: dict[str, Any]) -> None:
    if callback:
        await callback(payload)


async def run_phase_1(
    *,
    job_id: str,
    source_path: Path,
    language: str,
    transcriber: Transcriber,
    vad: VADProcessor | None = None,
    progress_callback: ProgressCallback | None = None,
) -> RawTranscript:
    """Phase 1: normalize → VAD → transcribe → postprocess → write raw md.

    Updates the job row's status as the phase progresses (phase1_running on
    entry, phase1_done on success, failed on exception). Re-raises whatever
    exception was caught so the queue worker can decide what to do.
    """
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()

    await update_status(job_id, JobStatus.PHASE1_RUNNING, phase1_started_at=started_at)
    await _emit(progress_callback, {"type": "phase1_started", "job_id": job_id})

    try:
        # 1. Normalize audio to 16kHz mono WAV.
        normalized_path = get_normalized_path(job_id)
        info = normalize_audio(source_path, normalized_path)
        logger.info(
            "phase1_normalized",
            job_id=job_id,
            duration_s=round(info.duration_s, 2),
        )
        await update_status(job_id, JobStatus.PHASE1_RUNNING, audio_duration_seconds=info.duration_s)

        # 2. Run VAD if enabled, else use the full audio.
        samples, sr = load_normalized(normalized_path)
        timestamp_map = None
        if settings.enable_vad:
            vad_proc = vad or VADProcessor()
            speech_only, timestamp_map = vad_proc.extract_speech_only(samples, sr)
            if speech_only.size == 0:
                # No speech at all — short-circuit to avoid feeding empty audio
                # to Whisper.
                logger.info("phase1_no_speech_detected", job_id=job_id)
                segments_clean: list[TranscriptSegment] = []
            else:
                segments_clean = await _transcribe_and_clean(
                    transcriber=transcriber,
                    audio=speech_only,
                    sample_rate=sr,
                    audio_path=normalized_path,
                    language=language,
                    timestamp_map=timestamp_map,
                    progress_callback=progress_callback,
                    job_id=job_id,
                )
        else:
            segments_clean = await _transcribe_and_clean(
                transcriber=transcriber,
                audio=samples,
                sample_rate=sr,
                audio_path=normalized_path,
                language=language,
                timestamp_map=None,
                progress_callback=progress_callback,
                job_id=job_id,
            )

        # 3. Render and write transcript-raw.md
        phase1_duration = time.monotonic() - started_monotonic
        md_path = get_transcript_path(job_id, "raw")
        md = render_raw_md(
            RawExportInputs(
                job_id=job_id,
                source_filename=source_path.name,
                language=language,
                audio_duration_seconds=info.duration_s,
                phase1_duration_seconds=phase1_duration,
                model=transcriber.model_name,
                segments=segments_clean,
            )
        )
        md_path.write_text(md, encoding="utf-8")

        completed_at = datetime.now(timezone.utc)
        await update_status(
            job_id,
            JobStatus.PHASE1_DONE,
            phase1_completed_at=completed_at,
        )
        await _emit(
            progress_callback,
            {
                "type": "phase1_completed",
                "job_id": job_id,
                "transcript_url": f"/api/jobs/{job_id}/transcript-raw.md",
            },
        )
        logger.info(
            "phase1_completed",
            job_id=job_id,
            segments=len(segments_clean),
            duration_s=round(phase1_duration, 2),
        )

        return RawTranscript(
            segments=segments_clean,
            transcript_path=md_path,
            audio_duration_seconds=info.duration_s,
            phase1_duration_seconds=phase1_duration,
        )

    except (AudioValidationError, Exception) as exc:  # noqa: BLE001
        msg = f"Phase 1 failed: {exc}"
        logger.error("phase1_failed", job_id=job_id, error=str(exc), exc_info=True)
        await update_status(job_id, JobStatus.FAILED, error_message=msg)
        await _emit(
            progress_callback,
            {"type": "error", "job_id": job_id, "message": msg, "phase": "phase1"},
        )
        raise


async def _transcribe_and_clean(
    *,
    transcriber: Transcriber,
    audio,
    sample_rate: int,
    audio_path: Path,
    language: str,
    timestamp_map,
    progress_callback: ProgressCallback | None,
    job_id: str,
) -> list[TranscriptSegment]:
    """Run mlx-whisper and apply delooping + BoH; emit chunks as we go.

    For now we run a single ``transcribe_streaming`` over the file (simpler
    than chunk-by-chunk feeding). Each Whisper segment fires a phase1_chunk
    event.
    """
    raw_segments: list[TranscriptSegment] = []

    async def on_chunk(chunk: TranscriptChunk) -> None:
        raw_segments.append(
            TranscriptSegment(text=chunk.text, start_s=chunk.start_s, end_s=chunk.end_s)
        )
        await _emit(
            progress_callback,
            {
                "type": "phase1_chunk",
                "job_id": job_id,
                "text": chunk.text,
                "start_s": chunk.start_s,
                "end_s": chunk.end_s,
                "is_partial": chunk.is_partial,
            },
        )

    if timestamp_map is not None:
        # Speech-only path: feed the array directly.
        transcriber.set_language(language)
        # mlx-whisper doesn't truly stream, but we synthesize a streaming
        # loop so on_chunk fires as if it did.
        segs = await transcriber.transcribe_segment(audio, sample_rate)
        for seg in segs:
            await on_chunk(
                TranscriptChunk(text=seg.text, start_s=seg.start_s, end_s=seg.end_s)
            )
    else:
        await transcriber.transcribe_streaming(audio_path, language, on_chunk)

    # Remap timestamps from concatenated to original domain (no-op if no map).
    remapped = remap_timestamps(raw_segments, timestamp_map)
    deduped = delooping(remapped)
    return bag_of_hallucinations_filter(deduped, language)


__all__ = ["run_phase_1", "RawTranscript", "ProgressCallback"]
