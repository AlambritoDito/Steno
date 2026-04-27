"""Tests for steno_server.pipeline.

Includes the strict "silence → no text" contract: after the full chain
(audio_io → vad → transcriber → delooping → BoH) runs over silence, the
output must be either empty or contain only whitespace-only segments.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from steno_server import jobs as jobs_mod
from steno_server import pipeline, storage
from steno_server.config import settings
from steno_server.jobs import JobStatus
from steno_server.postprocess import TranscriptSegment
from steno_server.transcriber import Transcriber, TranscriptChunk

from .conftest import needs_ffmpeg


@pytest.fixture(autouse=True)
async def isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")
    await jobs_mod.init_db()


def _make_fake_transcriber(segments: list[TranscriptSegment]) -> Transcriber:
    """Return a Transcriber whose streaming methods emit canned segments."""
    t = Transcriber(language="es")

    async def fake_streaming(audio_path, language, on_chunk):
        for seg in segments:
            await on_chunk(
                TranscriptChunk(text=seg.text, start_s=seg.start_s, end_s=seg.end_s)
            )
        return segments

    async def fake_segment(audio, sample_rate, prompt=None):
        return list(segments)

    t.transcribe_streaming = fake_streaming  # type: ignore[assignment]
    t.transcribe_segment = fake_segment  # type: ignore[assignment]
    t._loaded = True
    return t


async def _run_phase1(
    job_id: str, source: Path, language: str, t: Transcriber, events: list[dict[str, Any]]
):
    async def cb(payload):
        events.append(payload)

    return await pipeline.run_phase_1(
        job_id=job_id,
        source_path=source,
        language=language,
        transcriber=t,
        progress_callback=cb,
    )


class TestRunPhase1WithFakeTranscriber:
    async def test_writes_raw_md_and_emits_events(
        self, silence_wav: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Use silence as input but fake out transcription with canned content
        # so we only exercise the orchestration logic. (VAD on silence yields
        # no speech, so we also disable VAD to ensure a non-empty transcript
        # path runs.)
        monkeypatch.setattr(settings, "enable_vad", False)
        job = await jobs_mod.create_job(filename="x.wav", language="es")
        canned = [
            TranscriptSegment("hola mundo", 0.0, 1.5),
            TranscriptSegment("adiós", 1.5, 2.5),
        ]
        events: list[dict[str, Any]] = []
        result = await _run_phase1(
            job.id, silence_wav, "es", _make_fake_transcriber(canned), events
        )

        assert result.transcript_path.exists()
        content = result.transcript_path.read_text(encoding="utf-8")
        assert "hola mundo" in content
        assert "adiós" in content
        assert "type: raw" in content

        loaded = await jobs_mod.get_job(job.id)
        assert loaded.status is JobStatus.PHASE1_DONE

        kinds = [e["type"] for e in events]
        assert kinds[0] == "phase1_started"
        assert kinds[-1] == "phase1_completed"
        assert kinds.count("phase1_chunk") == 2

    async def test_failure_marks_job_failed_and_emits_error(
        self, silence_wav: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings, "enable_vad", False)
        job = await jobs_mod.create_job(filename="x.wav", language="es")

        boom_t = Transcriber(language="es")

        async def boom(audio_path, language, on_chunk):
            raise RuntimeError("boom!")

        boom_t.transcribe_streaming = boom  # type: ignore[assignment]
        boom_t._loaded = True

        events: list[dict[str, Any]] = []
        with pytest.raises(RuntimeError):
            await _run_phase1(job.id, silence_wav, "es", boom_t, events)

        loaded = await jobs_mod.get_job(job.id)
        assert loaded.status is JobStatus.FAILED
        assert "boom" in (loaded.error_message or "")
        assert any(e["type"] == "error" for e in events)


class TestSilenceFullPipelineNoText:
    """Strict contract: the full chain must produce no user-visible text on silence."""

    @needs_ffmpeg
    async def test_silence_full_pipeline_no_text(
        self, silence_wav: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("STENO_SERVER_TEST_MODEL", "mlx-community/whisper-tiny")
        # Real Whisper, real VAD, real postprocess — no mocks.
        job = await jobs_mod.create_job(filename="silent.wav", language="es")
        events: list[dict[str, Any]] = []
        transcriber = Transcriber(language="es")
        result = await _run_phase1(job.id, silence_wav, "es", transcriber, events)

        # The contract: 0 segments OR every segment has empty text.
        non_empty = [s for s in result.segments if s.text.strip()]
        assert non_empty == [], (
            f"Silence produced visible text after postprocess: {non_empty}"
        )
