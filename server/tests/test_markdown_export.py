"""Tests for steno_server.markdown_export."""

from __future__ import annotations

from steno_server.markdown_export import (
    CleanExportInputs,
    RawExportInputs,
    render_clean_md,
    render_raw_md,
)
from steno_server.postprocess import AnnotatedSegment, TranscriptSegment


def test_render_raw_md_has_required_frontmatter():
    md = render_raw_md(
        RawExportInputs(
            job_id="abc",
            source_filename="meeting.m4a",
            language="es",
            audio_duration_seconds=3621.5,
            phase1_duration_seconds=412.0,
            model="mlx-community/whisper-large-v3-turbo",
            segments=[TranscriptSegment("hola", 0.0, 1.5)],
        )
    )
    for required in (
        "job_id: abc",
        "type: raw",
        "source_filename: meeting.m4a",
        "language: es",
        "audio_duration_seconds: 3621.5",
        "model: mlx-community/whisper-large-v3-turbo",
        "vad: silero",
        "delooping",
        "bag_of_hallucinations_filter",
    ):
        assert required in md, f"missing {required!r} in raw md"
    assert "## [00:00:00 → 00:00:01]" in md or "## [00:00:00 → 00:00:02]" in md
    assert "hola" in md


def test_render_clean_md_speaker_summary():
    md = render_clean_md(
        CleanExportInputs(
            job_id="abc",
            source_filename="meeting.wav",
            language="es",
            audio_duration_seconds=600.0,
            phase1_duration_seconds=120.0,
            phase2_duration_seconds=480.0,
            model="mlx-community/whisper-large-v3-turbo",
            segments=[
                AnnotatedSegment("hola", 0.0, 60.0, speaker_id="SPEAKER_00"),
                AnnotatedSegment("adiós", 60.0, 120.0, speaker_id="SPEAKER_01"),
                AnnotatedSegment("una más", 120.0, 130.0, speaker_id="SPEAKER_00"),
            ],
        )
    )
    assert "type: clean" in md
    assert "denoise: demucs/htdemucs" in md
    assert "diarization: pyannote/speaker-diarization-3.1" in md
    assert "speakers_detected: 2" in md
    assert "## Speakers detectados" in md
    assert "`SPEAKER_00`" in md
    assert "`SPEAKER_01`" in md
    # Speaker labels should appear next to their utterances.
    assert "SPEAKER_00" in md
    assert "SPEAKER_01" in md


def test_render_clean_md_no_speakers_omits_summary():
    md = render_clean_md(
        CleanExportInputs(
            job_id="abc",
            source_filename="meeting.wav",
            language="es",
            audio_duration_seconds=10.0,
            phase1_duration_seconds=5.0,
            phase2_duration_seconds=5.0,
            model="m",
            segments=[AnnotatedSegment("hola", 0.0, 1.0, speaker_id=None)],
        )
    )
    assert "speakers_detected: 0" in md
    assert "Speakers detectados" not in md
