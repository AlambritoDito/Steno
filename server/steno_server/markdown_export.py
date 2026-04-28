"""Generate the .md artifacts (transcript-raw.md, transcript-clean.md).

Frontmatter is YAML. We hand-write it (instead of using PyYAML) to keep
the output format under our control and avoid pulling another dependency.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from . import __version__
from .postprocess import AnnotatedSegment, TranscriptSegment


def _format_timestamp(seconds: float) -> str:
    """Format a number of seconds as ``HH:MM:SS``."""
    if seconds < 0:
        seconds = 0.0
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _yaml_value(value) -> str:
    """Render a value into YAML scalar form."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "\n" + "\n".join(f"  - {item}" for item in value)
    return str(value)


def _frontmatter(values: dict[str, object]) -> str:
    lines = ["---"]
    for key, value in values.items():
        lines.append(f"{key}: {_yaml_value(value)}")
    lines.append("---")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Raw transcript (Phase 1)
# ---------------------------------------------------------------------------


@dataclass
class RawExportInputs:
    job_id: str
    source_filename: str
    language: str
    audio_duration_seconds: float
    phase1_duration_seconds: float
    model: str
    segments: list[TranscriptSegment]


def render_raw_md(inputs: RawExportInputs) -> str:
    fm = _frontmatter(
        {
            "job_id": inputs.job_id,
            "type": "raw",
            "source_filename": inputs.source_filename,
            "language": inputs.language,
            "created_at": datetime.now(UTC).isoformat(),
            "audio_duration_seconds": round(inputs.audio_duration_seconds, 2),
            "phase1_duration_seconds": round(inputs.phase1_duration_seconds, 2),
            "model": inputs.model,
            "vad": "silero",
            "hallucination_mitigation": ["delooping", "bag_of_hallucinations_filter"],
            "generator": f"steno-server v{__version__}",
        }
    )
    body = ["", "# Transcripción cruda", "",
            "> Generada en Fase 1. Sin denoise ni diarización. "
            "Para versión limpia, consulta `transcript-clean.md`.", ""]
    for seg in inputs.segments:
        body.append(
            f"## [{_format_timestamp(seg.start_s)} → {_format_timestamp(seg.end_s)}]"
        )
        body.append("")
        body.append(seg.text)
        body.append("")
    return fm + "\n" + "\n".join(body).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Clean transcript (Phase 2)
# ---------------------------------------------------------------------------


@dataclass
class CleanExportInputs:
    job_id: str
    source_filename: str
    language: str
    audio_duration_seconds: float
    phase1_duration_seconds: float
    phase2_duration_seconds: float
    model: str
    segments: list[AnnotatedSegment]


def _summarize_speakers(segments: Iterable[AnnotatedSegment]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for seg in segments:
        if not seg.speaker_id:
            continue
        totals[seg.speaker_id] = totals.get(seg.speaker_id, 0.0) + (seg.end_s - seg.start_s)
    return dict(sorted(totals.items(), key=lambda kv: -kv[1]))


def _format_minutes(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes} min {secs:02d} s"


def render_clean_md(inputs: CleanExportInputs) -> str:
    speakers = _summarize_speakers(inputs.segments)
    fm = _frontmatter(
        {
            "job_id": inputs.job_id,
            "type": "clean",
            "source_filename": inputs.source_filename,
            "language": inputs.language,
            "created_at": datetime.now(UTC).isoformat(),
            "audio_duration_seconds": round(inputs.audio_duration_seconds, 2),
            "phase1_duration_seconds": round(inputs.phase1_duration_seconds, 2),
            "phase2_duration_seconds": round(inputs.phase2_duration_seconds, 2),
            "model": inputs.model,
            "denoise": "demucs/htdemucs",
            "diarization": "pyannote/speaker-diarization-3.1",
            "speakers_detected": len(speakers),
            "generator": f"steno-server v{__version__}",
        }
    )

    body = ["", "# Transcripción procesada", ""]
    for seg in inputs.segments:
        speaker_label = f" {seg.speaker_id}" if seg.speaker_id else ""
        body.append(
            f"## [{_format_timestamp(seg.start_s)} → {_format_timestamp(seg.end_s)}]"
            f"{speaker_label}"
        )
        body.append("")
        body.append(seg.text)
        body.append("")

    if speakers:
        body.append("---")
        body.append("")
        body.append("## Speakers detectados")
        body.append("")
        for speaker_id, total in speakers.items():
            body.append(f"- `{speaker_id}` — duración total: {_format_minutes(total)}")
        body.append("")
        body.append(
            "> Renombra los speakers manualmente reemplazando `SPEAKER_NN` por "
            "el nombre real."
        )
    return fm + "\n" + "\n".join(body).rstrip() + "\n"
