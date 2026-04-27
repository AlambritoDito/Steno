"""Post-processing for raw Whisper output.

Whisper has two notorious failure modes that the streaming model itself
doesn't fix:

1. **Looping** — the model gets stuck and emits the same phrase 3, 5, 50
   times in a row. Detected by ``delooping`` and collapsed.

2. **Boilerplate hallucinations** — when input is silence or noise, Whisper
   sometimes regurgitates strings from its training data ("Subtítulos por
   la comunidad de Amara.org", "Thanks for watching", etc.). Filtered by
   ``bag_of_hallucinations_filter``.

This module also contains helpers used by Phase 2: remapping concatenated
speech-only timestamps back to the original audio timeline, and merging
diarization output (speaker labels) with transcript segments.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, replace
from typing import Iterable

# ---------------------------------------------------------------------------
# Bag of Hallucinations (extend cautiously — false positives are worse than
# letting one through, since users can edit the .md afterwards)
# ---------------------------------------------------------------------------

BAG_OF_HALLUCINATIONS_ES: tuple[str, ...] = (
    "Subtítulos por la comunidad de Amara.org",
    "Subtítulos realizados por la comunidad de Amara.org",
    "Subtítulos por Amara.org",
    "¡Suscríbete!",
    "¡Suscríbete al canal!",
    "Gracias por ver el video",
    "Muchas gracias por ver el video",
    "No olvides suscribirte",
    "Subtitulado por la comunidad de Amara.org",
)

BAG_OF_HALLUCINATIONS_EN: tuple[str, ...] = (
    "Thanks for watching!",
    "Thank you for watching",
    "Don't forget to subscribe",
    "Subtitles by the Amara.org community",
    "Please subscribe",
    "Like and subscribe",
    "Like, comment and subscribe",
)

BAG_OF_HALLUCINATIONS: dict[str, tuple[str, ...]] = {
    "es": BAG_OF_HALLUCINATIONS_ES,
    "en": BAG_OF_HALLUCINATIONS_EN,
}

# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TranscriptSegment:
    """A single Whisper segment (text + timestamps in seconds)."""

    text: str
    start_s: float
    end_s: float
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def with_metadata(self, **kv: str) -> "TranscriptSegment":
        merged = dict(self.metadata) | {k: str(v) for k, v in kv.items()}
        return replace(self, metadata=tuple(sorted(merged.items())))


@dataclass(frozen=True)
class SpeakerSegment:
    """A diarization region: which speaker spoke from start_s to end_s."""

    speaker_id: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class AnnotatedSegment:
    """Transcript segment after merging diarization."""

    text: str
    start_s: float
    end_s: float
    speaker_id: str | None = None


# ---------------------------------------------------------------------------
# Delooping
# ---------------------------------------------------------------------------

_REPETITION_THRESHOLD = 3  # collapse runs of >= this many consecutive repeats


def _normalize_for_comparison(text: str) -> str:
    """Lowercase + strip punctuation + collapse whitespace + drop accents.

    Used to detect repetition with trivial variation (case/punctuation/etc.)
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def delooping(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """Collapse runs of consecutive segments whose normalized text is identical.

    A run of N >= ``_REPETITION_THRESHOLD`` identical segments collapses into
    a single segment whose text is the original (un-normalized) first
    occurrence's text plus a marker ``[repetición detectada x N]``. Timestamps
    span the full run (start of first, end of last). Runs shorter than the
    threshold are left untouched.

    The comparison normalizes case, accents, punctuation, and whitespace so
    Whisper's casual variations don't hide a real loop.
    """
    if not segments:
        return []

    out: list[TranscriptSegment] = []
    i = 0
    while i < len(segments):
        current = segments[i]
        normalized = _normalize_for_comparison(current.text)
        # Only treat non-empty normalized text as a candidate for delooping;
        # don't collapse runs of empty/whitespace-only segments.
        if not normalized:
            out.append(current)
            i += 1
            continue

        run_end = i + 1
        while (
            run_end < len(segments)
            and _normalize_for_comparison(segments[run_end].text) == normalized
        ):
            run_end += 1

        run_length = run_end - i
        if run_length >= _REPETITION_THRESHOLD:
            collapsed = TranscriptSegment(
                text=f"{current.text.rstrip()} [repetición detectada x {run_length}]",
                start_s=current.start_s,
                end_s=segments[run_end - 1].end_s,
                metadata=current.metadata,
            ).with_metadata(deloop_run_length=str(run_length))
            out.append(collapsed)
        else:
            out.extend(segments[i:run_end])
        i = run_end

    return out


# ---------------------------------------------------------------------------
# Bag of Hallucinations filter
# ---------------------------------------------------------------------------


def bag_of_hallucinations_filter(
    segments: list[TranscriptSegment],
    language: str,
) -> list[TranscriptSegment]:
    """Drop segments whose normalized text matches a known boilerplate
    hallucination for the given language.

    Languages outside the dictionary pass through untouched (no false
    positives is more important than perfect coverage).
    """
    bag = BAG_OF_HALLUCINATIONS.get(language)
    if not bag:
        return list(segments)

    normalized_bag = {_normalize_for_comparison(phrase) for phrase in bag}
    out: list[TranscriptSegment] = []
    for seg in segments:
        if _normalize_for_comparison(seg.text) in normalized_bag:
            continue
        out.append(seg)
    return out


# ---------------------------------------------------------------------------
# Timestamp remap (concatenated speech-only domain → original domain)
# ---------------------------------------------------------------------------


def remap_timestamps(
    segments: list[TranscriptSegment],
    timestamp_map,  # vad.TimestampMap; avoid circular import at module load
) -> list[TranscriptSegment]:
    """Translate segment start/end seconds from concatenated to original domain."""
    if timestamp_map is None or not segments:
        return list(segments)
    out: list[TranscriptSegment] = []
    for seg in segments:
        start_orig = timestamp_map.remap_ms(int(round(seg.start_s * 1000))) / 1000
        end_orig = timestamp_map.remap_ms(int(round(seg.end_s * 1000))) / 1000
        if end_orig < start_orig:
            end_orig = start_orig
        out.append(replace(seg, start_s=start_orig, end_s=end_orig))
    return out


# ---------------------------------------------------------------------------
# Diarization merge
# ---------------------------------------------------------------------------


def merge_speakers_with_text(
    transcript_segments: list[TranscriptSegment],
    speaker_segments: list[SpeakerSegment],
) -> list[AnnotatedSegment]:
    """Annotate each transcript segment with the speaker that overlaps it most.

    For each transcript segment, sum the overlap (in seconds) against each
    speaker segment and assign the speaker with the largest overlap. If no
    speaker segments overlap, ``speaker_id`` is None.
    """
    if not transcript_segments:
        return []
    if not speaker_segments:
        return [
            AnnotatedSegment(text=s.text, start_s=s.start_s, end_s=s.end_s, speaker_id=None)
            for s in transcript_segments
        ]

    out: list[AnnotatedSegment] = []
    for ts in transcript_segments:
        best_speaker: str | None = None
        best_overlap = 0.0
        for sp in speaker_segments:
            overlap = max(0.0, min(ts.end_s, sp.end_s) - max(ts.start_s, sp.start_s))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = sp.speaker_id
        out.append(
            AnnotatedSegment(
                text=ts.text,
                start_s=ts.start_s,
                end_s=ts.end_s,
                speaker_id=best_speaker,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Helpers for callers that want the full clean-up in one call
# ---------------------------------------------------------------------------


def clean_transcript(
    segments: Iterable[TranscriptSegment],
    language: str,
) -> list[TranscriptSegment]:
    """Apply delooping followed by Bag-of-Hallucinations filtering."""
    deduped = delooping(list(segments))
    return bag_of_hallucinations_filter(deduped, language)
