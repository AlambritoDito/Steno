"""Tests for steno_server.postprocess.

The delooping cases follow the plan-level spec:
- single word repeated ≥3 times → collapses
- short phrase (2–4 words) repeated ≥3 times → collapses
- long sentence (>10 words) repeated ≥3 times → collapses
- 2× repetition (under threshold) → does NOT collapse
- trivial variation (case/punctuation/whitespace) → normalizes and collapses
"""

from __future__ import annotations

import pytest

from steno_server import postprocess
from steno_server.postprocess import (
    SpeakerSegment,
    TranscriptSegment,
    bag_of_hallucinations_filter,
    clean_transcript,
    delooping,
    merge_speakers_with_text,
    remap_timestamps,
)
from steno_server.vad import TimestampMap


def _seg(text: str, start: float, end: float) -> TranscriptSegment:
    return TranscriptSegment(text=text, start_s=start, end_s=end)


# ---------------------------------------------------------------------------
# delooping
# ---------------------------------------------------------------------------


class TestDelooping:
    def test_empty_input(self):
        assert delooping([]) == []

    def test_single_word_repeated_three_times_collapses(self):
        segs = [_seg("hola", 0, 1), _seg("hola", 1, 2), _seg("hola", 2, 3)]
        out = delooping(segs)
        assert len(out) == 1
        assert "[repetición detectada x 3]" in out[0].text
        assert out[0].start_s == 0
        assert out[0].end_s == 3

    def test_single_word_repeated_five_times_collapses(self):
        segs = [_seg("hola", i, i + 1) for i in range(5)]
        out = delooping(segs)
        assert len(out) == 1
        assert "[repetición detectada x 5]" in out[0].text

    def test_short_phrase_repeated_three_times_collapses(self):
        segs = [
            _seg("buenos días equipo", 0, 1),
            _seg("buenos días equipo", 1, 2),
            _seg("buenos días equipo", 2, 3),
        ]
        out = delooping(segs)
        assert len(out) == 1
        assert "[repetición detectada x 3]" in out[0].text

    def test_long_sentence_repeated_three_times_collapses(self):
        sentence = (
            "vamos a presentar los resultados del proyecto al equipo el viernes "
            "por la mañana"
        )
        segs = [_seg(sentence, i, i + 1) for i in range(3)]
        out = delooping(segs)
        assert len(out) == 1
        assert "[repetición detectada x 3]" in out[0].text

    def test_two_repetitions_below_threshold_pass_through(self):
        segs = [_seg("hola", 0, 1), _seg("hola", 1, 2)]
        out = delooping(segs)
        assert len(out) == 2
        assert all("[repetición" not in s.text for s in out)

    def test_trivial_variation_still_collapses(self):
        """Punctuation, casing, accents, and whitespace should be normalized."""
        segs = [
            _seg("Hola.", 0, 1),
            _seg(" hola ", 1, 2),
            _seg("HOLA!", 2, 3),
        ]
        out = delooping(segs)
        assert len(out) == 1
        assert "[repetición detectada x 3]" in out[0].text

    def test_mixed_run_with_non_repetition_in_between(self):
        segs = [
            _seg("hola", 0, 1),
            _seg("hola", 1, 2),
            _seg("hola", 2, 3),
            _seg("¿cómo estás?", 3, 4),
            _seg("bien gracias", 4, 5),
        ]
        out = delooping(segs)
        assert len(out) == 3
        assert "[repetición detectada x 3]" in out[0].text
        assert out[1].text == "¿cómo estás?"
        assert out[2].text == "bien gracias"

    def test_two_separate_loops(self):
        segs = [
            _seg("hola", 0, 1),
            _seg("hola", 1, 2),
            _seg("hola", 2, 3),
            _seg("adiós", 3, 4),
            _seg("adiós", 4, 5),
            _seg("adiós", 5, 6),
        ]
        out = delooping(segs)
        assert len(out) == 2
        assert "[repetición detectada x 3]" in out[0].text
        assert "[repetición detectada x 3]" in out[1].text

    def test_empty_text_runs_not_collapsed(self):
        # A run of empty/whitespace segments should not collapse — different
        # from real repetition; let downstream handle them.
        segs = [_seg("", 0, 1), _seg("  ", 1, 2), _seg("", 2, 3)]
        out = delooping(segs)
        assert len(out) == 3


# ---------------------------------------------------------------------------
# Bag of Hallucinations filter
# ---------------------------------------------------------------------------


class TestBagOfHallucinationsFilter:
    def test_es_amara_dropped(self):
        segs = [
            _seg("contenido real importante", 0, 5),
            _seg("Subtítulos por la comunidad de Amara.org", 5, 7),
        ]
        out = bag_of_hallucinations_filter(segs, "es")
        assert len(out) == 1
        assert "real" in out[0].text

    def test_en_amara_dropped(self):
        segs = [
            _seg("real meeting content", 0, 5),
            _seg("Subtitles by the Amara.org community", 5, 7),
        ]
        out = bag_of_hallucinations_filter(segs, "en")
        assert len(out) == 1

    def test_unknown_language_passes_through(self):
        segs = [_seg("Subtítulos por la comunidad de Amara.org", 0, 1)]
        out = bag_of_hallucinations_filter(segs, "fr")
        assert out == segs

    def test_case_and_punctuation_insensitive(self):
        segs = [_seg("subtitulos por la comunidad de amara.org!", 0, 1)]
        out = bag_of_hallucinations_filter(segs, "es")
        assert out == []

    def test_does_not_drop_partial_match(self):
        # The phrase contains "Amara.org" but isn't the boilerplate itself.
        segs = [_seg("Mencionó Amara.org como ejemplo de plataforma de subtítulos.", 0, 3)]
        out = bag_of_hallucinations_filter(segs, "es")
        assert len(out) == 1


# ---------------------------------------------------------------------------
# Timestamp remap
# ---------------------------------------------------------------------------


class TestRemapTimestamps:
    def test_no_map_passes_through(self):
        segs = [_seg("hola", 1.5, 2.5)]
        assert remap_timestamps(segs, None) == segs

    def test_offsets_into_original_timeline(self):
        # Concatenated 0-3s = original 5-8s
        m = TimestampMap(segments=((5_000, 8_000),))
        segs = [_seg("hola", 0.5, 1.5)]
        out = remap_timestamps(segs, m)
        assert out[0].start_s == pytest.approx(5.5, abs=0.01)
        assert out[0].end_s == pytest.approx(6.5, abs=0.01)


# ---------------------------------------------------------------------------
# Speaker merge
# ---------------------------------------------------------------------------


class TestMergeSpeakers:
    def test_no_speakers_yields_unannotated(self):
        segs = [_seg("hola", 0, 1)]
        out = merge_speakers_with_text(segs, [])
        assert len(out) == 1
        assert out[0].speaker_id is None

    def test_assigns_overlap_majority(self):
        segs = [_seg("hola", 0, 4)]  # 4-second segment
        speakers = [
            SpeakerSegment("SPEAKER_00", 0, 1),  # 1s overlap
            SpeakerSegment("SPEAKER_01", 1, 4),  # 3s overlap
        ]
        out = merge_speakers_with_text(segs, speakers)
        assert out[0].speaker_id == "SPEAKER_01"

    def test_no_overlap_yields_none(self):
        segs = [_seg("hola", 0, 1)]
        speakers = [SpeakerSegment("SPEAKER_00", 5, 10)]
        out = merge_speakers_with_text(segs, speakers)
        assert out[0].speaker_id is None


# ---------------------------------------------------------------------------
# Convenience clean_transcript
# ---------------------------------------------------------------------------


class TestCleanTranscript:
    def test_runs_delooping_then_filter(self):
        segs = [
            _seg("hola", 0, 1),
            _seg("hola", 1, 2),
            _seg("hola", 2, 3),
            _seg("Subtítulos por la comunidad de Amara.org", 3, 4),
            _seg("contenido real", 4, 5),
        ]
        out = clean_transcript(segs, "es")
        assert len(out) == 2
        assert "[repetición detectada" in out[0].text
        assert out[1].text == "contenido real"


# ---------------------------------------------------------------------------
# Sanity: BoH lists are non-empty and unique within a language
# ---------------------------------------------------------------------------


def test_bag_of_hallucinations_es_unique():
    bag = postprocess.BAG_OF_HALLUCINATIONS_ES
    assert len(bag) >= 5
    assert len(set(bag)) == len(bag)


def test_bag_of_hallucinations_en_unique():
    bag = postprocess.BAG_OF_HALLUCINATIONS_EN
    assert len(bag) >= 5
    assert len(set(bag)) == len(bag)
