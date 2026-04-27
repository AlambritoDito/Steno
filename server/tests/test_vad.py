"""Tests for steno_server.vad."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from steno_server import vad
from steno_server.audio_io import load_normalized


class TestTimestampMap:
    def test_empty_map_passes_through(self):
        m = vad.TimestampMap(segments=())
        assert m.remap_ms(1234) == 1234

    def test_single_segment(self):
        # Speech runs from 5 s to 15 s in the original; concatenated domain
        # has the same 0-10 s slice mapped back to 5-15 s.
        m = vad.TimestampMap(segments=((5_000, 15_000),))
        assert m.remap_ms(0) == 5_000
        assert m.remap_ms(5_000) == 10_000
        assert m.remap_ms(10_000) == 15_000

    def test_multi_segment_remap(self):
        # Original timeline: [2-5s speech] [silence] [10-13s speech]
        # Concatenated:      0----3s        |        3----6s
        m = vad.TimestampMap(segments=((2_000, 5_000), (10_000, 13_000)))
        # Within first speech segment
        assert m.remap_ms(0) == 2_000
        assert m.remap_ms(1_500) == 3_500
        # Boundary: 3000 ms in concat = end of first segment (5000 ms orig)
        assert m.remap_ms(3_000) == 5_000
        # Inside second segment: concat 4500ms = first segment is 3000ms long,
        # so 1500 ms into second = 10000+1500 = 11500.
        assert m.remap_ms(4_500) == 11_500

    def test_overshoot_clamps_to_last_segment_end(self):
        m = vad.TimestampMap(segments=((1_000, 2_000),))
        assert m.remap_ms(99_999) == 2_000


class TestVADProcessor:
    @pytest.fixture(scope="class")
    def processor(self) -> vad.VADProcessor:
        return vad.VADProcessor()

    def test_silence_returns_no_segments(self, processor: vad.VADProcessor):
        silence = np.zeros(16_000 * 5, dtype=np.float32)
        segments = processor.detect_speech(silence)
        assert segments == []

    def test_silence_extract_returns_empty(self, processor: vad.VADProcessor):
        silence = np.zeros(16_000 * 5, dtype=np.float32)
        audio_only, ts_map = processor.extract_speech_only(silence)
        assert audio_only.size == 0
        assert ts_map.segments == ()

    def test_clean_es_detects_speech(
        self, processor: vad.VADProcessor, clean_es_wav: Path
    ):
        samples, sr = load_normalized(clean_es_wav)
        assert sr == 16_000
        segments = processor.detect_speech(samples)
        assert len(segments) >= 1
        # At least 50% of the audio should be marked as speech (it's a clean
        # TTS sample with minimal silence).
        speech_ms = sum(s.duration_ms for s in segments)
        total_ms = (samples.size / sr) * 1000
        assert speech_ms / total_ms >= 0.5, (
            f"VAD captured only {speech_ms/total_ms:.0%} of clean audio as speech; "
            f"expected ≥50%"
        )

    def test_extract_speech_only_remap_within_100ms(
        self, processor: vad.VADProcessor, clean_es_wav: Path
    ):
        samples, sr = load_normalized(clean_es_wav)
        concatenated, ts_map = processor.extract_speech_only(samples)
        assert concatenated.size > 0
        assert ts_map.segments  # at least one segment
        # Round-trip check: the start of the concatenated audio (0 ms in
        # concat domain) should map back to within 100 ms of the first
        # speech segment's start.
        first_start_orig = ts_map.segments[0][0]
        remapped = ts_map.remap_ms(0)
        assert abs(remapped - first_start_orig) < 100

    def test_wrong_sample_rate_raises(self, processor: vad.VADProcessor):
        with pytest.raises(ValueError, match="16000"):
            processor.detect_speech(np.zeros(8_000, dtype=np.float32), sample_rate=8_000)
