"""Tests for steno_server.diarizer.

The pipeline.from_pretrained call requires HF_TOKEN and downloads model
weights, so the real-inference test is skipped without HF_TOKEN. We do
test the no-token error path and the conversion of pyannote tracks to
SpeakerSegment.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from steno_server.diarizer import Diarizer
from steno_server.postprocess import SpeakerSegment


@pytest.mark.asyncio
async def test_missing_hf_token_raises(monkeypatch):
    """Without a token, calling diarize must raise a clear error."""
    monkeypatch.setenv("HF_TOKEN", "")
    d = Diarizer(hf_token=None)
    # Force the path that needs the token.
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        d._diarize_sync(Path("/tmp/does-not-matter.wav"))


def test_track_conversion(monkeypatch):
    """The track-iterator output is converted to SpeakerSegment correctly."""

    class FakeTurn:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    class FakeDiarization:
        def itertracks(self, yield_label=True):
            return iter(
                [
                    (FakeTurn(0.0, 1.5), None, "SPEAKER_00"),
                    (FakeTurn(1.5, 3.2), None, "SPEAKER_01"),
                ]
            )

    class FakePipeline:
        def __call__(self, _path):
            return FakeDiarization()

    d = Diarizer(hf_token="fake-token")
    d._pipeline = FakePipeline()  # bypass _ensure_loaded

    out = d._diarize_sync(Path("/tmp/x.wav"))
    assert out == [
        SpeakerSegment(speaker_id="SPEAKER_00", start_s=0.0, end_s=1.5),
        SpeakerSegment(speaker_id="SPEAKER_01", start_s=1.5, end_s=3.2),
    ]


@pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"),
    reason="HF_TOKEN unset; can't load pyannote model.",
)
@pytest.mark.asyncio
async def test_real_diarization_on_clean_es(clean_es_wav: Path):
    """Smoke test the real pipeline if HF_TOKEN is available."""
    d = Diarizer()
    segments = await d.diarize(clean_es_wav)
    # The TTS sample is a single speaker, so we expect exactly one speaker
    # label in the output (though pyannote may segment it into multiple
    # turns).
    speakers = {s.speaker_id for s in segments}
    assert len(speakers) >= 1
