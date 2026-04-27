"""Tests for steno_server.transcriber.

The "real inference" tests are gated by ``needs_ffmpeg`` because mlx-whisper
shells out to ffmpeg for all audio decoding. They use ``whisper-tiny`` via
the ``STENO_SERVER_TEST_MODEL`` env override to keep download cost low.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from steno_server import transcriber
from steno_server.config import settings
from steno_server.postprocess import TranscriptSegment

from .conftest import needs_ffmpeg


def _set_test_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STENO_SERVER_TEST_MODEL", "mlx-community/whisper-tiny")


class TestWhisperCallKwargs:
    """The anti-hallucination contract is encoded in the kwargs dict; assert
    every load-bearing key is present and has the configured value."""

    def test_all_anti_hallucination_keys_present(self):
        kw = transcriber.whisper_call_kwargs("es")
        for key in (
            "language",
            "task",
            "temperature",
            "condition_on_previous_text",
            "compression_ratio_threshold",
            "logprob_threshold",
            "no_speech_threshold",
            "hallucination_silence_threshold",
            "word_timestamps",
            "fp16",
        ):
            assert key in kw, f"missing {key} in whisper_call_kwargs output"

    def test_temperature_is_zero(self):
        assert transcriber.whisper_call_kwargs("es")["temperature"] == 0.0

    def test_condition_on_previous_text_disabled(self):
        assert transcriber.whisper_call_kwargs("es")["condition_on_previous_text"] is False

    def test_language_passed_through(self):
        assert transcriber.whisper_call_kwargs("en")["language"] == "en"

    def test_word_timestamps_enabled_by_default(self):
        assert transcriber.whisper_call_kwargs("es")["word_timestamps"] is True


class TestTranscriberLifecycle:
    def test_default_model_from_settings(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("STENO_SERVER_TEST_MODEL", raising=False)
        t = transcriber.Transcriber()
        assert t.model_name == settings.model_name

    def test_test_model_override(self, monkeypatch: pytest.MonkeyPatch):
        _set_test_model(monkeypatch)
        t = transcriber.Transcriber()
        assert t.model_name == "mlx-community/whisper-tiny"

    def test_starts_unloaded(self, monkeypatch: pytest.MonkeyPatch):
        _set_test_model(monkeypatch)
        t = transcriber.Transcriber()
        assert t.is_loaded is False

    def test_set_language_validates(self):
        t = transcriber.Transcriber()
        with pytest.raises(ValueError, match="Unsupported language"):
            t.set_language("xyz")

    def test_set_language_accepts_supported(self):
        t = transcriber.Transcriber()
        t.set_language("en")
        assert t.language == "en"


class TestSegmentsFromResult:
    def test_empty_result(self):
        assert transcriber._segments_from_result({}) == []
        assert transcriber._segments_from_result({"segments": []}) == []

    def test_strips_whitespace_and_drops_empty(self):
        result = {
            "segments": [
                {"text": "  hola  ", "start": 0.0, "end": 1.0},
                {"text": "", "start": 1.0, "end": 2.0},
                {"text": "  ", "start": 2.0, "end": 3.0},
                {"text": "mundo", "start": 3.0, "end": 4.0},
            ]
        }
        out = transcriber._segments_from_result(result)
        assert [s.text for s in out] == ["hola", "mundo"]
        assert out[0].start_s == 0.0
        assert out[0].end_s == 1.0


class TestTranscribeWithMockedMLX:
    """Verify the executor + segment conversion plumbing without loading
    the real model. Patches the module-level ``_do_transcribe`` shim."""

    def test_transcribe_segment_returns_typed_segments(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        captured_kwargs: dict = {}

        def fake_transcribe(audio, kwargs, *, model_path):
            captured_kwargs.update(kwargs)
            captured_kwargs["model_path"] = model_path
            return {
                "segments": [
                    {"text": "uno", "start": 0.0, "end": 0.5},
                    {"text": "dos", "start": 0.5, "end": 1.0},
                ]
            }

        monkeypatch.setattr(transcriber, "_do_transcribe", fake_transcribe)
        # Avoid actually importing mlx_whisper in _load by short-circuiting it.
        monkeypatch.setattr(transcriber.Transcriber, "_load", lambda self: None)

        t = transcriber.Transcriber(language="es")
        import numpy as np

        result = asyncio.run(t.transcribe_segment(np.zeros(16_000, dtype=np.float32), 16_000))
        assert len(result) == 2
        assert all(isinstance(s, TranscriptSegment) for s in result)
        # Anti-hallucination params actually reached the call:
        assert captured_kwargs["temperature"] == 0.0
        assert captured_kwargs["condition_on_previous_text"] is False
        assert captured_kwargs["language"] == "es"


# ---------------------------------------------------------------------------
# Real inference (skipped when ffmpeg is missing)
# ---------------------------------------------------------------------------


@needs_ffmpeg
class TestRealInference:
    def test_silence_residual_segments_at_most_one(
        self, monkeypatch: pytest.MonkeyPatch, silence_wav: Path
    ):
        """Permissive contract on the transcriber alone: ``mlx-whisper`` may
        emit zero or one residual segments on pure silence. The strict
        no-text contract lives in test_pipeline (Phase 3b)."""
        _set_test_model(monkeypatch)

        async def run():
            t = transcriber.Transcriber(language="es")
            return await t.transcribe_streaming(
                silence_wav, "es", on_chunk=lambda chunk: _noop()
            )

        async def _noop():
            return None

        segments = asyncio.run(run())
        assert len(segments) <= 1, (
            f"Expected ≤1 residual segment on silence; got {len(segments)}: {segments}"
        )
