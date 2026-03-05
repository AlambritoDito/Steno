"""Tests for steno.transcriber module."""

import numpy as np
import pytest

from steno.config import Config
from steno.transcriber import Transcriber


def test_transcriber_instantiation():
    """Instantiates without loading the model."""
    t = Transcriber()
    assert t is not None


def test_is_loaded_false_on_init():
    """is_loaded() returns False before first transcription."""
    t = Transcriber()
    assert t.is_loaded() is False


def test_get_model_info_structure():
    """get_model_info() returns dict with model_name and loaded."""
    t = Transcriber()
    info = t.get_model_info()
    assert isinstance(info, dict)
    assert "model_name" in info
    assert "loaded" in info
    assert info["loaded"] is False


@pytest.mark.asyncio
async def test_silence_returns_empty_string():
    """Silence audio (zeros) returns ''."""
    t = Transcriber()
    audio = np.zeros(Config.SAMPLE_RATE, dtype=np.float32)
    result = await t.transcribe(audio)
    assert result == ""


def test_transcriber_has_correct_model_name():
    """Stored model name matches config."""
    t = Transcriber()
    assert t.get_model_info()["model_name"] == Config.MODEL_NAME
