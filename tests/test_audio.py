"""Tests for steno.audio module."""

import numpy as np
import pytest

from steno.audio import AudioCapture
from steno.config import Config


def test_list_devices_returns_list():
    """list_devices() returns a list (can be empty in CI)."""
    devices = AudioCapture.list_devices()
    assert isinstance(devices, list)


def test_audio_capture_instantiation():
    """AudioCapture() instantiates without error."""
    capture = AudioCapture()
    assert capture is not None


def test_is_recording_false_on_init():
    """is_recording() is False before start()."""
    capture = AudioCapture()
    assert capture.is_recording() is False


def test_silence_detection():
    """A zero-filled numpy array is detected as silence."""
    audio = np.zeros(Config.SAMPLE_RATE, dtype=np.float32)
    rms = float(np.sqrt(np.mean(audio**2)))
    assert rms < Config.SILENCE_THRESHOLD


def test_chunk_size_matches_config():
    """Chunk duration matches Config.CHUNK_DURATION."""
    capture = AudioCapture()
    expected_samples = int(Config.SAMPLE_RATE * Config.CHUNK_DURATION)
    assert capture._chunk_samples == expected_samples
