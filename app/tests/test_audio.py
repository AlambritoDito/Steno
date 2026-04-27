"""Tests for steno.audio module."""

import numpy as np
import pytest

from steno.audio import AudioCapture, AudioCaptureError, PORTAUDIO_AVAILABLE
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


# --- v0.2.0: PortAudio graceful detection ---


def test_portaudio_available_flag_is_bool():
    """PORTAUDIO_AVAILABLE is a boolean."""
    assert isinstance(PORTAUDIO_AVAILABLE, bool)


def test_list_devices_raises_when_portaudio_missing(monkeypatch):
    """list_devices() raises AudioCaptureError when PortAudio is missing."""
    import steno.audio as audio_mod
    monkeypatch.setattr(audio_mod, "PORTAUDIO_AVAILABLE", False)
    with pytest.raises(AudioCaptureError, match="PortAudio"):
        AudioCapture.list_devices()


def test_start_raises_when_portaudio_missing(monkeypatch):
    """start() raises AudioCaptureError when PortAudio is missing."""
    import steno.audio as audio_mod
    monkeypatch.setattr(audio_mod, "PORTAUDIO_AVAILABLE", False)
    capture = AudioCapture()
    with pytest.raises(AudioCaptureError, match="PortAudio"):
        capture.start(None, lambda chunk: None)
