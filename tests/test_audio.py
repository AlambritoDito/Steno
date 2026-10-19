"""Tests for steno.audio module."""

import numpy as np
import pytest

from steno.audio import AudioCapture, AudioCaptureError, PORTAUDIO_AVAILABLE
from steno.config import Config


def test_list_devices_returns_list(monkeypatch):
    """list_devices() returns a list (can be empty in CI)."""
    import steno.audio as audio_module
    from types import SimpleNamespace
    monkeypatch.setattr(audio_module, "PORTAUDIO_AVAILABLE", True)
    monkeypatch.setattr(audio_module, "sd", SimpleNamespace(query_devices=lambda: []))
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


def test_overlap_only_once_and_partial_tail_flushed():
    capture = AudioCapture()
    capture._chunk_samples = 10
    capture._overlap_samples = 2
    chunks = []
    capture._callback = chunks.append
    capture._queue.put(np.arange(25, dtype=np.float32))
    capture._stop_event.set()
    capture._consume()
    assert [chunk.tolist() for chunk in chunks] == [list(range(10)), list(range(8, 20)), list(range(18, 25))]


def test_writer_will_not_truncate_existing_audio(tmp_path):
    from steno.audio import WavWriter
    path = tmp_path / "take.wav"
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        WavWriter(path)
    assert path.read_bytes() == b"original"


def test_audio_callback_does_not_write_disk():
    capture = AudioCapture()
    class ForbiddenWriter:
        def write(self, data):
            raise AssertionError("Disk I/O in callback")
    capture._wav_writer = ForbiddenWriter()
    capture._audio_callback(np.ones((100, 1)), 100, None, None)
    assert capture._queue.qsize() == 1


def test_queue_overload_is_explicit_error(monkeypatch):
    from types import SimpleNamespace
    import steno.audio as audio_mod
    class Abort(Exception):
        pass
    monkeypatch.setattr(audio_mod, "sd", SimpleNamespace(CallbackAbort=Abort))
    capture = AudioCapture()
    for _ in range(capture._queue.maxsize):
        capture._queue.put(np.zeros(1))
    with pytest.raises(Abort):
        capture._audio_callback(np.ones((1, 1)), 1, None, None)
    assert capture.error
    assert not capture.is_recording()
