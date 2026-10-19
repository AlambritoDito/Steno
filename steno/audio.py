"""Bounded microphone capture; disk and chunk work never run in PortAudio's callback."""

import logging
import os
import queue
import threading
import wave
from pathlib import Path

import numpy as np

from steno.config import Config

logger = logging.getLogger("steno.audio")
PORTAUDIO_AVAILABLE = True
_portaudio_error = ""
try:
    import sounddevice as sd
except (OSError, ImportError) as exc:
    PORTAUDIO_AVAILABLE = False
    _portaudio_error = str(exc)
    sd = None


class AudioCaptureError(Exception):
    """Audio could not be captured without loss."""


class WavWriter:
    """A single, exclusively created take. Frames are readable while recording."""

    def __init__(self, path: Path, sample_rate: int = 16000):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._raw = self._path.open("xb")
        self._file = wave.open(self._raw, "wb")
        self._file.setnchannels(1)
        self._file.setsampwidth(2)
        self._file.setframerate(sample_rate)
        self._file.writeframes(b"")
        self._raw.flush()
        self._closed = False
        self.frames = 0

    def write(self, chunk: np.ndarray) -> None:
        if self._closed:
            raise AudioCaptureError("Audio archive is closed")
        pcm = (np.clip(chunk, -1, 1) * 32767).astype("<i2")
        self._file.writeframes(pcm.tobytes())
        self._raw.flush()
        # Publish the cursor only after bytes and the WAV header are visible.
        self.frames += len(pcm)

    def close(self) -> None:
        if not self._closed:
            try:
                self._file.close()
                self._raw.flush()
                os.fsync(self._raw.fileno())
            finally:
                self._raw.close()
                self._closed = True

    @property
    def path(self) -> Path:
        return self._path


def read_audio_range(path: Path, start: int, end: int) -> np.ndarray:
    """Read just one archived mono PCM chunk, never the full recording."""
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise AudioCaptureError("Expected mono 16-bit PCM audio")
        wav.setpos(min(start, wav.getnframes()))
        data = wav.readframes(max(0, end - start))
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


class AudioCapture:
    def __init__(self):
        self._stream = None
        self._recording = False
        self._chunk_samples = int(Config.SAMPLE_RATE * Config.CHUNK_DURATION)
        self._overlap_samples = int(Config.SAMPLE_RATE * Config.OVERLAP_DURATION)
        self._buffer = np.empty(0, dtype=np.float32)
        self._overlap_buffer = np.empty(0, dtype=np.float32)
        self._callback = None
        self._wav_writer = None
        self._queue = queue.Queue(maxsize=128)
        self._stop_event = threading.Event()
        self._worker = None
        self.error = None

    @staticmethod
    def list_devices() -> list[dict]:
        if not PORTAUDIO_AVAILABLE:
            raise AudioCaptureError("PortAudio is unavailable; reinstall Steno or check audio permissions")
        return [dict(index=i, name=d["name"], channels=d["max_input_channels"])
                for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]

    @property
    def frames_written(self) -> int:
        return self._wav_writer.frames if self._wav_writer else 0

    def start(self, device_index, callback=None, wav_writer=None) -> None:
        if not PORTAUDIO_AVAILABLE:
            raise AudioCaptureError("PortAudio is unavailable; reinstall Steno or check audio permissions")
        if self._recording:
            raise AudioCaptureError("Already recording")
        self._callback = callback
        self._wav_writer = wav_writer
        self._buffer = np.empty(0, dtype=np.float32)
        self._overlap_buffer = np.empty(0, dtype=np.float32)
        self._queue = queue.Queue(maxsize=128)
        self._stop_event.clear()
        self.error = None
        try:
            self._stream = sd.InputStream(
                samplerate=Config.SAMPLE_RATE, channels=1, dtype="float32",
                device=device_index, blocksize=int(Config.SAMPLE_RATE * 0.1),
                callback=self._audio_callback,
            )
            self._recording = True
            self._worker = threading.Thread(target=self._consume, name="steno-audio", daemon=True)
            self._worker.start()
            self._stream.start()
        except Exception as exc:
            self.stop()
            raise AudioCaptureError(f"Could not start audio capture: {exc}") from exc

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            self.error = f"Audio input interrupted: {status}"
            self._recording = False
            self._stop_event.set()
            raise sd.CallbackAbort
        try:
            self._queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            self.error = "Audio storage cannot keep up; recording stopped to prevent silent loss"
            self._recording = False
            self._stop_event.set()
            raise sd.CallbackAbort

    def _consume(self):
        try:
            while not self._stop_event.is_set() or not self._queue.empty():
                try:
                    audio = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if self._wav_writer:
                    self._wav_writer.write(audio)
                if self._callback:
                    self._buffer = np.concatenate((self._buffer, audio))
                    while len(self._buffer) >= self._chunk_samples:
                        current = self._buffer[:self._chunk_samples]
                        self._buffer = self._buffer[self._chunk_samples:]
                        self._emit(current)
            if self._callback and len(self._buffer):
                self._emit(self._buffer)
                self._buffer = np.empty(0, dtype=np.float32)
        except Exception as exc:
            logger.exception("Audio archive failed")
            self.error = str(exc)
            self._recording = False
            self._stop_event.set()

    def _emit(self, current):
        self._callback(np.concatenate((self._overlap_buffer, current)))
        self._overlap_buffer = current[-self._overlap_samples:] if self._overlap_samples else current[:0]

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        try:
            if stream is not None:
                try:
                    stream.stop()
                finally:
                    stream.close()
        except Exception as exc:
            self.error = self.error or f"Audio device disconnected: {exc}"
            logger.warning("Error closing audio device: %s", exc)
        finally:
            self._recording = False
            self._stop_event.set()
            if self._worker and self._worker is not threading.current_thread():
                self._worker.join()
            self._worker = None

    def is_recording(self) -> bool:
        return self._recording
