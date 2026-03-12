"""Microphone audio capture for Steno."""

import logging
import wave
from pathlib import Path

import numpy as np

from steno.config import Config

logger = logging.getLogger("steno.audio")

# Gracefully handle missing PortAudio (Issue #5)
PORTAUDIO_AVAILABLE = True
_portaudio_error = ""
try:
    import sounddevice as sd
except OSError as e:
    PORTAUDIO_AVAILABLE = False
    _portaudio_error = str(e)
    sd = None  # type: ignore[assignment]
    logger.error("PortAudio not available: %s", e)


class WavWriter:
    """Incrementally writes audio chunks to a WAV file."""

    def __init__(self, path: Path, sample_rate: int = 16000):
        self._path = path
        self._file = wave.open(str(path), "wb")
        self._file.setnchannels(1)
        self._file.setsampwidth(2)  # 16-bit
        self._file.setframerate(sample_rate)
        self._closed = False
        logger.info("WavWriter opened: %s", path)

    def write(self, chunk: np.ndarray) -> None:
        """Write a float32 audio chunk as 16-bit PCM."""
        if self._closed:
            return
        pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
        self._file.writeframes(pcm.tobytes())

    def close(self) -> None:
        """Finalize and close the WAV file."""
        if not self._closed:
            self._file.close()
            self._closed = True
            logger.info("WavWriter closed: %s", self._path)

    @property
    def path(self) -> Path:
        return self._path


class AudioCaptureError(Exception):
    """Raised when audio capture fails."""


class AudioCapture:
    """Captures audio from a microphone in chunks."""

    def __init__(self):
        self._stream: sd.InputStream | None = None
        self._recording = False
        self._overlap_buffer: np.ndarray | None = None
        self._chunk_samples = int(Config.SAMPLE_RATE * Config.CHUNK_DURATION)
        self._overlap_samples = int(Config.SAMPLE_RATE * Config.OVERLAP_DURATION)
        self._buffer: list[np.ndarray] = []
        self._callback = None
        self._wav_writer: WavWriter | None = None

    @staticmethod
    def list_devices() -> list[dict]:
        """List available input devices with index, name, and channels."""
        if not PORTAUDIO_AVAILABLE:
            raise AudioCaptureError(
                "PortAudio library not found. Please install it: brew install portaudio"
            )
        devices = sd.query_devices()
        result = []
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                result.append({
                    "index": i,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                })
        return result

    def start(self, device_index: int | None, callback, wav_writer: WavWriter | None = None) -> None:
        """Start capturing audio.

        Calls callback(chunk: numpy.ndarray) for each chunk
        (float32, mono, 16kHz).
        If wav_writer is provided, raw audio frames are written to the WAV file.
        """
        if not PORTAUDIO_AVAILABLE:
            raise AudioCaptureError(
                "PortAudio library not found. Please install it: brew install portaudio"
            )
        if self._recording:
            logger.warning("start() called but already recording")
            return

        self._callback = callback
        self._wav_writer = wav_writer
        self._buffer = []
        self._overlap_buffer = None

        logger.info("Starting audio capture: device=%s, sr=%d, chunk=%ds",
                     device_index, Config.SAMPLE_RATE, Config.CHUNK_DURATION)

        try:
            self._stream = sd.InputStream(
                samplerate=Config.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=device_index,
                blocksize=int(Config.SAMPLE_RATE * 0.1),  # 100ms blocks
                callback=self._audio_callback,
            )
            self._stream.start()
            self._recording = True
            logger.info("Audio capture started successfully")
        except Exception as e:
            logger.error("Audio capture failed: %s", e)
            raise AudioCaptureError(f"Could not start audio capture: {e}") from e

    def _audio_callback(self, indata, frames, time_info, status):
        """Internal callback from sounddevice."""
        if status:
            logger.warning("Audio stream status: %s", status)
        audio = indata[:, 0].copy()  # mono

        # Write raw audio to WAV before overlap processing
        if self._wav_writer is not None:
            try:
                self._wav_writer.write(audio)
            except Exception as e:
                logger.error("WavWriter error: %s", e)

        self._buffer.append(audio)

        total = sum(len(b) for b in self._buffer)
        if total >= self._chunk_samples:
            full = np.concatenate(self._buffer)
            chunk = full[:self._chunk_samples]
            # Keep only the overlap tail instead of re-concatenating (Issue #6)
            remaining = full[self._chunk_samples - self._overlap_samples:]
            self._buffer = [remaining] if len(remaining) > 0 else []

            # Prepend overlap from previous chunk
            if self._overlap_buffer is not None:
                chunk = np.concatenate([self._overlap_buffer, chunk])

            self._overlap_buffer = chunk[-self._overlap_samples:]

            if self._callback:
                self._callback(chunk)

    def stop(self) -> None:
        """Stop audio capture."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._recording = False
        self._buffer = []
        self._overlap_buffer = None

    def is_recording(self) -> bool:
        """Return whether audio is currently being captured."""
        return self._recording
