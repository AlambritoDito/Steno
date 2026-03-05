"""Microphone audio capture for Steno."""

import logging

import numpy as np
import sounddevice as sd

from steno.config import Config

logger = logging.getLogger("steno.audio")


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

    @staticmethod
    def list_devices() -> list[dict]:
        """List available input devices with index, name, and channels."""
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

    def start(self, device_index: int | None, callback) -> None:
        """Start capturing audio.

        Calls callback(chunk: numpy.ndarray) for each chunk
        (float32, mono, 16kHz).
        """
        if self._recording:
            logger.warning("start() called but already recording")
            return

        self._callback = callback
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
        self._buffer.append(audio)

        total = sum(len(b) for b in self._buffer)
        if total >= self._chunk_samples:
            chunk = np.concatenate(self._buffer)[:self._chunk_samples]
            self._buffer = [np.concatenate(self._buffer)[self._chunk_samples - self._overlap_samples:]]

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
