"""Voice Activity Detection (Silero VAD) wrapper.

Two paths through this module:

1. ``detect_speech`` — return a list of (start_ms, end_ms) speech segments,
   useful for analytics and as input to the Phase 2 timestamp remap.

2. ``extract_speech_only`` — concatenate just the speech regions into a
   single audio array (so Whisper doesn't waste compute on silence) AND
   record the offset map needed to remap concatenated-domain timestamps
   back to original-domain timestamps in ``postprocess.remap_timestamps``.

The speech-only flow is what feeds the transcriber.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio_io import SAMPLE_RATE
from .config import settings
from .logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SpeechSegment:
    """A continuous region of speech in the source audio (millisecond bounds)."""

    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class TimestampMap:
    """Offset table to remap timestamps from concatenated speech-only audio
    back to the original full-audio timeline.

    ``segments`` is the list of (orig_start_ms, orig_end_ms) regions that were
    kept, in order. To convert a timestamp ``t_concat_ms`` (measured in the
    concatenated domain) to ``t_orig_ms`` (in the original audio):

        cum = 0
        for orig_start, orig_end in segments:
            seg_len = orig_end - orig_start
            if t_concat_ms <= cum + seg_len:
                return orig_start + (t_concat_ms - cum)
            cum += seg_len
        return segments[-1][1]  # past end → clamp to last segment end
    """

    segments: tuple[tuple[int, int], ...]

    @classmethod
    def from_speech_segments(cls, segments: list[SpeechSegment]) -> TimestampMap:
        return cls(segments=tuple((s.start_ms, s.end_ms) for s in segments))

    def remap_ms(self, t_concat_ms: int) -> int:
        """Translate a timestamp in the concatenated domain to the original domain."""
        if not self.segments:
            return t_concat_ms
        cum = 0
        for orig_start, orig_end in self.segments:
            seg_len = orig_end - orig_start
            if t_concat_ms <= cum + seg_len:
                return orig_start + (t_concat_ms - cum)
            cum += seg_len
        return self.segments[-1][1]


class VADProcessor:
    """Wrapper around silero-vad's get_speech_timestamps utility.

    The Silero model is loaded lazily on the first call so importing this
    module stays cheap (matters for tests that don't exercise VAD).
    """

    def __init__(
        self,
        threshold: float | None = None,
        min_speech_ms: int | None = None,
        min_silence_ms: int | None = None,
        speech_pad_ms: int | None = None,
    ) -> None:
        self.threshold = threshold if threshold is not None else settings.vad_threshold
        self.min_speech_ms = (
            min_speech_ms if min_speech_ms is not None else settings.vad_min_speech_ms
        )
        self.min_silence_ms = (
            min_silence_ms if min_silence_ms is not None else settings.vad_min_silence_ms
        )
        self.speech_pad_ms = (
            speech_pad_ms if speech_pad_ms is not None else settings.vad_speech_pad_ms
        )
        self._model = None
        self._get_speech_timestamps = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # silero-vad v5 ships as a pip package with a load_silero_vad helper.
        from silero_vad import get_speech_timestamps, load_silero_vad

        self._model = load_silero_vad()
        self._get_speech_timestamps = get_speech_timestamps
        logger.info("vad_model_loaded", model="silero-vad")

    def detect_speech(
        self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE
    ) -> list[SpeechSegment]:
        """Return all speech regions as (start_ms, end_ms) segments."""
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"VAD expects {SAMPLE_RATE} Hz audio, got {sample_rate}. "
                "Run audio_io.normalize_audio first."
            )
        if audio.size == 0:
            return []

        self._ensure_loaded()

        # Silero takes a torch.Tensor; converting only here keeps numpy at
        # the public API boundary.
        import torch

        tensor = torch.as_tensor(audio, dtype=torch.float32)
        raw = self._get_speech_timestamps(  # type: ignore[misc]
            tensor,
            self._model,
            sampling_rate=sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=False,
        )
        # silero returns sample indices (start/end). Convert to ms.
        segments: list[SpeechSegment] = []
        for span in raw:
            start_ms = int(round(span["start"] / sample_rate * 1000))
            end_ms = int(round(span["end"] / sample_rate * 1000))
            if end_ms > start_ms:
                segments.append(SpeechSegment(start_ms=start_ms, end_ms=end_ms))
        return segments

    def extract_speech_only(
        self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE
    ) -> tuple[np.ndarray, TimestampMap]:
        """Concatenate speech regions and return them with a timestamp map.

        If VAD finds no speech, returns an empty array and an empty map; the
        transcriber should short-circuit in that case rather than feeding an
        empty buffer to Whisper.
        """
        segments = self.detect_speech(audio, sample_rate)
        if not segments:
            return np.zeros(0, dtype=audio.dtype), TimestampMap(segments=())

        pieces: list[np.ndarray] = []
        for seg in segments:
            start_idx = int(seg.start_ms / 1000 * sample_rate)
            end_idx = int(seg.end_ms / 1000 * sample_rate)
            pieces.append(audio[start_idx:end_idx])

        concatenated = np.concatenate(pieces) if pieces else np.zeros(0, dtype=audio.dtype)
        return concatenated, TimestampMap.from_speech_segments(segments)
