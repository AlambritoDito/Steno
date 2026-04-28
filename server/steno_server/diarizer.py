"""pyannote.audio wrapper — speaker diarization.

Loads ``pyannote/speaker-diarization-3.1`` (gated; needs HF_TOKEN). Backend
is MPS when available, CPU fallback. Returns a list of SpeakerSegment.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from .config import settings
from .logging_setup import get_logger
from .postprocess import SpeakerSegment

logger = get_logger(__name__)


class Diarizer:
    """Wrapper around pyannote.audio's Pipeline.

    Lazy-loads the pipeline on first use so importing this module doesn't
    require HF_TOKEN to be set.
    """

    def __init__(self, hf_token: str | None = None) -> None:
        self._token = hf_token if hf_token is not None else settings.hf_token
        self._pipeline = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        if not self._token:
            raise RuntimeError(
                "HF_TOKEN is not set. pyannote/speaker-diarization-3.1 is gated "
                "behind a free Hugging Face account; export HF_TOKEN before "
                "starting the server."
            )
        from pyannote.audio import Pipeline

        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self._token,
        )
        # Move to MPS if available.
        try:
            import torch

            if torch.backends.mps.is_available():
                self._pipeline.to(torch.device("mps"))
        except Exception:  # noqa: BLE001
            pass
        logger.info("diarizer_loaded", model="pyannote/speaker-diarization-3.1")

    async def diarize(self, audio_path: Path) -> list[SpeakerSegment]:
        """Run diarization. Heavy work runs in the executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._diarize_sync, audio_path)

    def _diarize_sync(self, audio_path: Path) -> list[SpeakerSegment]:
        self._ensure_loaded()
        diarization = self._pipeline(str(audio_path))  # type: ignore[misc]
        out: list[SpeakerSegment] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            out.append(
                SpeakerSegment(
                    speaker_id=str(speaker),
                    start_s=float(turn.start),
                    end_s=float(turn.end),
                )
            )
        return out
