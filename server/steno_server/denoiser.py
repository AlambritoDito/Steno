"""Demucs wrapper — separate voice from background noise.

We use the ``htdemucs`` model with ``--two-stems=vocals`` so the output is
voice vs. everything-else. Backend is MPS when available, CPU fallback.

The actual demucs API runs as a subprocess invocation of ``python -m demucs``
because demucs's Python API is awkward to call programmatically (it expects
``sys.argv``-style args). We construct the command and capture stdout/stderr
for logging.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

from .logging_setup import get_logger

logger = get_logger(__name__)

DEMUCS_MODEL = "htdemucs"


class Denoiser:
    """Run demucs and return the path to the isolated vocals WAV."""

    async def denoise(self, input_path: Path, output_path: Path) -> Path:
        """Separate vocals from the input; write to ``output_path``.

        Demucs always writes to a directory layout (separated/{model}/{stem}/{file}.wav);
        we then move the vocals WAV to the requested output_path and clean up.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Denoiser input not found: {input_path}")

        work_dir = output_path.parent / f".demucs_{output_path.stem}"
        work_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "demucs",
            "--two-stems=vocals",
            "-n",
            DEMUCS_MODEL,
            "-o",
            str(work_dir),
            "--device",
            self._best_device(),
            str(input_path),
        ]
        logger.info("denoise_starting", input=str(input_path), device=cmd[cmd.index("--device") + 1])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.error(
                "denoise_failed",
                returncode=proc.returncode,
                stderr=stderr.decode("utf-8", errors="replace")[-2000:],
            )
            raise RuntimeError(f"demucs failed (exit {proc.returncode})")

        # Demucs writes to: {work_dir}/{model}/{input_stem}/vocals.wav
        produced = work_dir / DEMUCS_MODEL / input_path.stem / "vocals.wav"
        if not produced.exists():
            raise RuntimeError(f"demucs ran but vocals.wav not found at {produced}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(produced), str(output_path))
        shutil.rmtree(work_dir, ignore_errors=True)
        logger.info("denoise_completed", output=str(output_path))
        return output_path

    def _best_device(self) -> str:
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"
