"""Tests for steno_server.denoiser.

The Denoiser shells out to demucs as a subprocess. We don't run demucs in
the test suite (multi-minute, GPU-bound, model download). We test the
construction of the command and the post-run file movement by stubbing
asyncio.create_subprocess_exec.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from steno_server.denoiser import DEMUCS_MODEL, Denoiser


class _FakeProc:
    def __init__(self, returncode: int = 0):
        self.returncode = returncode

    async def communicate(self):
        return b"", b""


@pytest.mark.asyncio
async def test_denoise_moves_vocals_wav_to_output(monkeypatch, tmp_path: Path):
    inp = tmp_path / "src.wav"
    inp.write_bytes(b"\x00" * 44)  # bogus, demucs is mocked
    out = tmp_path / "denoised.wav"

    captured = {}

    async def fake_subprocess(*args, **kwargs):
        captured["argv"] = args
        # Emulate demucs's output layout.
        work_dir = Path(args[args.index("-o") + 1])
        target_dir = work_dir / DEMUCS_MODEL / inp.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "vocals.wav").write_bytes(b"vocals")
        return _FakeProc(0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess)

    result = await Denoiser().denoise(inp, out)
    assert result == out
    assert out.exists()
    assert out.read_bytes() == b"vocals"


@pytest.mark.asyncio
async def test_denoise_raises_on_nonzero_exit(monkeypatch, tmp_path: Path):
    inp = tmp_path / "src.wav"
    inp.write_bytes(b"\x00" * 44)
    out = tmp_path / "denoised.wav"

    async def fake_subprocess(*args, **kwargs):
        return _FakeProc(returncode=1)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess)

    with pytest.raises(RuntimeError, match="demucs failed"):
        await Denoiser().denoise(inp, out)


@pytest.mark.asyncio
async def test_denoise_missing_input_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        await Denoiser().denoise(tmp_path / "missing.wav", tmp_path / "out.wav")


def test_best_device_returns_known_value():
    d = Denoiser()
    device = d._best_device()
    assert device in ("mps", "cpu")
