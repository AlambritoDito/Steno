"""Tests for steno_server.audio_io."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from steno_server import audio_io


def _make_wav(path: Path, *, duration_s: float, sample_rate: int = 16_000) -> Path:
    samples = np.zeros(int(duration_s * sample_rate), dtype=np.float32)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")
    return path


class TestValidateUpload:
    def test_accepts_silence_fixture(self, silence_wav: Path):
        result = audio_io.validate_upload(silence_wav)
        assert result.ok is True
        assert result.extension == ".wav"
        assert result.duration_s is not None and result.duration_s >= 4.5

    def test_missing_file(self, tmp_path: Path):
        result = audio_io.validate_upload(tmp_path / "ghost.wav")
        assert result.ok is False
        assert "not found" in (result.error or "").lower()

    def test_unsupported_extension(self, tmp_path: Path):
        bad = tmp_path / "audio.xyz"
        bad.write_bytes(b"junk")
        result = audio_io.validate_upload(bad)
        assert result.ok is False
        assert "unsupported" in (result.error or "").lower()
        assert result.extension == ".xyz"

    def test_too_large(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(audio_io.settings, "max_upload_size_mb", 0)  # everything's "too big"
        wav = _make_wav(tmp_path / "ok.wav", duration_s=2)
        result = audio_io.validate_upload(wav)
        assert result.ok is False
        assert "too large" in (result.error or "").lower()

    def test_too_short(self, tmp_path: Path):
        wav = _make_wav(tmp_path / "tiny.wav", duration_s=0.1)
        result = audio_io.validate_upload(wav)
        assert result.ok is False
        assert "too short" in (result.error or "").lower()

    def test_corrupt_file(self, tmp_path: Path):
        bad = tmp_path / "corrupt.wav"
        bad.write_bytes(b"this is not a valid wav file at all")
        result = audio_io.validate_upload(bad)
        assert result.ok is False
        assert "decode" in (result.error or "").lower()


class TestNormalize:
    def test_already_16k_mono_passes_through(self, tmp_path: Path, silence_wav: Path):
        out = tmp_path / "normalized.wav"
        info = audio_io.normalize_audio(silence_wav, out)
        assert info.sample_rate == 16_000
        assert info.path == out
        assert out.exists()
        with sf.SoundFile(str(out)) as f:
            assert f.samplerate == 16_000
            assert f.channels == 1

    def test_resample_higher_rate(self, tmp_path: Path):
        # Make a 22.05 kHz mono fixture and confirm normalize downsamples.
        src = tmp_path / "22k.wav"
        rate = 22_050
        samples = np.zeros(rate * 2, dtype=np.float32)
        sf.write(str(src), samples, rate, subtype="PCM_16")
        out = tmp_path / "norm.wav"
        info = audio_io.normalize_audio(src, out)
        assert info.sample_rate == 16_000
        with sf.SoundFile(str(out)) as f:
            assert f.samplerate == 16_000

    def test_stereo_to_mono(self, tmp_path: Path):
        src = tmp_path / "stereo.wav"
        rate = 16_000
        samples = np.zeros((rate, 2), dtype=np.float32)
        samples[:, 0] = 0.1  # only one channel non-zero, mean stays non-zero
        sf.write(str(src), samples, rate, subtype="PCM_16")
        out = tmp_path / "mono.wav"
        info = audio_io.normalize_audio(src, out)
        with sf.SoundFile(str(out)) as f:
            assert f.channels == 1
        assert info.duration_s == pytest.approx(1.0, rel=0.05)


class TestChunkAudio:
    def test_yields_correct_number(self):
        # 65 s of audio at 16 kHz, 30-s chunks → 3 chunks (30, 30, 5)
        samples = np.zeros(16_000 * 65, dtype=np.float32)
        chunks = list(audio_io.chunk_audio(samples, sample_rate=16_000, chunk_seconds=30))
        assert len(chunks) == 3
        assert chunks[0].start_s == 0
        assert chunks[0].end_s == 30
        assert chunks[-1].end_s == pytest.approx(65)

    def test_overlap(self):
        samples = np.zeros(16_000 * 60, dtype=np.float32)
        chunks = list(
            audio_io.chunk_audio(
                samples, sample_rate=16_000, chunk_seconds=30, overlap_seconds=5
            )
        )
        # step = 25 s → starts at 0, 25, 50; last ends at 60
        assert [c.start_s for c in chunks] == [0, 25, 50]
        assert chunks[-1].end_s == pytest.approx(60)

    def test_empty_input(self):
        empty = np.zeros(0, dtype=np.float32)
        assert list(audio_io.chunk_audio(empty)) == []

    def test_invalid_chunk_seconds(self):
        with pytest.raises(ValueError):
            list(audio_io.chunk_audio(np.zeros(16_000), chunk_seconds=0))

    def test_invalid_overlap(self):
        with pytest.raises(ValueError):
            list(
                audio_io.chunk_audio(
                    np.zeros(16_000), chunk_seconds=10, overlap_seconds=10
                )
            )


class TestLoadNormalized:
    def test_round_trip(self, tmp_path: Path):
        src_samples = np.random.default_rng(0).uniform(-0.5, 0.5, 16_000).astype(np.float32)
        path = tmp_path / "round.wav"
        sf.write(str(path), src_samples, 16_000, subtype="PCM_16")
        loaded, sr = audio_io.load_normalized(path)
        assert sr == 16_000
        assert loaded.dtype == np.float32
        assert loaded.shape == src_samples.shape
