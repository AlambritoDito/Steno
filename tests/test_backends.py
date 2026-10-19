"""Exercise device fallback and lazy inference without requiring CUDA or Metal."""

import sys
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from steno.backends.base import BackendInferenceError, BackendUnavailableError, decode_audio, iter_audio_chunks, validate_model_files
from steno.backends.faster_whisper import FasterWhisperBackend
from steno.backends.mlx import MLXBackend


def install_fake_ct2(monkeypatch, *, cuda_count=1, fail_cuda=False, fail_job=False):
    calls = []
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(
        get_cuda_device_count=lambda: cuda_count,
        get_supported_compute_types=lambda device: {"float16"} if device == "cuda" else {"int8"},
    ))

    class FakeModel:
        def __init__(self, path, **kwargs):
            calls.append(("load", kwargs))
            self.device = kwargs["device"]
            self.model = SimpleNamespace(unload_model=lambda: calls.append(("unload", self.device)))

        def transcribe(self, audio, **kwargs):
            def generate():
                calls.append(("inference", self.device))
                if self.device == "cuda" and fail_cuda:
                    raise RuntimeError("cuDNN DLL missing")
                if kwargs["beam_size"] != 1 and fail_job:
                    raise RuntimeError("CUDA out of memory")
                yield SimpleNamespace(start=0.25, end=0.8, text=" hello ")
            return generate(), SimpleNamespace()

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeModel))
    return calls


def test_cuda_float16_selected_and_real_lazy_probe_consumed(monkeypatch, tmp_path):
    calls = install_fake_ct2(monkeypatch)
    backend = FasterWhisperBackend()
    backend.load(tmp_path)
    assert backend.capabilities()["validated"]
    assert ("inference", "cuda") in calls
    assert calls[0][1]["compute_type"] == "float16"
    assert calls[0][1]["local_files_only"] is True


def test_missing_cuda_uses_cpu_int8(monkeypatch, tmp_path):
    calls = install_fake_ct2(monkeypatch, cuda_count=0)
    backend = FasterWhisperBackend()
    backend.load(tmp_path)
    assert backend.capabilities()["device"] == "cpu"
    assert backend.capabilities()["fallback_reason"] == "cuda_unavailable"
    assert calls[0][1]["compute_type"] == "int8"
    assert backend.capabilities()["cpu_available"] is True


def test_cpu_capability_requires_actual_int8_runtime_probe(monkeypatch):
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(
        get_cuda_device_count=lambda: 0,
        get_supported_compute_types=lambda device: {"float32"},
    ))
    capabilities = FasterWhisperBackend().capabilities()
    assert not capabilities["available"]
    assert capabilities["cpu_available"] is False
    assert capabilities["fallback_reason"] == "cpu_int8_unavailable"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows native runtime regression")
def test_installed_windows_runtime_imports_and_supports_cpu_without_model():
    import ctranslate2
    import faster_whisper
    import onnxruntime
    import pkg_resources
    assert "int8" in ctranslate2.get_supported_compute_types("cpu")
    assert faster_whisper.WhisperModel is not None
    assert "CPUExecutionProvider" in onnxruntime.get_available_providers()
    assert callable(pkg_resources.resource_filename)


def test_missing_cuda_dll_falls_back_only_during_warmup(monkeypatch, tmp_path):
    calls = install_fake_ct2(monkeypatch, fail_cuda=True)
    backend = FasterWhisperBackend()
    backend.load(tmp_path)
    assert backend.capabilities()["device"] == "cpu"
    assert backend.capabilities()["fallback_reason"] == "cuda_initialization_failed"
    assert [value["device"] for action, value in calls if action == "load"] == ["cuda", "cpu"]


def test_explicit_cuda_never_silently_falls_back(monkeypatch, tmp_path):
    calls = install_fake_ct2(monkeypatch, fail_cuda=True)
    backend = FasterWhisperBackend(device="cuda")
    with pytest.raises(BackendUnavailableError):
        backend.load(tmp_path)
    assert [value["device"] for action, value in calls if action == "load"] == ["cuda"]


def test_runtime_cuda_failure_returns_retryable_error_without_replay(monkeypatch, tmp_path):
    calls = install_fake_ct2(monkeypatch, fail_job=True)
    backend = FasterWhisperBackend()
    backend.load(tmp_path)
    with pytest.raises(BackendInferenceError) as exc:
        backend.transcribe(np.ones(16000, dtype=np.float32), "en")
    assert exc.value.retry_cpu
    assert backend.capabilities()["device"] == "cuda"
    assert len([call for call in calls if call[0] == "load"]) == 1


def test_ct2_normalizes_segments_and_unloads_native_owner(monkeypatch, tmp_path):
    calls = install_fake_ct2(monkeypatch)
    backend = FasterWhisperBackend(device="cpu")
    backend.load(tmp_path)
    assert backend.transcribe(np.ones(16000), "es") == [{"start": 0.25, "end": 0.8, "text": "hello"}]
    backend.unload()
    assert ("unload", "cpu") in calls
    assert not backend.capabilities()["validated"]


def test_mlx_unload_clears_actual_library_cache(monkeypatch, tmp_path):
    holder = SimpleNamespace(model=object(), model_path=str(tmp_path))
    cleared = []
    module = SimpleNamespace(transcribe=lambda *a, **kw: {"segments": []})
    monkeypatch.setitem(sys.modules, "mlx_whisper", module)
    monkeypatch.setitem(sys.modules, "mlx_whisper.transcribe", SimpleNamespace(ModelHolder=holder))
    monkeypatch.setitem(sys.modules, "mlx.core", SimpleNamespace(clear_cache=lambda: cleared.append(True)))
    backend = MLXBackend()
    backend.load(tmp_path)
    backend.unload()
    assert holder.model is None
    assert holder.model_path is None
    assert cleared == [True]


def test_pyav_decodes_stereo_wav_to_16khz_mono_without_subprocess(tmp_path):
    path = tmp_path / "stereo.wav"
    audio = np.full((4800, 2), 4096, dtype=np.int16)
    with wave.open(str(path), "wb") as stream:
        stream.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
        stream.writeframes(audio.tobytes())
    decoded = decode_audio(str(path))
    assert decoded.dtype == np.float32
    assert decoded.ndim == 1
    assert len(decoded) == 1600
    assert np.isfinite(decoded).all()


def test_decode_iterator_bounds_memory_and_keeps_final_partial_window(tmp_path):
    path = tmp_path / "long.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        for _ in range(61):
            stream.writeframes(np.full(16000, 1000, dtype="<i2").tobytes())
    chunks = list(iter_audio_chunks(str(path)))
    assert [offset for offset, _ in chunks] == [0, 480000, 960000]
    assert [len(audio) for _, audio in chunks] == [480000, 480000, 16000]
    assert all(np.allclose(audio, 1000 / 32768) for _, audio in chunks)


@pytest.mark.parametrize("vocabulary,content,preprocessor", [
    ("vocabulary.txt", "token\n", None),
    ("vocabulary.json", '["token"]', '{"feature_size": 128}'),
])
def test_ct2_validator_accepts_official_small_and_large_model_layouts(tmp_path, vocabulary, content, preprocessor):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "model.bin").write_bytes(b"weights")
    (tmp_path / vocabulary).write_text(content)
    if preprocessor:
        (tmp_path / "preprocessor_config.json").write_text(preprocessor)
    validate_model_files(tmp_path, "faster-whisper")
    (tmp_path / vocabulary).unlink()
    with pytest.raises(ValueError, match="vocabulary is missing"):
        validate_model_files(tmp_path, "faster-whisper")


def test_ct2_validator_checks_optional_preprocessor_integrity(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "model.bin").write_bytes(b"weights")
    (tmp_path / "vocabulary.txt").write_text("token\n")
    (tmp_path / "preprocessor_config.json").write_text("[]")
    with pytest.raises(ValueError, match="preprocessor_config"):
        validate_model_files(tmp_path, "faster-whisper")
