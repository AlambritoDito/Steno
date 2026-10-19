"""Tests for steno.transcriber module."""

import numpy as np
import pytest

from steno.config import Config
from steno.transcriber import Transcriber


def test_transcriber_instantiation():
    """Instantiates without loading the model."""
    t = Transcriber()
    assert t is not None


def test_is_loaded_false_on_init():
    """is_loaded() returns False before first transcription."""
    t = Transcriber()
    assert t.is_loaded() is False


def test_get_model_info_structure():
    """get_model_info() returns dict with model_name and loaded."""
    t = Transcriber()
    info = t.get_model_info()
    assert isinstance(info, dict)
    assert "model_name" in info
    assert "loaded" in info
    assert info["loaded"] is False


@pytest.mark.asyncio
async def test_silence_returns_empty_string():
    """Silence audio (zeros) returns ''."""
    t = Transcriber()
    audio = np.zeros(Config.SAMPLE_RATE, dtype=np.float32)
    result = await t.transcribe(audio)
    assert result == ""


def test_transcriber_has_correct_model_name():
    """Stored model name matches config."""
    t = Transcriber()
    assert t.get_model_info()["model_name"] == Config.MODEL_NAME


# --- v0.2.0: unload_model / set_model ---


def test_unload_model_resets_loaded_state():
    """unload_model() sets is_loaded() to False and _model to None."""
    t = Transcriber()
    t._loaded = True
    t._model = "fake-model-object"
    t.unload_model()
    assert t.is_loaded() is False
    assert t._model is None


def test_set_model_changes_model_name_and_unloads():
    """set_model() with a different name updates _model_name and unloads."""
    t = Transcriber(model_name="model-a")
    t._loaded = True
    t.set_model("model-b")
    assert t._model_name == "model-b"
    assert t.is_loaded() is False


def test_set_model_same_name_is_noop():
    """set_model() with the current name does not unload."""
    t = Transcriber(model_name="model-a")
    t._loaded = True
    t.set_model("model-a")
    assert t._model_name == "model-a"
    assert t.is_loaded() is True


def test_warmup_requires_local_model_without_network(monkeypatch):
    from steno.backends.base import ModelNotDownloadedError
    monkeypatch.setattr(Config, "model_snapshot_path", classmethod(lambda cls, repo: None))
    with pytest.raises(ModelNotDownloadedError):
        Transcriber().warmup()


@pytest.mark.asyncio
async def test_empty_and_invalid_audio_never_load_model():
    transcriber = Transcriber()
    assert await transcriber.transcribe(np.empty(0)) == ""
    with pytest.raises(ValueError):
        await transcriber.transcribe(np.array([np.nan]))
    with pytest.raises(ValueError):
        await transcriber.transcribe(np.ones((10, 2)))
    assert not transcriber.is_loaded()


def test_one_native_model_owner_and_explicit_job_language(monkeypatch, tmp_path):
    from steno import transcriber as module
    from types import SimpleNamespace
    events = []

    def create(name, device):
        return SimpleNamespace(
            load=lambda path: events.append("load"),
            unload=lambda: events.append("unload"),
            transcribe=lambda audio, language: [{"start": 0.0, "end": 1.0, "text": language}],
        )

    monkeypatch.setattr(module, "create_backend", create)
    monkeypatch.setattr(module, "_active_transcriber", None)
    monkeypatch.setattr(Config, "model_snapshot_path", classmethod(lambda cls, repo: tmp_path))
    first = Transcriber(language="es")
    second = Transcriber(language="en")
    first.warmup()
    assert second.transcribe_segments(np.ones(16000))[0]["text"] == "en"
    assert events == ["load", "unload", "load"]
    assert not first.is_loaded()
    assert second.transcribe_segments(np.ones(16000), language="fr")[0]["text"] == "fr"


def test_model_mutation_rejected_while_other_thread_runs(monkeypatch, tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    from steno import transcriber as module
    from steno.backends.base import BackendBusyError

    entered, finish = threading.Event(), threading.Event()
    def infer(audio, language):
        entered.set()
        assert finish.wait(5)
        return []
    monkeypatch.setattr(module, "create_backend", lambda *a, **kw: SimpleNamespace(
        load=lambda path: None, unload=lambda: None, transcribe=infer,
    ))
    monkeypatch.setattr(module, "_active_transcriber", None)
    monkeypatch.setattr(Config, "model_snapshot_path", classmethod(lambda cls, repo: tmp_path))
    transcriber = Transcriber()
    with ThreadPoolExecutor(max_workers=1) as executor:
        task = executor.submit(transcriber.transcribe_segments, np.ones(16000))
        try:
            assert entered.wait(5)
            with pytest.raises(BackendBusyError):
                transcriber.set_model("another-model")
            with pytest.raises(BackendBusyError):
                transcriber.set_language("es")
        finally:
            finish.set()
        task.result()


def test_download_tqdm_protocol_byte_progress_and_validation(monkeypatch, tmp_path):
    from steno.transcriber import _download_with_progress
    from tqdm.contrib.concurrent import thread_map
    model = tmp_path / "snapshot"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "weights.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(Config, "models_dir", classmethod(lambda cls: tmp_path))
    state = {}
    def download(repo, **kwargs):
        assert kwargs["token"] is False
        tracker = kwargs["tqdm_class"]
        assert thread_map(lambda item: item * 2, [1, 2], tqdm_class=tracker, disable=True) == [2, 4]
        with tracker(total=0, unit="B", disable=True, name="huggingface_hub.snapshot_download") as bar:
            bar.total = 100
            bar.refresh()
            bar.update(40)
            assert state["bytes_downloaded"] == 40
            assert state["bytes_total"] == 100
            bar.update(60)
            bar.set_description("Download complete")
        return str(model)
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    _download_with_progress("mlx-community/whisper-tiny", state)
    assert state == {"bytes_downloaded": 100, "bytes_total": 100, "status": "complete"}


def test_download_cancellation_does_not_mark_complete(monkeypatch, tmp_path):
    from steno.transcriber import DownloadCancelled, _download_with_progress
    monkeypatch.setattr(Config, "models_dir", classmethod(lambda cls: tmp_path))
    state = {}
    def download(repo, **kwargs):
        with kwargs["tqdm_class"](total=100, unit="B", disable=True) as bar:
            state["cancel_requested"] = True
            bar.update(10)
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    with pytest.raises(DownloadCancelled):
        _download_with_progress("mlx-community/whisper-tiny", state)
    assert state["status"] == "cancelled"


def test_download_incomplete_model_is_error(monkeypatch, tmp_path):
    from steno.transcriber import _download_with_progress
    monkeypatch.setattr(Config, "models_dir", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **kw: str(tmp_path))
    state = {}
    with pytest.raises(ValueError):
        _download_with_progress("mlx-community/whisper-tiny", state)
    assert state["status"] == "error"


def test_file_iterator_preserves_absolute_offsets_and_trailing_silence(monkeypatch):
    from steno import transcriber as module
    closed = []
    def chunks(*args):
        try:
            yield 3600 * 16000, np.ones(16000)
            yield 3601 * 16000, np.zeros(16000)
        finally:
            closed.append(True)
    monkeypatch.setattr(module, "iter_audio_chunks", chunks)
    transcriber = Transcriber()
    monkeypatch.setattr(transcriber, "transcribe_segments", lambda audio:
                        [{"start": .25, "end": .75, "text": "word"}] if audio.any() else [])
    result = list(transcriber.iter_transcribe_file("unused.wav"))
    assert result == [(3601.0, [{"start": 3600.25, "end": 3600.75, "text": "word"}]), (3602.0, [])]
    assert closed == [True]


def test_file_iterator_resumes_without_inference_for_durable_windows(monkeypatch):
    from steno import transcriber as module
    def chunks(*args):
        yield 0, np.ones(16000)
        yield 16000, np.ones(16000)
        yield 32000, np.ones(16000)
    monkeypatch.setattr(module, "iter_audio_chunks", chunks)
    transcriber = Transcriber()
    calls = []
    def infer(audio):
        calls.append(len(audio))
        return [{"start": 0, "end": len(audio) / 16000, "text": "remaining"}]
    monkeypatch.setattr(transcriber, "transcribe_segments", infer)
    result = list(transcriber.iter_transcribe_file("unused.wav", start_frame=24000))
    assert calls == [8000, 16000]
    assert result == [(2.0, [{"start": 1.5, "end": 2.0, "text": "remaining"}]),
                      (3.0, [{"start": 2.0, "end": 3.0, "text": "remaining"}])]
