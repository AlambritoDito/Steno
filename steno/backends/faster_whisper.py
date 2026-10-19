"""Windows CTranslate2 adapter with explicit CPU and CUDA execution."""

import gc
import importlib
import os
from pathlib import Path

import numpy as np

from steno.backends.base import BackendInferenceError, BackendUnavailableError


class FasterWhisperBackend:
    name = "faster-whisper"

    def __init__(self, device: str = "auto"):
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Device must be auto, cpu, or cuda")
        self.requested_device = device
        self.device = "cpu"
        self.compute_type = "int8"
        self._model = None
        self._capabilities = None
        self._validated = False

    def capabilities(self) -> dict:
        if self._capabilities is None:
            info = {
                "backend": self.name, "available": True, "cuda_available": False,
                "fallback_reason": None, "validated": False,
            }
            try:
                ct2 = importlib.import_module("ctranslate2")
                count = ct2.get_cuda_device_count()
                if count:
                    info["cuda_available"] = "float16" in ct2.get_supported_compute_types("cuda")
                if self.requested_device != "cpu" and info["cuda_available"]:
                    self.device, self.compute_type = "cuda", "float16"
                elif self.requested_device == "cuda":
                    info["available"] = False
                    info["fallback_reason"] = "cuda_unavailable"
                elif self.requested_device == "auto":
                    info["fallback_reason"] = "cuda_unavailable"
            except (ImportError, OSError, RuntimeError) as exc:
                # A CUDA probe failure can still leave a usable CPU runtime.
                try:
                    ct2 = importlib.import_module("ctranslate2")
                    info["available"] = "int8" in ct2.get_supported_compute_types("cpu")
                except (ImportError, OSError, RuntimeError):
                    info["available"] = False
                if self.requested_device == "cuda":
                    info["available"] = False
                info["fallback_reason"] = "cuda_probe_failed" if info["available"] else "runtime_unavailable"
                info["diagnostic"] = str(exc)
            self._capabilities = info
        return {
            **self._capabilities, "device": self.device, "compute_type": self.compute_type,
            "validated": self._validated,
        }

    def _initialize(self, path: Path):
        WhisperModel = importlib.import_module("faster_whisper").WhisperModel
        self._model = WhisperModel(
            str(path), device=self.device, compute_type=self.compute_type,
            cpu_threads=max(1, min(4, os.cpu_count() or 1)), num_workers=1,
            local_files_only=True,
        )
        segments, _ = self._model.transcribe(
            np.zeros(16000, dtype=np.float32), language="en", beam_size=1,
            vad_filter=False, condition_on_previous_text=False,
        )
        list(segments)  # Inference is lazy; consume it to validate cuDNN/cuBLAS too.

    def load(self, model_path: Path) -> None:
        info = self.capabilities()
        if not info["available"]:
            raise BackendUnavailableError(info["fallback_reason"] or "Inference runtime unavailable")
        try:
            self._initialize(model_path)
        except Exception as exc:
            self.unload()
            if self.requested_device != "auto" or self.device != "cuda":
                raise BackendUnavailableError(f"Model initialization failed: {exc}") from exc
            # Only initialization may fall back. Never silently replay a job on CPU.
            self.device, self.compute_type = "cpu", "int8"
            self._capabilities["fallback_reason"] = "cuda_initialization_failed"
            self._capabilities["diagnostic"] = str(exc)
            try:
                self._initialize(model_path)
            except Exception as cpu_exc:
                self.unload()
                raise BackendUnavailableError(f"CPU initialization failed: {cpu_exc}") from cpu_exc
        self._validated = True

    def transcribe(self, audio: np.ndarray, language: str | None) -> list[dict]:
        try:
            segments, _ = self._model.transcribe(
                audio, language=language, beam_size=5, vad_filter=True,
                condition_on_previous_text=False,
            )
            return [
                {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
                for s in segments if s.text.strip()
            ]
        except Exception as exc:
            raise BackendInferenceError(
                f"Transcription failed on {self.device}: {exc}", retry_cpu=self.device == "cuda",
            ) from exc

    def unload(self) -> None:
        if self._model is not None:
            native = getattr(self._model, "model", None)
            if native is not None:
                native.unload_model()
            self._model = None
            gc.collect()
        self._validated = False
